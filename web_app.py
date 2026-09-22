"""Giao diện web cho Vision Analyzer.

Chạy:  .venv\\Scripts\\python.exe web_app.py
Sau đó mở http://127.0.0.1:8000

Đặt ảnh/video cần phân tích vào thư mục media_input/ rồi bấm "Quét lại" trên giao diện.
Kết quả được lưu vào vision_storage/<tên file>/ và tải về được dưới dạng .md, .json hoặc .zip.
"""

import datetime
import io
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
import zipfile

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

import vision_analyzer_app as vision
from vision_analyzer_app import LocalVisionAnalyzer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(BASE_DIR, "media_input")
STORAGE_DIR = os.path.join(BASE_DIR, "vision_storage")
STATIC_DIR = os.path.join(BASE_DIR, "static")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".mpg", ".mpeg", ".flv"}

DEFAULT_FPS = 2.0
# Cac muc FPS cho nguoi dung chon. FPS cao = bat duoc nhieu chi tiet theo thoi gian hon,
# nhung so doan va thoi gian chay tang theo ty le thuan.
FPS_CHOICES = [
    {"value": 0.25, "label": "0.25 — rất thưa", "hint": "1 khung / 4 giây. Chỉ hợp video dài, cảnh ít biến động."},
    {"value": 0.5, "label": "0.5 — thưa", "hint": "1 khung / 2 giây. Nhanh, đủ cho tóm tắt tổng quát."},
    {"value": 1.0, "label": "1.0 — cân bằng", "hint": "1 khung / giây. Đủ cho hầu hết video."},
    {"value": 2.0, "label": "2.0 — chi tiết (mặc định)", "hint": "Mặc định của Qwen2.5-VL. Bắt được hành động nhanh."},
    {"value": 4.0, "label": "4.0 — rất chi tiết", "hint": "Chỉ nên dùng cho video ngắn, thời gian chạy tăng gấp đôi."},
]

DEFAULT_IMAGE_PROMPT = "Hãy mô tả chi tiết hình ảnh này, liệt kê các đối tượng chính và chữ (OCR) nếu có."
DEFAULT_VIDEO_PROMPT = (
    "Hãy tóm tắt diễn biến trong video, chia theo các mốc thời gian (timestamp) "
    "và nêu rõ hành động của từng nhân vật/đối tượng."
)

os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

app = FastAPI(title="Vision Analyzer")


# ==========================================
# QUẢN LÝ JOB (một GPU nên chỉ chạy một job mỗi lúc)
# ==========================================
class Job:
    def __init__(self, job_id, filename, media_type, prompt, fps=DEFAULT_FPS):
        self.id = job_id
        self.filename = filename
        self.media_type = media_type
        self.prompt = prompt
        self.fps = fps
        self.status = "queued"          # queued | running | done | error
        self.percent = 0.0
        self.messages = []
        self.result = None
        self.error = None
        self.created_at = datetime.datetime.now().isoformat(timespec="seconds")

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "media_type": self.media_type,
            "status": self.status,
            "percent": round(self.percent, 1),
            "messages": self.messages,
            "result": self.result,
            "error": self.error,
        }


jobs = {}
jobs_lock = threading.Lock()
job_queue = queue.Queue()
analyzer = None
analyzer_lock = threading.Lock()

# ==========================================
# THEO DOI TAB TRINH DUYET: dong tab => tat ung dung
# ==========================================
# Dung WebSocket thay vi heartbeat bang setInterval, vi trinh duyet bop timer cua tab
# chay nen xuong con 1 lan/phut -> heartbeat se bao "da dong tab" nham. WebSocket dong
# ngay lap tuc va dang tin cay khi tab bi dong, bi reload hoac trinh duyet bi crash.
BROWSER_GRACE_SECONDS = 6.0  # du dai de song qua mot lan F5, du ngan de tat nhanh khi dong tab

ws_clients = set()
ws_lock = threading.Lock()
had_any_client = False
clients_empty_since = None


def shutdown_now(reason):
    """Tat ca tien trinh ngay lap tuc, ke ca khi dang phan tich do dang.

    Dung os._exit vi tac vu AI chay trong luong nen va dang giu GPU: khong the dung
    no mot cach lich su. Khi tien trinh chet, driver tu giai phong toan bo VRAM."""
    print(f"\n[!] {reason}", flush=True)
    print("[!] Dang tat ung dung va giai phong GPU ngay lap tuc...", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _browser_watchdog():
    """Tat ung dung khi tab trinh duyet cuoi cung da dong qua thoi gian an han."""
    while True:
        time.sleep(0.5)
        with ws_lock:
            empty_since = clients_empty_since if (had_any_client and not ws_clients) else None
        if empty_since is not None and time.time() - empty_since > BROWSER_GRACE_SECONDS:
            shutdown_now("Tab trinh duyet da dong.")


def _media_type_of(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return None


def _safe_media_path(filename):
    """Chỉ cho phép truy cập file nằm trực tiếp trong media_input/."""
    name = os.path.basename(filename)
    path = os.path.realpath(os.path.join(MEDIA_DIR, name))
    if os.path.dirname(path) != os.path.realpath(MEDIA_DIR) or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file trong media_input: {name}")
    return path


def _safe_result_dir(stem):
    """Chỉ cho phép truy cập thư mục kết quả nằm trực tiếp trong vision_storage/."""
    name = os.path.basename(stem)
    path = os.path.realpath(os.path.join(STORAGE_DIR, name))
    if os.path.dirname(path) != os.path.realpath(STORAGE_DIR) or not os.path.isdir(path):
        raise HTTPException(status_code=404, detail=f"Chưa có kết quả cho: {name}")
    return path


def _result_stem_for(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"[^\w\-. ]", "_", stem).strip() or "khong_ten"


def _worker():
    """Luồng nền xử lý job tuần tự."""
    global analyzer
    while True:
        job_id = job_queue.get()
        with jobs_lock:
            job = jobs.get(job_id)
        if job is None:
            continue

        job.status = "running"
        try:
            def on_progress(message, percent):
                job.messages.append(message.rstrip())
                if percent is not None:
                    job.percent = percent

            with analyzer_lock:
                if analyzer is None:
                    job.messages.append("[*] Lần chạy đầu tiên: đang nạp mô hình lên GPU (mất ~30 giây)...")
                    analyzer = LocalVisionAnalyzer(
                        model_id="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
                        storage_dir=STORAGE_DIR,
                        progress_callback=on_progress,
                    )
                analyzer.progress_callback = on_progress

            media_path = os.path.join(MEDIA_DIR, job.filename)
            if job.media_type == "image":
                result = analyzer.analyze_image(media_path, job.prompt)
            else:
                result = analyzer.analyze_video(media_path, job.prompt, fps=job.fps)

            job.result = {
                "text": result["result"],
                "stem": result["report_stem"],
                "num_segments": result.get("num_segments", 1),
                "keyframes": result.get("keyframes", []),
                "segments": result.get("segments", []),
            }
            job.percent = 100.0
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 - hiển thị mọi lỗi lên giao diện
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.messages.append(f"[X] Lỗi: {job.error}")
        finally:
            if analyzer is not None:
                analyzer.progress_callback = None
            job_queue.task_done()


threading.Thread(target=_worker, daemon=True).start()
threading.Thread(target=_browser_watchdog, daemon=True).start()


@app.websocket("/ws")
async def websocket_liveness(websocket: WebSocket):
    """Kenh song/chet voi tab trinh duyet. Khong truyen du lieu, chi de biet tab con mo."""
    global had_any_client, clients_empty_since
    await websocket.accept()
    with ws_lock:
        ws_clients.add(websocket)
        had_any_client = True
        clients_empty_since = None
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass  # tab dong hoac mat ket noi
    finally:
        with ws_lock:
            ws_clients.discard(websocket)
            if not ws_clients:
                clients_empty_since = time.time()


# ==========================================
# API
# ==========================================
@app.get("/", response_class=HTMLResponse)
def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, encoding="utf-8") as f:
        return f.read()


@app.get("/api/files")
def list_files():
    """Quét media_input/ và trả về mọi file ảnh/video nhận diện được."""
    files = []
    for name in sorted(os.listdir(MEDIA_DIR)):
        path = os.path.join(MEDIA_DIR, name)
        if not os.path.isfile(path):
            continue
        media_type = _media_type_of(name)
        if media_type is None:
            continue

        stem = _result_stem_for(name)
        result_dir = os.path.join(STORAGE_DIR, stem)
        has_result = os.path.isfile(os.path.join(result_dir, f"{stem}.json"))

        info = {
            "filename": name,
            "media_type": media_type,
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
            "stem": stem,
            "has_result": has_result,
            "duration_sec": None,
        }
        if media_type == "video":
            try:
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                if fps > 0 and frames > 0:
                    info["duration_sec"] = round(frames / fps, 1)
            except Exception:
                pass
        files.append(info)

    return {
        "media_dir": MEDIA_DIR,
        "files": files,
        "default_prompts": {"image": DEFAULT_IMAGE_PROMPT, "video": DEFAULT_VIDEO_PROMPT},
        "fps_choices": FPS_CHOICES,
        "default_fps": DEFAULT_FPS,
    }


class AnalyzeRequest(BaseModel):
    filename: str
    prompt: str | None = None
    fps: float | None = None


@app.post("/api/analyze")
def start_analyze(req: AnalyzeRequest):
    _safe_media_path(req.filename)
    name = os.path.basename(req.filename)
    media_type = _media_type_of(name)
    if media_type is None:
        raise HTTPException(status_code=400, detail="Định dạng file không được hỗ trợ.")

    fps = req.fps if req.fps is not None else DEFAULT_FPS
    if not 0.05 <= fps <= 10:
        raise HTTPException(status_code=400, detail="FPS phải nằm trong khoảng 0.05 đến 10.")

    prompt = (req.prompt or "").strip() or (
        DEFAULT_IMAGE_PROMPT if media_type == "image" else DEFAULT_VIDEO_PROMPT
    )
    job = Job(uuid.uuid4().hex[:12], name, media_type, prompt, fps=fps)
    with jobs_lock:
        jobs[job.id] = job
    job_queue.put(job.id)
    return {"job_id": job.id, "queue_size": job_queue.qsize()}


@app.get("/api/estimate")
def estimate(filename: str, fps: float = DEFAULT_FPS):
    """Uoc tinh so doan va thoi gian chay ung voi FPS da chon, truoc khi bat dau."""
    path = _safe_media_path(filename)
    if _media_type_of(os.path.basename(filename)) != "video":
        return {"applicable": False}
    if not 0.05 <= fps <= 10:
        raise HTTPException(status_code=400, detail="FPS phải nằm trong khoảng 0.05 đến 10.")

    # Luon dung ngan sach mac dinh: doc VRAM trong luc dang phan tich se ra so sai lech.
    chunks, frames_per_chunk = vision.plan_video_chunks(
        path, fps, vision.DEFAULT_MAX_PIXELS, vision.DEFAULT_PATCH_BUDGET
    )
    duration = vision.video_duration(path)
    segments = len(chunks)

    # Do duoc tren GPU nay: moi doan ~55 giay, buoc tong hop cuoi ~40 giay.
    seconds = segments * 55 + (40 if segments > 1 else 0)
    return {
        "applicable": True,
        "segments": segments,
        "frames_per_chunk": frames_per_chunk,
        "sampled_frames": int(duration * fps),
        "duration_sec": round(duration, 1),
        "estimated_seconds": seconds,
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy job.")
    return job.to_dict()


@app.get("/api/result/{stem}")
def get_result(stem: str):
    """Đọc lại kết quả đã lưu trước đó."""
    result_dir = _safe_result_dir(stem)
    json_path = os.path.join(result_dir, f"{os.path.basename(stem)}.json")
    if not os.path.isfile(json_path):
        raise HTTPException(status_code=404, detail="Chưa có kết quả đã lưu cho file này.")
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/media/{filename}")
def get_media(filename: str):
    """Trả file gốc để xem trước trên giao diện."""
    return FileResponse(_safe_media_path(filename))


@app.get("/api/result-file/{stem}/{name}")
def get_result_file(stem: str, name: str):
    """Trả file trong thư mục kết quả (keyframe, ảnh gốc đã lưu)."""
    result_dir = _safe_result_dir(stem)
    path = os.path.realpath(os.path.join(result_dir, os.path.basename(name)))
    if os.path.dirname(path) != result_dir or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Không tìm thấy file kết quả.")
    return FileResponse(path)


@app.get("/api/download/{stem}")
def download(stem: str, fmt: str = "md"):
    """Tải kết quả về máy: fmt = md | json | zip."""
    result_dir = _safe_result_dir(stem)
    safe_stem = os.path.basename(stem)

    if fmt in ("md", "json"):
        path = os.path.join(result_dir, f"{safe_stem}.{fmt}")
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail=f"Không có file .{fmt} cho kết quả này.")
        media_type = "text/markdown" if fmt == "md" else "application/json"
        return FileResponse(path, filename=f"{safe_stem}.{fmt}", media_type=media_type)

    if fmt == "zip":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, filenames in os.walk(result_dir):
                for filename in filenames:
                    full = os.path.join(root, filename)
                    zf.write(full, os.path.relpath(full, result_dir))
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{safe_stem}.zip"'},
        )

    raise HTTPException(status_code=400, detail="fmt phải là md, json hoặc zip.")


if __name__ == "__main__":
    if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    print(f"[*] Thư mục ảnh/video đầu vào: {MEDIA_DIR}")
    print(f"[*] Thư mục lưu kết quả:       {STORAGE_DIR}")
    print("[*] Mở giao diện tại: http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
