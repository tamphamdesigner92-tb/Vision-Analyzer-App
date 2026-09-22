"""Sidecar nhúng đa phương thức — Qwen3-VL-Embedding-2B.

VÌ SAO PHẢI LÀ MỘT TIẾN TRÌNH RIÊNG:
Model này đòi transformers>=4.57 và torch 2.8, trong khi ứng dụng chính bị khoá ở
transformers 4.51.3 / torch 2.5.1 vì AutoAWQ (xem requirements.txt — nâng lên là vỡ
Qwen2.5-VL-7B-AWQ). Hai bộ thư viện không thể sống chung trong một tiến trình Python.
Nên sidecar chạy bằng venv riêng (.venv-embed), nói chuyện với app chính qua HTTP.

Khác với reranker (chỉ đọc chữ), model này nhúng THẲNG ảnh/video và chữ vào cùng một
không gian vector, nên so khớp chỉ là phép nhân ma trận — dùng để lọc nhanh top-K ứng
viên trước khi reranker chấm kỹ từng cặp.

Chạy:  .venv-embed\\Scripts\\python.exe embedding_sidecar\\server.py
"""

import importlib.util
import os
import sys
import threading

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "Qwen/Qwen3-VL-Embedding-2B")
PORT = int(os.environ.get("EMBEDDING_PORT", "8011"))


def _patch_qwen_vl_utils_file_uri_bug():
    """Va mot loi tuong thich giua qwen_vl_utils va vendor script cua Qwen3-VL-Embedding.

    qwen_vl_utils._read_video_torchvision() chi strip tien to "file://" khi
    torchvision < 0.19.0 (vet tich lich su, xem source cua no). Nhung sidecar nay bat
    buoc dung torchvision >= 0.22 (vi transformers>=4.57 / torch 2.8), con vendor script
    cua Qwen3-VL-Embedding (qwen3_vl_embedding.py, tai ve tu HF hub) lai LUON them tien
    to "file://" vao MOI duong dan local, bat ke phien ban torchvision. Ket qua: torchvision
    nhan duong dan "file://C:\\..." nhu mot chuoi khong hop le, doc that bai va tra ve info
    thieu key 'video_fps' -> KeyError: 'video_fps' cho MOI video local, khong loai tru.

    Xac nhan bang thuc nghiem: tat ca 7/7 video that trong media_input deu loi giong het
    nhau khi goi dung quy uoc cua vendor. Va o day bang cach strip "file://" TRUOC khi giao
    cho ham doc that su - khong dung duoc den viec sua qwen_vl_utils (dependency ngoai)."""
    from qwen_vl_utils import vision_process

    original = vision_process._read_video_torchvision

    def patched(ele, *args, **kwargs):
        video = ele.get("video")
        if isinstance(video, str) and video.startswith("file://"):
            ele = {**ele, "video": video[len("file://"):]}
        return original(ele, *args, **kwargs)

    vision_process._read_video_torchvision = patched
    vision_process.VIDEO_READER_BACKENDS["torchvision"] = patched


_patch_qwen_vl_utils_file_uri_bug()

app = FastAPI(title="Qwen3-VL Embedding sidecar")
_model = None
_lock = threading.Lock()

# Trang thai de app chinh hoi va hien len giao dien web. Muc dich: nguoi dung khong phai
# mo cua so console nay ra nhin moi biet sidecar dang tai hay dang nhung den dau.
_state = {"state": "trong", "message": "Chưa nạp mô hình", "done": 0, "total": 0}
_state_lock = threading.Lock()


def _set_state(state, message, done=0, total=0):
    with _state_lock:
        _state.update(state=state, message=message, done=done, total=total)
    print(f"[{state}] {message}", flush=True)


def _load_embedder_class():
    """Lớp Qwen3VLEmbedder nằm trong chính repo của model, không có trong transformers.

    Tải file scripts/qwen3_vl_embedding.py về cache của huggingface rồi nạp như module
    — cách này bám đúng phiên bản chính chủ thay vì chép tay code vào đây rồi lệch dần.
    """
    path = hf_hub_download(MODEL_ID, "scripts/qwen3_vl_embedding.py")
    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["qwen3_vl_embedding"] = module
    spec.loader.exec_module(module)
    return module.Qwen3VLEmbedder


def _get_model():
    global _model
    if _model is None:
        embedder_cls = _load_embedder_class()
        # Lan dau se tai ~4GB trong so; watcher bao tien do de giao dien khong dung im.
        watcher = _DownloadWatcher(MODEL_ID)
        watcher.start()
        try:
            _set_state("dang_nap", f"Đang nạp {MODEL_ID} lên GPU")
            _model = embedder_cls(
                model_name_or_path=MODEL_ID,
                torch_dtype=torch.float16,
                # Mac dinh cua vendor la fps=1, max_frames=64: voi video do phan giai
                # thuc te (1080x1920) da tung lam OutOfMemoryError (thu cap phat 13.44 GiB)
                # o attention cua vision tower - cung mot dang O(N^2) da gap voi
                # Qwen2.5-VL. Ha manh max_frames la don bay chac chan nhat vi no duoc
                # qwen_vl_utils chuyen tiep dung (khac voi total_pixels: vendor lam rot
                # tham so nay khi truyen video la duong dan chuoi, xem _preprocess_inputs).
                fps=0.5,
                max_frames=8,
            )
        finally:
            watcher.stop()
        _set_state("san_sang", "Mô hình đã sẵn sàng trên GPU")
    return _model


class _DownloadWatcher(threading.Thread):
    """Do tien do tai bang cach so dung luong cache voi tong dung luong repo.

    Khong moc vao thanh tien do cua huggingface_hub vi chi tiet do doi theo phien ban;
    do dung luong thu muc thi luon dung. Im lang neu trong so da co san trong cache."""

    def __init__(self, model_id, interval=2.0):
        super().__init__(daemon=True)
        self.model_id = model_id
        self.interval = interval
        # KHONG duoc dat ten la _stop: Thread._stop() la phuong thuc noi bo ma join()
        # goi den, ghi de no bang mot Event se lam join() no TypeError.
        self._stop_event = threading.Event()

    def _cache_dir(self):
        from huggingface_hub import constants
        return os.path.join(constants.HF_HUB_CACHE, "models--" + self.model_id.replace("/", "--"))

    def _total_bytes(self):
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(self.model_id, files_metadata=True)
            return sum(f.size or 0 for f in info.siblings) or 0
        except Exception:  # noqa: BLE001
            return 0

    def _dir_size(self, path):
        total = 0
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
        return total

    def run(self):
        path, total = self._cache_dir(), self._total_bytes()
        while not self._stop_event.is_set():
            size = self._dir_size(path) if os.path.isdir(path) else 0
            if total and size < total * 0.98:
                _set_state(
                    "dang_tai",
                    f"Đang tải trọng số: {size / 1024**3:.1f}/{total / 1024**3:.1f} GB",
                    size, total,
                )
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=3)


class EmbedItem(BaseModel):
    text: str | None = None
    image: str | None = None       # đường dẫn tuyệt đối hoặc URL
    video: str | None = None
    instruction: str | None = None


class EmbedRequest(BaseModel):
    items: list[EmbedItem]
    batch_size: int = 4


@app.get("/health")
def health():
    with _state_lock:
        snapshot = dict(_state)
    return {"ok": True, "model": MODEL_ID, "loaded": _model is not None, **snapshot}


@app.post("/embed")
def embed(req: EmbedRequest):
    if not req.items:
        return {"embeddings": []}
    for item in req.items:
        for path in (item.image, item.video):
            # Ảnh/video phải có thật: để model tự vấp thì lỗi rất khó đọc.
            if path and not path.startswith(("http://", "https://")) and not os.path.isfile(path):
                raise HTTPException(status_code=400, detail=f"Không tìm thấy file: {path}")

    with _lock:
        model = _get_model()
        out = []
        payload = [i.model_dump(exclude_none=True) for i in req.items]
        for start in range(0, len(payload), req.batch_size):
            chunk = payload[start:start + req.batch_size]
            _set_state(
                "dang_nhung",
                f"Đang nhúng {min(start + len(chunk), len(payload))}/{len(payload)} mục",
                start + len(chunk), len(payload),
            )
            vectors = model.process(chunk)
            if isinstance(vectors, tuple):
                vectors = vectors[0]
            out.extend(vectors.float().cpu().tolist())
        _set_state("san_sang", "Mô hình đã sẵn sàng trên GPU")
    return {"embeddings": out, "dim": len(out[0]) if out else 0}


@app.post("/unload")
def unload():
    """Trả VRAM lại cho app chính. GPU 16GB không gánh nổi ba mô hình cùng lúc."""
    global _model
    with _lock:
        if _model is not None:
            del _model
            _model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    _set_state("trong", "Đã nhả VRAM, mô hình không còn trên GPU")
    return {"ok": True, "loaded": False}


def _port_occupant(port):
    """Kiem tra cong da bi chiem chua, va neu roi thi boi ai.

    Tra ve None neu cong con trong, "sidecar" neu chinh mot sidecar khac dang chay,
    hoac "khac" neu la mot chuong trinh la. Phan biet duoc hai truong hop nay la quan
    trong: mot ban sidecar chay san thi khong phai loi gi ca, chi can dong cua so nay.
    """
    import socket
    import urllib.request

    with socket.socket() as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", port)) != 0:
            return None

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            import json as _json
            if _json.load(resp).get("model"):
                return "sidecar"
    except Exception:  # noqa: BLE001
        pass
    return "khac"


if __name__ == "__main__":
    if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    occupant = _port_occupant(PORT)
    if occupant == "sidecar":
        print(f"[i] Da co mot sidecar chay san tai http://127.0.0.1:{PORT}")
        print("[i] Khong can mo them ban thu hai - ung dung chinh dang dung ban do roi.")
        print("[i] Dong cua so nay lai la duoc.")
        sys.exit(0)
    if occupant == "khac":
        print(f"[X] Cong {PORT} dang bi mot chuong trinh khac chiem (khong phai sidecar).")
        print()
        print("    Tim xem la tien trinh nao:")
        print(f"        netstat -ano | findstr :{PORT}")
        print("        (cot cuoi cung la PID, tat bang: taskkill /PID <pid> /F)")
        print()
        other = PORT + 1
        print("    Hoac doi sidecar sang cong khac roi chay lai:")
        print(f"        set EMBEDDING_PORT={other}")
        print(f"    (nho dat cung gia tri cho ung dung chinh: set EMBEDDING_SIDECAR_URL=http://127.0.0.1:{other})")
        sys.exit(1)

    print(f"[*] Sidecar nhung da phuong thuc: {MODEL_ID}")
    print(f"[*] Lang nghe tai http://127.0.0.1:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
