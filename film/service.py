"""Lớp nối giữa web_app và các dự án phim.

  - Chạy pipeline trong worker GPU sẵn có của web_app (mỗi lúc một việc dùng GPU): job "film"
    chạy tới khi xong hoặc bị tạm dừng; tạm dừng thì worker rảnh cho việc khác, bấm Tiếp tục
    là xếp một job "film" mới, chạy tiếp từ checkpoint.
  - Lưu dự án (chuyển dữ liệu) khi đang chạy: tạm dừng sau đơn vị đang làm -> chuyển -> tự
    xếp job chạy tiếp ở chỗ mới.
  - Hỏi đáp: giữ llama-server sống giữa các câu hỏi (nạp mất ~1 phút), tự tắt sau 10 phút rảnh
    hoặc ngay khi một việc GPU khác cần chỗ.
  - Hộp thoại chọn thư mục/file của Windows: trình duyệt không cho biết đường dẫn thật trên đĩa,
    nên server tự bật hộp thoại gốc (tkinter) trong một tiến trình con.
"""

import collections
import os
import subprocess
import sys
import threading
import time

import hardware_profile
from film import project as P
from film.pipeline import FilmPipeline

QA_IDLE_SECONDS = 600
VIDEO_FILETYPES = "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.wmv *.mpg *.mpeg *.flv *.ts"


class FilmService:
    def __init__(self, enqueue):
        self.enqueue = enqueue          # enqueue(kind, payload) -> job_id, do web_app cung cấp
        self.lock = threading.Lock()
        self.live = {}                  # project_id -> trạng thái chạy hiện tại
        self.moves = {}                 # project_id -> tiến độ lưu dự án
        self._qa = None                 # (project_id, LlamaServer)
        self._qa_last = 0.0
        threading.Thread(target=self._qa_idle_loop, daemon=True).start()

    # ==========================================
    # CHẠY PIPELINE (gọi từ worker GPU)
    # ==========================================
    def queue_run(self, project_id):
        with self.lock:
            st = self.live.get(project_id)
            if st and st["state"] in ("queued", "running"):
                return st["job_id"]
            job_id = self.enqueue("film", {"project_id": project_id})
            self.live[project_id] = {"state": "queued", "job_id": job_id, "stage": None, "percent": 0.0,
                                     "messages": collections.deque(maxlen=300), "eta_seconds": None,
                                     "pipeline": None, "error": None, "started_at": time.time()}
            return job_id

    def run_pipeline(self, project_id, job_report):
        """Chạy trong worker GPU. Trả về "done" | "paused"."""
        self.release_gpu()
        st = self.live[project_id]
        project = P.find_project(project_id)
        profile = hardware_profile.detect()

        def report(stage, pct, msg):
            st["stage"] = stage
            if pct is not None:
                st["percent"] = pct
            if msg:
                st["messages"].append(msg)
            job_report(msg or "", self._overall(project, stage, st["percent"]))

        pipe = FilmPipeline(project, profile, report)
        st.update({"state": "running", "pipeline": pipe, "profile": profile["name"]})
        try:
            outcome = pipe.run()
            st["state"] = outcome
            return outcome
        except Exception as exc:  # noqa: BLE001
            st.update({"state": "error", "error": str(exc)})
            raise
        finally:
            st["pipeline"] = None

    # Tỉ trọng thời gian của từng bước trong một lần chạy trọn phim (đo trên phim thử):
    # mô tả cảnh chiếm phần lớn, nên thanh tiến độ tổng không "đứng" ở 20% suốt nhiều giờ.
    WEIGHTS = {"prepare": 1, "asr": 4, "scenes": 8, "cards": 70, "synthesis": 17}

    def _overall(self, project, stage, pct):
        done = sum(w for s, w in self.WEIGHTS.items()
                   if s != stage and project.stage(s)["status"] in ("done", "skipped"))
        return round(min(100.0, done + self.WEIGHTS.get(stage, 0) * (pct or 0) / 100), 1)

    def pause(self, project_id, reason="người dùng bấm Tạm dừng"):
        st = self.live.get(project_id)
        if not st:
            return False
        if st["pipeline"] is not None:
            st["pipeline"].pause(reason)
            st["messages"].append(f"[*] Đang tạm dừng: {reason} — dừng sau đơn vị đang làm...")
            return True
        if st["state"] == "queued":
            st["state"] = "cancel_queued"     # worker gặp job này sẽ bỏ qua
            return True
        return False

    def is_cancelled(self, project_id):
        st = self.live.get(project_id)
        return bool(st) and st["state"] == "cancel_queued"

    def live_status(self, project_id):
        st = self.live.get(project_id)
        if not st:
            return None
        pipe = st.get("pipeline")
        return {
            "state": st["state"], "stage": st["stage"], "percent": round(st["percent"] or 0, 1),
            "messages": list(st["messages"])[-80:], "error": st["error"], "job_id": st["job_id"],
            "profile": st.get("profile"),
            "pressure": pipe.monitor.pressure if pipe else None,
        }

    def busy(self):
        """Còn việc dài hạn nào không - đóng tab trình duyệt KHÔNG được tắt app giữa chừng."""
        return (any(s["state"] in ("queued", "running") for s in self.live.values())
                or any(m["status"] == "running" for m in self.moves.values()))

    # ==========================================
    # LƯU DỰ ÁN (chuyển dữ liệu)
    # ==========================================
    def save_as(self, project_id, dest):
        if self.moves.get(project_id, {}).get("status") == "running":
            raise P.ProjectError("Đang lưu dự án này rồi.")
        mv = {"status": "running", "percent": 0, "message": "Chuẩn bị lưu...", "dest": dest, "error": None}
        self.moves[project_id] = mv
        threading.Thread(target=self._save_as, args=(project_id, dest, mv), daemon=True).start()

    def _save_as(self, project_id, dest, mv):
        try:
            st = self.live.get(project_id)
            resume = bool(st) and st["state"] in ("queued", "running")
            if resume:
                mv["message"] = "Tạm dừng sau đơn vị đang làm để lưu dự án..."
                self.pause(project_id, "đang lưu dự án sang thư mục mới")
                while st["state"] in ("running", "queued"):
                    time.sleep(1)
            if self._qa and self._qa[0] == project_id:
                self.release_gpu()          # llama-server đang mở file log trong dự án
            project = P.find_project(project_id)

            def progress(msg, pct=None):
                if msg:
                    mv["message"] = msg
                if pct is not None:
                    mv["percent"] = round(pct, 1)

            moved = P.move_project(project, dest, progress)
            mv.update({"status": "done", "percent": 100, "message": f"Đã lưu vào {moved.root}", "path": moved.root})
            if resume:
                self.queue_run(project_id)
        except Exception as exc:  # noqa: BLE001
            mv.update({"status": "error", "error": str(exc), "message": str(exc)})

    # ==========================================
    # HỎI ĐÁP (gọi từ worker GPU)
    # ==========================================
    def answer(self, project_id, question, job_report):
        from film import qa
        from film.llm_client import LlamaServer
        import resource_guard

        project = P.find_project(project_id)
        profile = hardware_profile.detect()
        if not self._qa or self._qa[0] != project_id:
            self.release_gpu()
            from film.pipeline import NEEDS
            ram, vram = NEEDS[profile["name"]]["synthesis"]
            ok, msg = resource_guard.check_resources(f"{profile['name']}:synthesis", ram, vram,
                                                     profile["vram_reserve_gb"])
            if not ok:
                raise RuntimeError(f"Chưa đủ tài nguyên để nạp model hỏi đáp: {msg}")
            job_report("[*] Nạp model hỏi đáp (khoảng 1 phút, các câu sau sẽ nhanh)...", 5)
            server = LlamaServer(profile, project.log_path("llama-server.log"), project.path("tmp")).start()
            self._qa = (project_id, server)
        self._qa_last = time.time()
        try:
            return qa.answer(project, self._qa[1], question, profile["qa_max_tokens"],
                             lambda m: job_report(f"[*] {m}", None))
        finally:
            self._qa_last = time.time()

    def release_gpu(self):
        """Tắt llama-server của phần hỏi đáp (gọi trước mọi việc GPU khác)."""
        with self.lock:
            if self._qa:
                self._qa[1].stop()
                self._qa = None

    def _qa_idle_loop(self):
        while True:
            time.sleep(30)
            if self._qa and time.time() - self._qa_last > QA_IDLE_SECONDS:
                self.release_gpu()

    def qa_loaded(self):
        return self._qa[0] if self._qa else None


# ==========================================
# HỘP THOẠI CHỌN THƯ MỤC / FILE CỦA WINDOWS
# ==========================================
_DIALOG = r"""
import sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True); root.update()
kind, title, initial = sys.argv[1], sys.argv[2], sys.argv[3] or None
if kind == "folder":
    path = filedialog.askdirectory(title=title, initialdir=initial, mustexist=False)
else:
    path = filedialog.askopenfilename(title=title, initialdir=initial,
                                      filetypes=[("Video", sys.argv[4]), ("Tất cả", "*.*")])
sys.stdout.buffer.write((path or "").encode("utf-8"))
"""


def pick(kind, title, initial=""):
    """Bật hộp thoại gốc của Windows và chờ người dùng chọn. Trả về đường dẫn hoặc "" nếu huỷ."""
    out = subprocess.run([sys.executable, "-c", _DIALOG, kind, title, initial or "", VIDEO_FILETYPES],
                         capture_output=True, timeout=3600)
    return os.path.normpath(out.stdout.decode("utf-8").strip()) if out.stdout.strip() else ""
