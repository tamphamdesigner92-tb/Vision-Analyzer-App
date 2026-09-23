"""Phía TIẾN TRÌNH CHA: chạy một bước trong tiến trình con riêng và theo dõi nó.

Mỗi bước một tiến trình con là cách duy nhất chắc chắn trả SẠCH VRAM/RAM sau hàng giờ chạy:
tiến trình thoát thì driver thu hồi toàn bộ bộ nhớ, không còn phân mảnh hay rò rỉ dồn lại
giữa các bước. Tiến trình con chạy ở mức ưu tiên BELOW_NORMAL để máy vẫn mượt khi người
dùng làm việc khác. Giao thức xem stage_io.py.
"""

import json
import os
import subprocess
import sys
import threading

from film.project import APP_DIR, write_json_atomic
from film.stage_io import EXIT_DONE, EXIT_PAUSED, control_path

_PRIORITY = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class StageProcess:
    def __init__(self, project, stage, on_event, extra_args=(), threads=8):
        self.project = project
        self.stage = stage
        self.on_event = on_event
        self.extra_args = list(extra_args)
        self.threads = threads
        self.proc = None
        self.final = None          # sự kiện done/paused/error cuối cùng
        self._reader = None

    def start(self):
        ctl = control_path(self.project, self.stage)
        if os.path.exists(ctl):
            os.remove(ctl)
        env = dict(os.environ)
        env.update({
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": str(self.threads),
            "MKL_NUM_THREADS": str(self.threads),
            # Luồng con không mở cửa sổ trình duyệt/console riêng, không đụng biến của server.
        })
        log_path = self.project.log_path(f"{self.stage}.log")
        self._log = open(log_path, "a", encoding="utf-8", errors="replace")
        self._log.write(f"\n===== bắt đầu bước {self.stage} (pid cha {os.getpid()}) =====\n")
        self._log.flush()
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-m", f"film.stage_{self.stage}", "--project", self.project.root,
             *self.extra_args],
            cwd=APP_DIR, env=env, stdout=subprocess.PIPE, stderr=self._log,
            creationflags=_PRIORITY | _NO_WINDOW,
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        for raw in self.proc.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except ValueError:
                self._log.write(line + "\n")
                continue
            if evt.get("type") in ("done", "paused", "error"):
                self.final = evt
            try:
                self.on_event(evt)
            except Exception:  # noqa: BLE001 - lỗi hiển thị không được làm chết luồng đọc
                pass

    def request_pause(self, reason):
        """Xin con dừng sau đơn vị đang làm (con tự lưu xong rồi mới thoát)."""
        write_json_atomic(control_path(self.project, self.stage), {"action": "pause", "reason": reason})

    def kill(self):
        """Tắt hẳn cả cây tiến trình (bước + mọi tiến trình con của nó như llama-server)."""
        if self.proc and self.proc.poll() is None:
            try:
                import psutil
                for child in psutil.Process(self.proc.pid).children(recursive=True):
                    child.kill()
            except Exception:  # noqa: BLE001
                pass
            self.proc.kill()

    def wait(self, timeout=None):
        """Chờ con thoát. Trả về "done" | "paused" | "error" | "killed"."""
        code = self.proc.wait(timeout=timeout)
        if self._reader:
            self._reader.join(timeout=5)
        # Bước chết ngang (bị kill, crash) thì llama-server của nó thành mồ côi - dọn ngay.
        from film.llm_client import kill_orphan
        if kill_orphan(self.project.path("tmp")):
            self._log.write("[!] Đã tắt llama-server mồ côi của bước này.\n")
        self._log.write(f"===== kết thúc bước {self.stage}, mã thoát {code} =====\n")
        self._log.close()
        try:
            os.remove(control_path(self.project, self.stage))
        except OSError:
            pass
        if code == EXIT_DONE:
            return "done"
        if code == EXIT_PAUSED:
            return "paused"
        if self.final and self.final.get("type") == "error":
            return "error"
        return "killed"

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None
