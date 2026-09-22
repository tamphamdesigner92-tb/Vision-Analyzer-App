"""Trạng thái hệ thống — để giao diện luôn nói rõ đang xảy ra chuyện gì.

Ứng dụng này có ba mô hình nặng, thay nhau lên xuống GPU, và lần đầu dùng mỗi mô hình
còn phải tải hàng GB từ mạng. Nếu không báo gì, người dùng nhìn thấy một thanh tiến độ
đứng yên nhiều phút và không biết là đang tải, đang nạp, hay đã treo. Module này giữ
trạng thái đó ở một chỗ để mọi nơi cùng đọc.

Bốn trạng thái của một mô hình:
    trong     - không nằm trên GPU (chưa dùng, hoặc đã nhường chỗ cho mô hình khác)
    dang_tai  - đang tải trọng số từ HuggingFace về đĩa (chỉ xảy ra lần đầu)
    dang_nap  - trọng số đã có trên đĩa, đang đưa lên GPU
    san_sang  - đang nằm trên GPU, dùng được ngay
"""

import os
import threading
import time

import torch
from huggingface_hub import constants as hf_constants

_lock = threading.Lock()
_models = {"vision": "trong", "reranker": "trong"}
_download = None


# ==========================================
# TRẠNG THÁI MÔ HÌNH
# ==========================================
def set_model_state(name, state):
    with _lock:
        _models[name] = state


def get_models():
    with _lock:
        return dict(_models)


def vram():
    """VRAM đang dùng trên toàn GPU (kể cả tiến trình khác, ví dụ sidecar)."""
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {
        "used_mb": round((total - free) / (1024 ** 2)),
        "total_mb": round(total / (1024 ** 2)),
    }


# ==========================================
# TIẾN ĐỘ TẢI TRỌNG SỐ
# ==========================================
# Đo bằng cách so dung lượng thư mục cache với tổng dung lượng repo, thay vì móc vào
# thanh tiến độ của huggingface_hub. Cách này không phụ thuộc chi tiết cài đặt bên trong
# thư viện (vốn đổi giữa các phiên bản) và đúng cho cả tiến trình sidecar ở venv khác.
def _repo_cache_dir(repo_id):
    return os.path.join(hf_constants.HF_HUB_CACHE, "models--" + repo_id.replace("/", "--"))


def _dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass  # file tạm của huggingface có thể biến mất giữa chừng
    return total


def repo_total_bytes(repo_id):
    """Tổng dung lượng repo trên HuggingFace; None nếu không hỏi được (mất mạng...)."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, files_metadata=True)
        return sum(f.size or 0 for f in info.siblings) or None
    except Exception:  # noqa: BLE001 - không tra được thì chỉ mất phần trăm, không hỏng gì
        return None


def set_download(model_id, downloaded, total):
    global _download
    with _lock:
        _download = {
            "model": model_id,
            "downloaded_mb": round(downloaded / (1024 ** 2)),
            "total_mb": round(total / (1024 ** 2)) if total else None,
            "percent": round(100 * downloaded / total, 1) if total else None,
        }


def clear_download():
    global _download
    with _lock:
        _download = None


def get_download():
    with _lock:
        return dict(_download) if _download else None


class DownloadWatcher:
    """Theo dõi tiến độ tải một repo trong lúc from_pretrained() đang chạy.

    Dùng như context manager. Nếu trọng số đã có sẵn trong cache thì nó im lặng —
    không báo "đang tải" nhầm khi thực ra chỉ mất vài giây nạp từ đĩa.
    """

    def __init__(self, model_id, report=None, interval=2.0):
        self.model_id = model_id
        self.report = report
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        path = _repo_cache_dir(self.model_id)
        total = repo_total_bytes(self.model_id)
        last_reported = -10.0
        while not self._stop.is_set():
            size = _dir_size(path) if os.path.isdir(path) else 0
            if total and size < total * 0.98:
                set_download(self.model_id, size, total)
                percent = 100 * size / total
                if self.report and percent - last_reported >= 5:
                    last_reported = percent
                    self.report(
                        f"[*] Đang tải trọng số {self.model_id}: "
                        f"{size / 1024**3:.1f}/{total / 1024**3:.1f} GB ({percent:.0f}%)",
                        None,
                    )
            self._stop.wait(self.interval)
        clear_download()

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        clear_download()
        return False
