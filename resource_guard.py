"""Giữ máy không nghẽn RAM / không lấn sang pagefile trong lúc chạy phim dài nhiều giờ.

Windows không tắt được pagefile một cách an toàn, nên cách làm là:
  - Trước mỗi bước: đủ RAM + VRAM trống thì mới chạy (need_* của bước, hiệu chỉnh dần bằng số
    đo thật lưu ở .ram_profile.json).
  - Trong lúc chạy: cứ 2 giây đo RAM trống + pagefile + VRAM, ghi ra resource.log. Pagefile
    tăng quá ngưỡng hoặc RAM trống xuống thấp thì bật cờ "căng" - đường ống sẽ TẠM DỪNG sau
    đơn vị đang làm (một cảnh), không huỷ, và tự chạy tiếp khi máy thoáng lại.

Đọc VRAM bằng nvidia-smi, không qua torch: server không phải tạo CUDA context chỉ để đo.
"""

import datetime
import json
import os
import threading
import time

import psutil

from hardware_profile import gpu_info

GB = 1024 ** 3
MB = 1024 ** 2
PAGEFILE_GROWTH_LIMIT = float(os.environ.get("PAGEFILE_GUARD_MB", "512")) * MB
RAM_FLOOR = float(os.environ.get("RAM_FLOOR_GB", "2")) * GB           # chừa khi KIỂM TRA TRƯỚC khi nạp
RAM_CRITICAL = float(os.environ.get("RAM_CRITICAL_GB", "0.75")) * GB  # ngưỡng báo căng khi ĐANG chạy

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_PATH = os.path.join(APP_DIR, ".ram_profile.json")


def snapshot():
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    gpu = (gpu_info() or [None])[0]
    return {
        "ram_total": vm.total,
        "ram_available": vm.available,
        "pagefile_used": sw.used,
        "pagefile_total": sw.total,
        "vram_total": int(gpu["vram_total_gb"] * GB) if gpu else None,
        "vram_used": int(gpu["vram_used_gb"] * GB) if gpu else None,
    }


def status_chip():
    """Số liệu gọn cho thanh trạng thái của giao diện."""
    s = snapshot()
    return {
        "ram_total_gb": round(s["ram_total"] / GB, 1),
        "ram_available_gb": round(s["ram_available"] / GB, 1),
        "pagefile_used_gb": round(s["pagefile_used"] / GB, 1),
    }


# ----- nhu cầu từng bước (ước tính, rồi hiệu chỉnh bằng đỉnh đo thật) -----
def _load_profile():
    try:
        with open(PROFILE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def record_peak(key, ram_bytes=None, vram_bytes=None):
    data = _load_profile()
    entry = data.setdefault(key, {})
    if ram_bytes and ram_bytes > entry.get("ram", 0):
        entry["ram"] = int(ram_bytes)
    if vram_bytes and vram_bytes > entry.get("vram", 0):
        entry["vram"] = int(vram_bytes)
    tmp = PROFILE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, PROFILE_PATH)


def check_resources(key, need_ram_gb, need_vram_gb, vram_reserve_gb=0.8):
    """(được chạy?, thông điệp). Lấy số lớn hơn giữa ước tính và đỉnh đo thật lần trước."""
    measured = _load_profile().get(key, {})
    need_ram = max(need_ram_gb * GB, measured.get("ram", 0))
    need_vram = max(need_vram_gb * GB, measured.get("vram", 0)) + vram_reserve_gb * GB
    s = snapshot()
    problems = []
    if s["ram_available"] - need_ram < RAM_FLOOR:
        problems.append(f"RAM trống {s['ram_available'] / GB:.1f} GB, bước này cần ~{need_ram / GB:.1f} GB "
                        f"(+{RAM_FLOOR / GB:.0f} GB chừa cho Windows)")
    if s["vram_total"] is not None and need_vram_gb > 0:
        free_vram = s["vram_total"] - s["vram_used"]
        if free_vram < need_vram:
            problems.append(f"VRAM trống {free_vram / GB:.1f} GB, bước này cần ~{need_vram / GB:.1f} GB")
    if problems:
        return False, "; ".join(problems) + ". Hãy đóng bớt ứng dụng nặng."
    return True, None


class Monitor:
    """Luồng nền đo tài nguyên trong lúc một bước đang chạy."""

    def __init__(self, log_path, interval=2.0):
        self.log_path = log_path
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.pressure = None          # None hoặc chuỗi mô tả vì sao "căng"
        self.peak_ram_used = 0
        self.peak_vram_used = 0
        self._baseline = None
        self._pagefile_ref = None
        self.pagefile_during_load = 0

    def start(self, label):
        self.stop()
        self._stop.clear()
        self.pressure = None
        s = snapshot()
        self._baseline = s
        # Mốc pagefile chỉ có sau khi bước báo "đã nạp xong model" (mark_loaded): lúc nạp,
        # Windows đẩy bớt trang nhàn rỗi của ứng dụng khác xuống pagefile - bình thường, không
        # phải dấu hiệu máy đang swap liên tục. Thứ cần bắt là pagefile TIẾP TỤC tăng sau đó.
        self._pagefile_ref = None
        self.pagefile_during_load = 0
        self.peak_ram_used = s["ram_total"] - s["ram_available"]
        self.peak_vram_used = s["vram_used"] or 0
        self._label = label
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def mark_loaded(self):
        s = snapshot()
        self._pagefile_ref = s["pagefile_used"]
        self.pagefile_during_load = max(0, s["pagefile_used"] - self._baseline["pagefile_used"])

    def _loop(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as log:
            if log.tell() == 0:
                log.write("thoi_gian,buoc,ram_trong_gb,pagefile_gb,vram_dung_gb,trang_thai\n")
            while not self._stop.wait(self.interval):
                s = snapshot()
                self.peak_ram_used = max(self.peak_ram_used, s["ram_total"] - s["ram_available"])
                self.peak_vram_used = max(self.peak_vram_used, s["vram_used"] or 0)
                reason = None
                if self._pagefile_ref is not None:
                    grown = s["pagefile_used"] - self._pagefile_ref
                    if grown > PAGEFILE_GROWTH_LIMIT:
                        reason = (f"pagefile tăng thêm {grown / MB:.0f} MB sau khi đã nạp xong model "
                                  f"(ngưỡng {PAGEFILE_GROWTH_LIMIT / MB:.0f} MB)")
                if reason is None and s["ram_available"] < RAM_CRITICAL:
                    reason = f"RAM trống chỉ còn {s['ram_available'] / GB:.1f} GB"
                self.pressure = reason
                log.write(
                    f"{datetime.datetime.now().isoformat(timespec='seconds')},{self._label},"
                    f"{s['ram_available'] / GB:.2f},{s['pagefile_used'] / GB:.2f},"
                    f"{(s['vram_used'] or 0) / GB:.2f},{'CANG: ' + reason if reason else 'ok'}\n"
                )
                log.flush()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def peak_deltas(self):
        """(RAM, VRAM) tăng thêm cao nhất so với lúc bắt đầu bước = nhu cầu thật của bước đó."""
        if not self._baseline:
            return None, None
        base_ram = self._baseline["ram_total"] - self._baseline["ram_available"]
        base_vram = self._baseline["vram_used"] or 0
        return max(0, self.peak_ram_used - base_ram), max(0, self.peak_vram_used - base_vram)


def wait_until_relieved(check, is_cancelled, report, poll=10, calm_for=30):
    """Chờ máy thoáng lại liên tục `calm_for` giây rồi mới cho chạy tiếp."""
    calm_since = None
    while not is_cancelled():
        ok, msg = check()
        if ok:
            calm_since = calm_since or time.time()
            if time.time() - calm_since >= calm_for:
                return True
        else:
            calm_since = None
            report(f"[!] Đang chờ máy thoáng lại: {msg}")
        time.sleep(poll)
    return False
