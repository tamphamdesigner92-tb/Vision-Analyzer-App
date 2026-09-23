"""Điều phối đường ống của một dự án phim: chạy lần lượt các bước, mỗi bước một tiến trình con.

  - Bước nào xong (hoặc được bỏ qua) thì lần sau không chạy lại. Bước đang dở thì chạy tiếp
    từ checkpoint của chính nó (vd: bước mô tả cảnh bỏ qua cảnh đã có file).
  - Trước mỗi bước kiểm tra RAM/VRAM; thiếu thì chờ máy thoáng lại thay vì cố nạp model.
  - Trong lúc chạy, Monitor canh pagefile/RAM: căng thì xin tiến trình con dừng sau đơn vị
    đang làm, chờ máy thoáng, rồi tự chạy tiếp - không huỷ.
  - pause() (người dùng bấm Tạm dừng, hoặc trước khi Lưu dự án sang chỗ khác) cũng dừng sau
    đơn vị đang làm; run() trả về "paused" và lần gọi run() sau chạy tiếp đúng chỗ đó.
"""

import threading
import time

import resource_guard
from film.project import STAGE_LABELS, STAGES, read_json
from film.stage_runner import StageProcess

# Bước đã viết xong (các bước sau thêm dần theo từng mốc).
IMPLEMENTED = ["prepare", "asr", "scenes", "cards", "synthesis"]

# Nhu cầu ước tính (RAM GB, VRAM GB) theo hồ sơ. Sau lần chạy đầu, resource_guard dùng đỉnh đo thật.
# synthesis: đo trên máy chính - llama-server Qwen3-30B-A3B Q4_K_M, ctx 64K, KV q8, --load-mode none
# giữ ~8.2GB RAM thật (working set) khi --fit đã đưa được ~10GB expert lên GPU; máy ít VRAM hơn thì
# phần expert nằm lại RAM nhiều hơn, nên hồ sơ "low" đòi cao hơn.
# VRAM chỉ cần phần tối thiểu (lớp chung + KV): --fit của llama.cpp tự đưa thêm expert lên GPU
# cho tới khi còn đúng khoảng dự phòng, nên không cần đòi đủ 16GB trống.
NEEDS = {
    "high": {"prepare": (1, 0), "asr": (3, 5), "scenes": (2, 0), "cards": (4, 8), "synthesis": (10, 5)},
    "low": {"prepare": (1, 0), "asr": (3, 3), "scenes": (2, 0), "cards": (6, 4.5), "synthesis": (16, 4)},
}


class PipelineStopped(Exception):
    pass


class FilmPipeline:
    def __init__(self, project, profile, report=None):
        self.project = project
        self.profile = profile
        self.report = report or (lambda stage, pct, msg: None)
        self.current_stage = None
        self.current_percent = 0.0
        self._proc = None
        self._pause_reason = None
        self._lock = threading.Lock()
        self.monitor = resource_guard.Monitor(project.log_path("resource.log"))

    # ----- điều khiển từ bên ngoài (luồng khác) -----
    def pause(self, reason="người dùng bấm Tạm dừng"):
        with self._lock:
            self._pause_reason = reason
            if self._proc and self._proc.running:
                self._proc.request_pause(reason)

    @property
    def pause_requested(self):
        return self._pause_reason is not None

    # ----- chạy -----
    def _say(self, stage, pct, msg):
        if pct is not None:
            self.current_percent = pct
        self.report(stage, pct, msg)

    def _skip_reason(self, stage):
        if stage == "asr":
            prep = self.project.stage("prepare").get("result", {})
            src = prep.get("dialogue_source")
            if src in ("srt", "embedded"):
                return f"dùng phụ đề có sẵn ({prep.get('subtitle_file')})"
            if src == "none":
                return "phim không có tiếng"
        return None

    def run(self):
        """Chạy tới hết (trả "done"), hoặc tới khi bị tạm dừng (trả "paused")."""
        self.project.acquire_lock()
        try:
            for stage in STAGES:
                if stage not in IMPLEMENTED:
                    break
                if self.project.stage(stage)["status"] in ("done", "skipped"):
                    continue
                reason = self._skip_reason(stage)
                if reason:
                    self.project.set_stage(stage, "skipped", detail=reason)
                    self._say(stage, 100, f"[{STAGE_LABELS[stage]}] Bỏ qua: {reason}")
                    continue
                outcome = self._run_stage(stage)
                if outcome == "paused":
                    return "paused"
            return "done"
        finally:
            self.monitor.stop()
            self.project.release_lock()

    def _check(self, stage):
        ram, vram = NEEDS[self.profile["name"]][stage]
        return resource_guard.check_resources(f"{self.profile['name']}:{stage}", ram, vram,
                                              self.profile["vram_reserve_gb"])

    def _run_stage(self, stage):
        label = STAGE_LABELS[stage]
        pressure_pauses = 0
        while True:
            if self.pause_requested:
                self.project.set_stage(stage, "paused", detail=self._pause_reason)
                return "paused"
            ok, msg = self._check(stage)
            if not ok:
                self._say(stage, None, f"[{label}] Chưa đủ tài nguyên: {msg}")
                self.project.set_stage(stage, "waiting", detail=msg)
                resource_guard.wait_until_relieved(
                    lambda: self._check(stage), lambda: self.pause_requested,
                    lambda m: self._say(stage, None, m),
                )
                continue

            self.current_stage = stage
            self.project.set_stage(stage, "running", started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            self._say(stage, 0, f"[{label}] Bắt đầu...")

            def on_event(evt, stage=stage, label=label):
                if evt.get("loaded"):
                    self.monitor.mark_loaded()
                if evt["type"] == "progress":
                    self._say(stage, evt.get("percent"), f"[{label}] {evt['message']}" if evt.get("message") else None)
                elif evt["type"] in ("log", "error"):
                    self._say(stage, None, f"[{label}] {evt.get('message')}")

            proc = StageProcess(self.project, stage, on_event, threads=self.profile["torch_threads"])
            with self._lock:
                self._proc = proc
                proc.start()
                if self._pause_reason:
                    proc.request_pause(self._pause_reason)
            self.monitor.start(stage)
            pressure_sent = None
            while proc.running:
                time.sleep(1)
                if self.monitor.pressure and not pressure_sent and not self.pause_requested:
                    pressure_sent = self.monitor.pressure
                    self._say(stage, None, f"[{label}] Máy đang căng ({pressure_sent}) — dừng sau đơn vị đang làm.")
                    proc.request_pause(f"máy căng: {pressure_sent}")
            outcome = proc.wait()
            self.monitor.stop()
            # Ghi đỉnh đo thật cả khi bị dừng vì căng: lần kiểm tra sau đòi đúng mức RAM đã thấy.
            if outcome in ("done", "paused"):
                ram_peak, vram_peak = self.monitor.peak_deltas()
                # llama.cpp --fit tự lấp đầy VRAM còn trống -> mức VRAM nó dùng không phải "nhu cầu".
                resource_guard.record_peak(f"{self.profile['name']}:{stage}", ram_peak,
                                           None if stage == "synthesis" else vram_peak)
            if self.monitor.pagefile_during_load > 256 * resource_guard.MB:
                self._say(stage, None, f"[{label}] Lúc nạp model, Windows đã đẩy "
                                       f"{self.monitor.pagefile_during_load / resource_guard.MB:.0f} MB của ứng dụng "
                                       f"khác xuống pagefile (bình thường khi RAM sát nút).")
            with self._lock:
                self._proc = None
            self.project.reload()

            if outcome == "done":
                result = (proc.final or {}).get("result", {})
                self.project.set_stage(stage, "done", result=result, detail=None)
                self._say(stage, 100, f"[{label}] Xong.")
                return "done"
            if outcome == "paused":
                if self.pause_requested:
                    self.project.set_stage(stage, "paused", detail=self._pause_reason)
                    self._say(stage, None, f"[{label}] Đã tạm dừng.")
                    return "paused"
                # Tự dừng vì máy căng: chờ thoáng rồi quay lại đầu vòng lặp để chạy tiếp.
                # Căng tới lần thứ 2 ngay trong cùng bước thì chính bước này không vừa máy lúc này
                # (chạy lại chỉ nạp model rồi căng tiếp) - dừng hẳn và nói rõ, không lặp vô tận.
                pressure_pauses += 1
                if pressure_pauses >= 2:
                    msg = (f"Bước này liên tục làm máy thiếu RAM ({pressure_sent}). Hãy đóng bớt ứng dụng "
                           f"nặng (trình dựng 3D, máy ảo WSL/Docker, trình duyệt nhiều tab) rồi bấm Tiếp tục.")
                    self.project.set_stage(stage, "paused", detail=msg)
                    self._say(stage, None, f"[{label}] {msg}")
                    return "paused"
                self.project.set_stage(stage, "waiting", detail=pressure_sent)
                resource_guard.wait_until_relieved(
                    lambda: self._check(stage), lambda: self.pause_requested,
                    lambda m: self._say(stage, None, m),
                )
                continue
            err = (proc.final or {}).get("message") or f"tiến trình con thoát bất thường ({outcome})"
            self.project.set_stage(stage, "error", detail=err)
            self._say(stage, None, f"[{label}] LỖI: {err} — xem 10_nhat_ky/{stage}.log")
            raise RuntimeError(f"{label}: {err}")


def dialogue_lines(project):
    return (read_json(project.path("01_loi_thoai", "dialogue.json"), {}) or {}).get("lines", [])
