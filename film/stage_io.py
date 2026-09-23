"""Phía TIẾN TRÌNH CON của mỗi bước: báo tiến độ, nhận lệnh tạm dừng, thoát sạch.

Giao thức với tiến trình cha (stage_runner.py):
  - stdout: mỗi dòng một JSON {"type": progress|log|done|paused|error, ...}. Để chắc chắn
    không thư viện nào (transformers, ctranslate2, tqdm…) in lẫn rác vào kênh này, fd 1 bị
    chuyển sang stderr ngay đầu tiến trình và giao thức đi qua một bản sao riêng của stdout.
  - stderr: log tự do, cha ghi thẳng vào 10_nhat_ky/<bước>.log.
  - Lệnh tạm dừng: cha ghi tmp/control_<bước>.json; con kiểm tra giữa hai đơn vị việc (vd:
    giữa hai cảnh) qua should_pause() và thoát với mã 3 sau khi đã lưu xong đơn vị đang làm.
  - Mã thoát: 0 xong, 3 tạm dừng, 1 lỗi.
"""

import argparse
import json
import os
import sys
import traceback

EXIT_DONE, EXIT_ERROR, EXIT_PAUSED = 0, 1, 3

_proto = None


def _open_protocol():
    global _proto
    if _proto is None:
        _proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
        os.dup2(2, 1)                      # print()/thư viện C in ra fd 1 -> đi vào log
        sys.stdout = sys.stderr
    return _proto


def emit(kind, **data):
    _open_protocol().write(json.dumps({"type": kind, **data}, ensure_ascii=False) + "\n")


def progress(percent=None, message=None, **extra):
    emit("progress", percent=None if percent is None else round(float(percent), 2),
         message=message, **extra)


def log(message):
    emit("log", message=message)


def model_loaded(percent=None, message=None):
    """Báo cho cha: đã nạp xong model. Từ đây cha mới bắt đầu canh pagefile (lúc nạp, Windows
    đẩy bớt trang nhàn rỗi của ứng dụng khác xuống pagefile là chuyện bình thường)."""
    progress(percent, message, loaded=True)


class PauseRequested(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def control_path(project, stage):
    return project.path("tmp", f"control_{stage}.json")


def should_pause(project, stage):
    try:
        with open(control_path(project, stage), encoding="utf-8") as f:
            ctl = json.load(f)
    except (OSError, ValueError):
        return None
    if ctl.get("action") != "pause":
        return None
    return ctl.get("reason") or "tạm dừng"


def check_pause(project, stage):
    """Gọi giữa hai đơn vị việc. Có lệnh tạm dừng thì ném PauseRequested để thoát sạch."""
    reason = should_pause(project, stage)
    if reason:
        raise PauseRequested(reason)


def limit_threads(n):
    """Giới hạn số luồng CPU của các thư viện số học - phải gọi TRƯỚC khi import torch."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(n))


def run_stage(stage, main, add_args=None):
    """Khung chung cho `python -m film.stage_xxx --project <thư mục>`."""
    _open_protocol()
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    if add_args:
        add_args(ap)
    args = ap.parse_args()

    import hardware_profile
    profile = hardware_profile.detect()
    limit_threads(profile["torch_threads"])

    from film.project import Project
    project = Project(args.project)
    try:
        result = main(project, profile, args) or {}
    except PauseRequested as exc:
        emit("paused", reason=exc.reason)
        sys.exit(EXIT_PAUSED)
    except Exception as exc:  # noqa: BLE001 - mọi lỗi đều phải báo lên giao diện
        traceback.print_exc()
        emit("error", message=f"{type(exc).__name__}: {exc}")
        sys.exit(EXIT_ERROR)
    emit("done", result=result)
    sys.exit(EXIT_DONE)
