"""Hồ sơ phần cứng: chọn model, backend và thông số chạy theo đúng cái máy đang dùng.

    high - GPU từ đời Turing (sm 7.5) trở lên và >= 8GB VRAM (vd: RTX 4070 Ti SUPER 16GB)
    low  - còn lại (vd: GTX 1060 6GB): AutoAWQ không chạy được trên Pascal, fp16 cực chậm
           trên GTX 10xx, nên mọi thứ đi qua int8 / GGUF lượng tử của llama.cpp.

Ép hồ sơ bằng biến môi trường FILM_PROFILE=high|low (vd: thử đường "low" ngay trên máy mạnh).
Không import torch: hàm này được gọi cả trong server lẫn trong các tiến trình con nhẹ, và đọc
GPU qua nvidia-smi thì không phải tạo CUDA context (~300MB VRAM) chỉ để hỏi thông tin.
"""

import os
import subprocess

import psutil

GB = 1024 ** 3

PROFILES = {
    "high": {
        "whisper_compute": "float16",
        "vision_backend": "transformers_awq",
        "vision_fps": 1.0,
        "vision_max_frames": 48,
        "scene_min_sec": 20,
        "scene_max_sec": 90,
        "llm_ctx": 65536,
        "llm_threads": 8,
        "llm_ubatch": 512,
        "torch_threads": 8,
        "qa_max_tokens": 40000,
        "vram_reserve_gb": 0.8,
    },
    "low": {
        "whisper_compute": "int8",
        "vision_backend": "llamacpp_gguf",
        "vision_fps": 0.5,
        "vision_max_frames": 12,
        "vision_image_side": 448,
        "scene_min_sec": 30,
        "scene_max_sec": 120,
        "llm_ctx": 32768,
        "llm_threads": 6,
        "llm_ubatch": 256,     # kernel ngắn hơn -> GPU vẫn kịp vẽ màn hình, không giật
        "torch_threads": 4,
        "qa_max_tokens": 20000,
        "vram_reserve_gb": 1.0,
    },
}


def gpu_info():
    """[{name, vram_total_gb, vram_used_gb, compute_cap}] qua nvidia-smi; [] nếu không có GPU NVIDIA."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            try:
                gpus.append({"name": parts[0], "vram_total_gb": round(float(parts[1]) / 1024, 1),
                             "vram_used_gb": round(float(parts[2]) / 1024, 1),
                             "compute_cap": float(parts[3])})
            except ValueError:
                continue
    return gpus


def detect():
    gpus = gpu_info()
    gpu = gpus[0] if gpus else None
    forced = os.environ.get("FILM_PROFILE", "").strip().lower()
    if forced in PROFILES:
        name, reason = forced, f"ép bằng FILM_PROFILE={forced}"
    elif gpu and gpu["compute_cap"] >= 7.5 and gpu["vram_total_gb"] >= 8:
        name, reason = "high", f"{gpu['name']} sm {gpu['compute_cap']}, {gpu['vram_total_gb']} GB"
    else:
        name = "low"
        reason = (f"{gpu['name']} sm {gpu['compute_cap']}, {gpu['vram_total_gb']} GB" if gpu
                  else "không thấy GPU NVIDIA")
    return {
        "name": name,
        "reason": reason,
        "gpu": gpu,
        "ram_total_gb": round(psutil.virtual_memory().total / GB, 1),
        "cpu_threads": psutil.cpu_count(logical=True),
        **PROFILES[name],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(detect(), ensure_ascii=False, indent=2))
