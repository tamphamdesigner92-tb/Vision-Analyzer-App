"""Backend thị giác cho bước mô tả cảnh - một giao diện, nhiều cách chạy theo hồ sơ phần cứng.

    transformers_awq  - Qwen2.5-VL-7B-Instruct-AWQ qua transformers (máy chính, hồ sơ "high").
                        Bọc LocalVisionAnalyzer có sẵn: dùng lại phần vá lm_head AWQ, truyền đúng
                        FPS lấy mẫu, và ngân sách patch theo VRAM trống.
    llamacpp_gguf     - Qwen2.5-VL-7B GGUF Q4_K_M + mmproj qua llama-server (hồ sơ "low", vd GTX
                        1060 6GB: AutoAWQ không chạy trên Pascal, fp16 lại cực chậm). --fit tự để
                        những lớp không vừa VRAM lại RAM. llama.cpp chưa nhận video nên khung hình
                        được gửi thành NHIỀU ẢNH, mỗi ảnh có nhãn "[giây X]" đứng trước để model
                        vẫn biết thứ tự và mốc thời gian.

Giao diện chung: max_frames(), describe(frames, sample_fps, prompt, max_new_tokens),
text(prompt, max_new_tokens), close().
"""

import base64
import io
import os

VL_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
VL_GGUF_REPO = "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF"
VL_GGUF_FILE = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
VL_MMPROJ_FILE = "mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf"
VL_CTX = 16384       # 12 ảnh x ~256 token + lời thoại + câu trả lời, dư nhiều


class TransformersAWQBackend:
    name = "transformers_awq"
    model = VL_MODEL_ID

    def __init__(self, profile, work_dir, log_dir):
        from vision_analyzer_app import DEFAULT_MAX_PIXELS, LocalVisionAnalyzer
        self.max_pixels = DEFAULT_MAX_PIXELS
        self.analyzer = LocalVisionAnalyzer(model_id=VL_MODEL_ID, storage_dir=work_dir)

    def max_frames(self):
        return self.analyzer.frames_for_budget(self.max_pixels)

    def describe(self, frames, sample_fps, prompt, max_new_tokens=1536):
        return self.analyzer.describe_frames(frames, sample_fps, prompt, self.max_pixels, max_new_tokens)

    def text(self, prompt, max_new_tokens=1536):
        """Lượt chỉ có chữ (gộp các lượt nhìn thành thẻ cảnh, hoặc sửa lại thẻ lỗi ngôn ngữ)."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self.analyzer._generate(messages, max_new_tokens=max_new_tokens)

    def close(self):
        pass        # tiến trình con thoát là trả VRAM


def _gguf(filename):
    env = os.environ.get("VL_GGUF_DIR")
    if env:
        path = os.path.join(env, filename)
        if os.path.isfile(path):
            return path
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(VL_GGUF_REPO, filename)
    if not isinstance(path, str) or not os.path.isfile(path):
        raise RuntimeError(f"Chưa có {filename} (repo {VL_GGUF_REPO}) — xem docs/TRIEN_KHAI_GTX1060.md.")
    return path


class LlamaCppGGUFBackend:
    name = "llamacpp_gguf"
    model = f"{VL_GGUF_REPO}/{VL_GGUF_FILE} + {VL_MMPROJ_FILE}"

    def __init__(self, profile, work_dir, log_dir):
        from film.llm_client import LlamaServer
        self.profile = profile
        self.side = profile.get("vision_image_side", 448)
        self.max_pixels = self.side * self.side
        # pid ghi vào tmp/ của dự án (thư mục cha của work_dir) - đúng chỗ stage_runner tìm để
        # dọn llama-server mồ côi nếu bước bị tắt ngang.
        self.server = LlamaServer(profile, os.path.join(log_dir, "llama-server-vl.log"), os.path.dirname(work_dir),
                                  model=_gguf(VL_GGUF_FILE), mmproj=_gguf(VL_MMPROJ_FILE),
                                  ctx=VL_CTX, kv_q8=False).start()

    def max_frames(self):
        # Không giới hạn bởi VRAM như bản transformers (ảnh được mã hoá từng cái), mà bởi thời
        # gian: mỗi ảnh thêm ~256 token cho model đọc. Lấy đúng mức của hồ sơ.
        return self.profile["vision_max_frames"]

    def _image_part(self, img):
        img = img.copy()
        img.thumbnail((self.side, self.side))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=88)
        return {"type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}

    def describe(self, frames, sample_fps, prompt, max_new_tokens=1536):
        content = []
        for i, img in enumerate(frames):
            # Cùng cách lấy mẫu với vision_analyzer_app.sample_frames: mốc giữa mỗi khoảng.
            t = (i + 0.5) / max(sample_fps, 1e-6)
            content.append({"type": "text", "text": f"[giây {t:.1f}]"})
            content.append(self._image_part(img))
        content.append({"type": "text", "text": f"Trên đây là {len(frames)} khung hình liên tiếp, mỗi khung "
                                                f"có mốc giây ghi ngay trước nó.\n\n{prompt}"})
        return self.server.chat_raw(content, max_tokens=max_new_tokens)

    def text(self, prompt, max_new_tokens=1536):
        return self.server.chat_raw([{"type": "text", "text": prompt}], max_tokens=max_new_tokens)

    def close(self):
        self.server.stop()


def create(profile, work_dir, log_dir):
    os.makedirs(work_dir, exist_ok=True)
    if profile["vision_backend"] == "transformers_awq":
        return TransformersAWQBackend(profile, work_dir, log_dir)
    if profile["vision_backend"] == "llamacpp_gguf":
        return LlamaCppGGUFBackend(profile, work_dir, log_dir)
    raise ValueError(f"Không biết backend thị giác '{profile['vision_backend']}'.")
