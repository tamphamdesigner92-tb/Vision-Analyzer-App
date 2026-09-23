import sys
import os
import shutil
import sqlite3
import datetime
import difflib
import uuid
import json
import re

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import torch
from PIL import Image
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class VisionStorageDB:
    """Quản lý lưu trữ thông tin phân tích vào SQLite"""
    def __init__(self, db_path="analysis_history.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    media_type TEXT,
                    source_path TEXT,
                    prompt TEXT,
                    analysis_result TEXT,
                    saved_images TEXT,
                    created_at TIMESTAMP
                )
            """)
            conn.commit()

    def save_record(self, record_id, media_type, source_path, prompt, result, saved_images):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO records (id, media_type, source_path, prompt, analysis_result, saved_images, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (record_id, media_type, source_path, prompt, result, ";".join(saved_images), datetime.datetime.now()))
            conn.commit()


PATCH_PIXELS = 14 * 14                                   # một patch thị giác = 14x14 px
VIDEO_MIN_PIXELS_PER_FRAME = int(128 * 28 * 28 * 1.05)   # sàn min_pixels/frame của qwen_vl_utils
FRAME_FACTOR = 2                                         # qwen_vl_utils làm tròn số frame theo bội số này
DEFAULT_MAX_PIXELS = 360 * 420                           # độ phân giải mỗi frame khi lấy mẫu video

# Ngân sách patch dùng khi CHƯA nạp model (để ước tính trước cho giao diện). Con số này
# khớp với ngân sách thực đo được trên GPU 16GB sau khi model đã chiếm chỗ.
DEFAULT_PATCH_BUDGET = 8000


def video_duration(video_path):
    """Độ dài video theo giây; trả về 0.0 nếu không đọc được metadata."""
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if source_fps > 0 and total_frames > 0:
        return total_frames / source_fps
    return 0.0


def sample_frames(video_path, start, end, nframes):
    """Lấy nframes khung hình PIL trải đều trong [start, end) - tua thẳng tới đoạn cần, KHÔNG
    giải mã lại từ đầu phim. Trả về (khung hình, mốc giây của từng khung).

    Dùng cho phim dài: đọc qua qwen_vl_utils với video_start/video_end phải giải mã từ đầu
    file mỗi lần, cảnh ở phút 150 của một bộ phim sẽ tốn hàng phút chỉ để tới được đó.
    seek() của ffmpeg nhảy tới KEYFRAME gần nhất trước mốc cần, nên sau khi seek vẫn giải mã
    tiếp tới đúng mốc (bỏ bước này thì mọi mốc trong cùng một GOP ra y hệt một khung).
    Theo cách làm của nhánh Vision-on-Mac (sample_video_frames)."""
    import av

    span = max(0.0, end - start)
    if span <= 0 or nframes <= 0:
        return [], []
    targets = [start + span * (i + 0.5) / nframes for i in range(nframes)]
    frames, times = [], []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(int(max(0.0, start) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            # while: khi lấy mẫu dày hơn fps thật, nhiều mốc rơi vào cùng một khung hình.
            while len(frames) < nframes and t + 1e-3 >= targets[len(frames)]:
                frames.append(frame.to_image())
                times.append(round(t, 3))
            if len(frames) >= nframes or t > end + 1:
                break
    while frames and len(frames) < nframes:      # file cắt dở: bù bằng khung cuối
        frames.append(frames[-1])
        times.append(times[-1])
    return frames, times


def timeline_instruction(start, end):
    """Ràng buộc mốc thời gian gắn vào cuối mọi prompt video.

    Không có nó, model không biết video dài bao lâu và hay ghi mốc kiểu "0:00 - 1:00"
    (đọc thành phút), rồi cứ thế đếm tiếp tới hàng chục "phút" cho một video vài giây.
    Ghi thẳng bằng GIÂY, nêu rõ mốc đầu/cuối và bắt buộc các khoảng nối liền nhau thì hết
    nhập nhằng. Những giây liền nhau không có gì thay đổi được gộp thành một dòng."""
    first, last = int(start), max(int(start) + 1, int(round(end)))
    return (
        f"\n\nQUY ĐỊNH VỀ MỐC THỜI GIAN (bắt buộc):\n"
        f"- Đoạn video này kéo dài từ giây {first} đến giây {last} ({last - first} giây).\n"
        f"- Chia diễn biến thành các khoảng giây nối liền nhau, mỗi khoảng một dòng, đúng định dạng:\n"
        f"  - Giây a–b: <mô tả>\n"
        f"- Chỉ mở dòng mới khi có thay đổi (hành động, đối tượng xuất hiện/biến mất, góc máy). "
        f"Nhiều giây liên tiếp không có gì thay đổi thì GỘP thành MỘT dòng, ví dụ \"Giây 1–10: ...\"; "
        f"không viết lại cùng một nội dung cho từng giây.\n"
        f"- Dòng đầu bắt đầu từ giây {first}, dòng cuối kết thúc ở giây {last}; khoảng sau bắt đầu "
        f"đúng chỗ khoảng trước kết thúc, không bỏ trống giây nào. Không ghi mốc nào vượt quá "
        f"giây {last}, không dùng định dạng phút:giây."
    )


# Dòng mốc đúng định dạng timeline_instruction yêu cầu: "- Giây 3–7: mô tả"
_RANGE_LINE = re.compile(r"^\s*[-*•]?\s*Giây\s*(\d+)\s*[–-]\s*(\d+)\s*:\s*(.*)$", re.IGNORECASE)
# Câu model hay dùng khi một giây chẳng có gì mới so với giây trước.
_NO_CHANGE = re.compile(r"không (?:có )?(?:sự )?thay đổi|giữ nguyên|như (?:cũ|trước)|tương tự", re.IGNORECASE)
_FILLER = re.compile(r"\b(?:vẫn|tiếp tục|còn|đang|lại)\b", re.IGNORECASE)


def _same_scene(a, b):
    norm = lambda s: " ".join(_FILLER.sub(" ", re.sub(r"[^\w\s]", " ", s.lower())).split())
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() >= 0.85


def merge_timeline(text):
    """Lưới an toàn thứ hai: gộp các dòng liền nhau mà nội dung không đổi.

    Model vẫn có lúc viết từng giây một dù đã được dặn gộp, chỉ đổi chữ "vẫn"/"tiếp tục"
    ("Tòa nhà vẫn giữ nguyên..." rồi "Tòa nhà tiếp tục giữ nguyên..."). Dòng nối tiếp
    (bắt đầu đúng chỗ dòng trước kết thúc) mà gần như trùng câu, hoặc chỉ báo "không thay
    đổi", được nhập vào dòng trước và giữ mô tả của dòng trước."""
    out = []   # phần tử: [start, end, desc] cho dòng mốc, hoặc chuỗi cho dòng thường
    for line in text.splitlines():
        m = _RANGE_LINE.match(line)
        if not m:
            out.append(line)
            continue
        start, end, desc = int(m.group(1)), int(m.group(2)), m.group(3).strip()
        prev = out[-1] if out and isinstance(out[-1], list) else None
        if prev and prev[1] == start and (_NO_CHANGE.search(desc) or _same_scene(prev[2], desc)):
            prev[1] = end
            if _NO_CHANGE.search(prev[2]) and not _NO_CHANGE.search(desc):
                prev[2] = desc
            continue
        out.append([start, end, desc])
    return "\n".join(
        f"- Giây {x[0]}–{x[1]}: {x[2]}" if isinstance(x, list) else x for x in out
    )


# Dòng mở đầu bằng một mốc thời gian: "- Giây 7–8: ...", "* 9:00 - 10:00: ...", "12s-13s: ..."
_TIMELINE_LINE = re.compile(
    r"^\s*[-*•]?\s*(?:Giây\s*)?(\d+)(?::(\d{2}))?\s*(?:s|giây)?\s*[–-]", re.IGNORECASE
)


def clamp_timeline(text, duration):
    """Lưới an toàn: bỏ những dòng mốc thời gian bắt đầu từ sau khi video đã hết.

    Chỉ đụng tới dòng mở đầu bằng một mốc thời gian; đoạn văn tự do giữ nguyên."""
    if not duration:
        return text
    kept = []
    for line in text.splitlines():
        m = _TIMELINE_LINE.match(line)
        if m:
            seconds = int(m.group(1)) * 60 + int(m.group(2)) if m.group(2) else int(m.group(1))
            if seconds >= duration:
                continue
        kept.append(line)
    return "\n".join(kept).rstrip()


def plan_video_chunks(video_path, fps, max_pixels, patch_budget):
    """Chia video thành các đoạn đủ ngắn để vision tower không tràn VRAM.

    Trả về (danh sách (start, end) theo giây, số frame tối đa mỗi đoạn).
    Tách khỏi class để giao diện web ước tính được số đoạn trước khi nạp model."""
    # qwen_vl_utils áp sàn min_pixels cho mỗi frame, nên độ phân giải thực tế
    # không thể thấp hơn sàn đó dù max_pixels được đặt nhỏ hơn.
    pixels_per_frame = max(max_pixels, VIDEO_MIN_PIXELS_PER_FRAME)
    patches_per_frame = pixels_per_frame / PATCH_PIXELS

    # temporal_patch_size=2: mỗi 2 frame mới sinh ra một hàng patch theo trục thời gian
    frames_per_chunk = int(2 * patch_budget / patches_per_frame)
    frames_per_chunk -= frames_per_chunk % FRAME_FACTOR
    frames_per_chunk = max(FRAME_FACTOR, frames_per_chunk)

    seconds_per_chunk = frames_per_chunk / fps
    duration = video_duration(video_path)
    if duration <= seconds_per_chunk:
        return [(None, None)], frames_per_chunk

    chunks = []
    start = 0.0
    while start < duration - 0.1:
        chunks.append((start, min(start + seconds_per_chunk, duration)))
        start += seconds_per_chunk
    return chunks, frames_per_chunk


class LocalVisionAnalyzer:

    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct-AWQ", storage_dir="./storage",
                 progress_callback=None):
        self.storage_dir = storage_dir
        self.progress_callback = progress_callback
        os.makedirs(self.storage_dir, exist_ok=True)
        self.db = VisionStorageDB(os.path.join(self.storage_dir, "analysis_history.db"))

        self._report(f"[*] Đang tải mô hình {model_id} lên GPU...", 2)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self._fix_awq_lm_head_if_broken(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self._report("[+] Khởi tạo mô hình thành công!\n", 8)

    def _report(self, message, percent=None):
        """In tiến trình ra console và đẩy cho giao diện web (nếu có)."""
        print(message, flush=True)
        if self.progress_callback:
            self.progress_callback(message, percent)

    def _task_dir_for(self, media_path):
        """Thư mục kết quả đặt theo tên file gốc, để tra cứu theo tên ảnh/video."""
        stem = os.path.splitext(os.path.basename(media_path))[0]
        safe_stem = re.sub(r"[^\w\-. ]", "_", stem).strip() or "khong_ten"
        task_dir = os.path.join(self.storage_dir, safe_stem)
        os.makedirs(task_dir, exist_ok=True)
        return task_dir, safe_stem

    def _fix_awq_lm_head_if_broken(self, model_id):
        """Một số checkpoint AWQ (vd: Qwen2.5-VL-7B-Instruct-AWQ) khai báo quantization_config
        không loại trừ lm_head, trong khi checkpoint chỉ lưu lm_head.weight ở dạng thường (chưa
        lượng tử hoá). Kết quả là transformers khởi tạo ngẫu nhiên lm_head.qweight thay vì nạp
        trọng số thật, gây lỗi dtype khi suy luận và làm sai toàn bộ kết quả sinh văn bản.
        Hàm này phát hiện tình huống đó và vá lại lm_head bằng trọng số thật lấy từ checkpoint."""
        lm_head = self.model.lm_head
        if not hasattr(lm_head, "qweight") or lm_head.qweight.dtype == torch.int32:
            return  # lm_head bình thường hoặc đã được lượng tử hoá đúng cách

        self._report("[!] Phát hiện lm_head bị khởi tạo sai do lỗi quantization_config của checkpoint. Đang vá lại...", 5)
        index_path = hf_hub_download(model_id, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        shard_file = index["weight_map"]["lm_head.weight"]
        shard_path = hf_hub_download(model_id, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            real_weight = f.get_tensor("lm_head.weight")

        new_lm_head = torch.nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False)
        new_lm_head.weight = torch.nn.Parameter(real_weight.to(dtype=self.model.dtype))
        self.model.lm_head = new_lm_head.to(lm_head.qweight.device)
        self._report("[+] Đã vá lm_head thành công.\n", 6)

    def _generate(self, messages, max_new_tokens=1024):
        """Hàm suy luận chung"""
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        extra = {}
        if video_inputs:
            # FPS lấy mẫu THẬT (qwen_vl_utils làm tròn số frame nên có thể lệch FPS yêu cầu).
            # Thiếu nó, processor mặc định 2.0 và model tính sai giây thực của từng khung hình.
            extra["fps"] = video_kwargs["fps"]
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **extra,
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, repetition_penalty=1.05)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        del inputs, generated_ids, generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return response[0]

    def _vram_patch_budget(self):
        """Số patch thị giác (N) tối đa mà vision tower còn đủ VRAM để xử lý trong một lượt.

        Ở 4 block full-attention, transformers truyền attention_mask tường minh vào
        scaled_dot_product_attention, buộc PyTorch dùng backend "math" và hiện thực hoá
        ma trận điểm [1, num_heads, N, N]. Chi phí do đó là O(N²):
            bytes ≈ num_heads × N² × 2 (fp16) × 2 (bản sao cho softmax)
        Đây là lý do video càng dài thì bộ nhớ tăng theo bình phương, không phải tuyến tính."""
        bytes_per_patch_squared = self.model.config.vision_config.num_heads * 2 * 2
        if self.model.device.type != "cuda":
            return 8000

        free_bytes, _ = torch.cuda.mem_get_info(self.model.device)
        usable_bytes = free_bytes * 0.5  # chừa chỗ cho KV cache và activation của phần LLM
        n_max = int((usable_bytes / bytes_per_patch_squared) ** 0.5)
        return max(2000, min(n_max, 16000))

    def frames_for_budget(self, max_pixels=DEFAULT_MAX_PIXELS):
        """Số khung hình tối đa (chẵn) một lượt suy luận chịu được với VRAM đang trống."""
        pixels_per_frame = max(max_pixels, VIDEO_MIN_PIXELS_PER_FRAME)
        n = int(2 * self._vram_patch_budget() / (pixels_per_frame / PATCH_PIXELS))
        return max(FRAME_FACTOR, n - n % FRAME_FACTOR)

    def describe_frames(self, frames, sample_fps, prompt, max_pixels=DEFAULT_MAX_PIXELS,
                        max_new_tokens=1536):
        """Mô tả một chuỗi khung hình đã lấy mẫu sẵn (vd: một cảnh phim). sample_fps phải là
        mật độ lấy mẫu THẬT của chuỗi này để model tính đúng giây của từng khung."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "sample_fps": sample_fps, "max_pixels": max_pixels},
                {"type": "text", "text": prompt},
            ],
        }]
        return self._generate(messages, max_new_tokens=max_new_tokens)

    def _plan_video_chunks(self, video_path, fps, max_pixels):
        return plan_video_chunks(video_path, fps, max_pixels, self._vram_patch_budget())

    def _analyze_video_segment(self, video_path, prompt, fps, max_pixels, start=None, end=None):
        """Chạy suy luận trên toàn bộ video (start/end = None) hoặc trên một đoạn thời gian."""
        video_content = {
            "type": "video",
            "video": video_path,
            "max_pixels": max_pixels,  # Giới hạn độ phân giải để tránh tràn VRAM
            "fps": fps,                # Số khung hình lấy mẫu mỗi giây
        }
        if start is not None and end is not None:
            video_content["video_start"] = start
            video_content["video_end"] = end

        messages = [
            {
                "role": "user",
                "content": [video_content, {"type": "text", "text": prompt}],
            }
        ]
        return self._generate(messages)

    def _synthesize_timeline(self, segment_reports, prompt):
        """Gộp các bản phân tích từng đoạn thành một báo cáo thống nhất cho cả video."""
        parts = "\n\n".join(
            f"[Đoạn {i}: giây {start:.0f} đến giây {end:.0f}]\n{text}"
            for i, (start, end, text) in enumerate(segment_reports, 1)
        )
        total_duration = segment_reports[-1][1]
        synthesis_prompt = (
            f"Dưới đây là các bản phân tích rời rạc của những đoạn liên tiếp thuộc cùng một video. "
            f"Video này dài đúng {total_duration:.0f} giây, gồm {len(segment_reports)} đoạn:\n\n"
            f"{parts}\n\n"
            f"Hãy tổng hợp thành một báo cáo thống nhất cho toàn bộ video. Mọi mốc thời gian phải nằm "
            f"trong khoảng 0 đến {total_duration:.0f} giây; không được nêu bất kỳ mốc nào vượt quá "
            f"{total_duration:.0f} giây. Nối liền diễn biến giữa các đoạn, loại bỏ phần trùng lặp và "
            f"tuyệt đối không bịa thêm chi tiết không có trong các bản phân tích trên."
            f"\n\nYêu cầu ban đầu: {prompt}"
            f"{timeline_instruction(0, total_duration)}"
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": synthesis_prompt}]}]
        return self._generate(messages)

    def _save_reports(self, task_dir, report_stem, record_id, media_type, source_path, prompt, result,
                      image_rel_paths, segment_reports=None, settings=None):
        """Lưu kết quả ra <tên file gốc>.md và <tên file gốc>.json trong thư mục của file đó."""
        md_path = self._save_markdown_report(
            task_dir, report_stem, record_id, media_type, prompt, result, image_rel_paths, segment_reports,
            settings
        )
        json_path = os.path.join(task_dir, f"{report_stem}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "record_id": record_id,
                    "media_type": media_type,
                    "source_file": os.path.basename(source_path),
                    "source_path": source_path,
                    "prompt": prompt,
                    "settings": settings or {},
                    "analyzed_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    "result": result,
                    "images": image_rel_paths,
                    "segments": [
                        {"index": i, "start_sec": start, "end_sec": end, "text": text}
                        for i, (start, end, text) in enumerate(segment_reports or [], 1)
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return md_path, json_path

    def _save_markdown_report(self, task_dir, report_stem, record_id, media_type, prompt, result,
                              image_rel_paths, segment_reports=None, settings=None):
        """Tạo file Markdown tổng hợp cả ảnh và text để xem lại nhanh"""
        md_path = os.path.join(task_dir, f"{report_stem}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Kết Quả Phân Tích ({media_type.upper()})\n\n")
            f.write(f"- **Mã phân tích (ID):** `{record_id}`\n")
            f.write(f"- **Thời gian:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Yêu cầu (Prompt):** {prompt}\n")
            if settings:
                mota = ", ".join(f"{k}={v}" for k, v in settings.items())
                f.write(f"- **Cài đặt lấy mẫu:** {mota}\n")
            f.write("\n")
            f.write("## 1. Dữ liệu Hình Ảnh Đã Lưu\n\n")
            for img_p in image_rel_paths:
                f.write(f"![Preview]({img_p})\n\n")
            f.write("## 2. Nội dung Phân Tích Trích Xuất (Text)\n\n")
            f.write(f"{result}\n")
            if segment_reports:
                f.write("\n## 3. Phân Tích Gốc Theo Từng Đoạn\n\n")
                for i, (start, end, text) in enumerate(segment_reports, 1):
                    f.write(f"### Đoạn {i}: giây {start:.0f} đến giây {end:.0f}\n\n{text}\n\n")
        return md_path

    def analyze_image(self, image_path: str, prompt: str):
        """Phân tích ảnh tĩnh và lưu trữ kết quả"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy file: {image_path}")

        record_id = f"img_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        task_dir, report_stem = self._task_dir_for(image_path)

        # 1. Chạy suy luận qua mô hình
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        self._report(f"[*] Đang phân tích ảnh: {os.path.basename(image_path)}...", 20)
        result = self._generate(messages)

        # 2. Lưu trữ ảnh gốc vào thư mục riêng
        self._report("[*] Đang lưu kết quả...", 90)
        saved_img_name = f"source_{os.path.basename(image_path)}"
        saved_img_path = os.path.join(task_dir, saved_img_name)
        shutil.copy2(image_path, saved_img_path)

        # 3. Lưu vào Database, file Markdown và JSON
        self.db.save_record(record_id, "image", image_path, prompt, result, [saved_img_path])
        md_path, json_path = self._save_reports(
            task_dir, report_stem, record_id, "image", image_path, prompt, result, [saved_img_name]
        )

        self._report(f"[✓] Đã hoàn thành! Kết quả lưu tại: {task_dir}\n", 100)
        return {
            "id": record_id,
            "result": result,
            "storage_dir": task_dir,
            "report_stem": report_stem,
            "md_path": md_path,
            "json_path": json_path,
        }

    def _extract_keyframes(self, video_path: str, output_dir: str, num_frames=4):
        """Hỗ trợ trích xuất một số khung hình tiêu biểu từ video để lưu lại kèm text"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        step = max(1, total_frames // num_frames)
        saved_frames = []
        for i in range(num_frames):
            frame_idx = min(i * step, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_filename = f"keyframe_{i+1:02d}.jpg"
                frame_filepath = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_filepath, frame)
                saved_frames.append(frame_filename)
        cap.release()
        return saved_frames

    def analyze_video(self, video_path: str, prompt: str, fps=2.0, max_pixels=DEFAULT_MAX_PIXELS,
                      num_keyframes_saved=4):
        """Phân tích video, trích xuất khung hình đại diện và lưu trữ kết quả"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Không tìm thấy file: {video_path}")

        record_id = f"vid_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        task_dir, report_stem = self._task_dir_for(video_path)

        # 1. Trích xuất các khung hình tiêu biểu để lưu trữ kèm báo cáo
        self._report("[*] Đang trích xuất khung hình tiêu biểu...", 12)
        keyframes = self._extract_keyframes(video_path, task_dir, num_frames=num_keyframes_saved)

        # 2. Chạy suy luận video qua Qwen2.5-VL, chia đoạn nếu video quá dài so với VRAM
        chunks, frames_per_chunk = self._plan_video_chunks(video_path, fps, max_pixels)
        segment_reports = []

        duration = video_duration(video_path)
        if len(chunks) == 1:
            self._report(f"[*] Đang phân tích video: {os.path.basename(video_path)} (FPS lấy mẫu: {fps})...", 20)
            result = self._analyze_video_segment(
                video_path, prompt + timeline_instruction(0, duration), fps, max_pixels
            )
        else:
            self._report(
                f"[*] Video dài hơn ngân sách VRAM, chia thành {len(chunks)} đoạn "
                f"(tối đa {frames_per_chunk} frame mỗi đoạn)...",
                15,
            )
            for i, (start, end) in enumerate(chunks, 1):
                # Các đoạn chiếm dải tiến trình 15% -> 80%
                self._report(
                    f"    [{i}/{len(chunks)}] Đang phân tích giây {start:.0f} đến {end:.0f}...",
                    15 + (i - 1) * 65 / len(chunks),
                )
                segment_prompt = (
                    f"Đây là đoạn từ giây {start:.0f} đến giây {end:.0f} của một video dài hơn. "
                    f"Hãy mô tả diễn biến trong đoạn này và ghi mốc thời gian theo thời gian thật của "
                    f"toàn bộ video (tức cộng thêm {start:.0f} giây).\n\nYêu cầu: {prompt}"
                    f"{timeline_instruction(start, end)}"
                )
                segment_text = self._analyze_video_segment(
                    video_path, segment_prompt, fps, max_pixels, start=start, end=end
                )
                segment_reports.append((start, end, segment_text))

            self._report(f"[*] Đang tổng hợp {len(segment_reports)} đoạn thành báo cáo cuối...", 80)
            result = self._synthesize_timeline(segment_reports, prompt)
        result = merge_timeline(clamp_timeline(result, duration))

        # 3. Lưu vào Database, file Markdown và JSON
        self._report("[*] Đang lưu kết quả...", 92)
        full_keyframe_paths = [os.path.join(task_dir, kf) for kf in keyframes]
        self.db.save_record(record_id, "video", video_path, prompt, result, full_keyframe_paths)
        md_path, json_path = self._save_reports(
            task_dir, report_stem, record_id, "video", video_path, prompt, result, keyframes,
            segment_reports=segment_reports,
            settings={"fps": fps, "max_pixels": max_pixels, "so_doan": len(chunks)},
        )

        self._report(f"[✓] Đã hoàn thành! Kết quả và ảnh trích xuất lưu tại: {task_dir}\n", 100)
        return {
            "id": record_id,
            "result": result,
            "storage_dir": task_dir,
            "report_stem": report_stem,
            "md_path": md_path,
            "json_path": json_path,
            "keyframes": keyframes,
            "num_segments": len(chunks),
            "segments": [
                {"index": i, "start_sec": start, "end_sec": end, "text": text}
                for i, (start, end, text) in enumerate(segment_reports, 1)
            ],
        }


# ==========================================
# VÍ DỤ SỬ DỤNG TRỰC TIẾP
# ==========================================
if __name__ == "__main__":
    # Khởi tạo ứng dụng
    app = LocalVisionAnalyzer(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        storage_dir="./vision_storage"
    )

    # --- 1. Thử nghiệm với Ảnh ---
    # Thay bằng đường dẫn ảnh thực tế của bạn (jpg/png)
    sample_image = "test_image.jpg"
    if os.path.exists(sample_image):
        img_prompt = "Hãy mô tả chi tiết hình ảnh này, liệt kê các đối tượng chính và chữ (OCR) nếu có."
        img_res = app.analyze_image(sample_image, img_prompt)
        print("KẾT QUẢ PHÂN TÍCH ẢNH:")
        print(img_res["result"])
    else:
        print(f"[!] Vui lòng đặt một ảnh '{sample_image}' cùng thư mục để thử nghiệm ảnh.")

    # --- 2. Thử nghiệm với Video ---
    # Thay bằng đường dẫn video thực tế của bạn (mp4/mkv/mov)
    sample_video = "test_video.mp4"
    if os.path.exists(sample_video):
        vid_prompt = (
            "Hãy tóm tắt diễn biến trong video, chia theo các mốc thời gian (timestamp) "
            "và nêu rõ hành động của từng nhân vật/đối tượng."
        )
        vid_res = app.analyze_video(
            video_path=sample_video,
            prompt=vid_prompt,
            fps=1.0,               # Lấy mẫu 1 hình/giây (tùy chỉnh 0.5 nếu video dài)
            max_pixels=360 * 420,  # Giữ ở mức này để tối ưu VRAM
            num_keyframes_saved=4  # Số khung hình trích xuất để lưu làm bằng chứng trong báo cáo
        )
        print("KẾT QUẢ PHÂN TÍCH VIDEO:")
        print(vid_res["result"])
    else:
        print(f"[!] Vui lòng đặt một video '{sample_video}' cùng thư mục để thử nghiệm video.")