import sys
import os
import time
import shutil
import sqlite3
import datetime
import difflib
import uuid
import json
import re

# HF_HUB_CACHE phải được đặt TRƯỚC khi import huggingface_hub/transformers: các thư viện đó
# đọc biến này đúng một lần lúc import và đóng băng vào hằng số. Đặt vào AI Hub để mọi thứ
# tải từ HuggingFace đều gom về một chỗ, không sinh thêm một bản sao nhiều GB trong
# ~/.cache/huggingface.
_AIHUB_HUB = os.environ.get("AIHUB_HUB", "/Users/mac/.aihub/models/hf/hub")
if os.path.isdir(_AIHUB_HUB):
    os.environ.setdefault("HF_HUB_CACHE", _AIHUB_HUB)


if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Dùng PyAV (av) chứ không dùng OpenCV cho phần video. Không phải vì PyAV tiện hơn, mà vì
# hai gói này mỗi gói bundle một bản ffmpeg riêng: av mang libavdevice 62, opencv-python mang
# libavdevice 61. Nạp cả hai vào cùng một tiến trình thì runtime ObjC của macOS thấy các lớp
# AVFFrameReceiver/AVFAudioReceiver được đăng ký hai lần và cảnh báo "may cause spurious
# casting failures and mysterious crashes" - đúng ngay đường xử lý video của ứng dụng này.
# Không bỏ được av: gptqmodel import nó ngay khi nạp gói (qua MiniCPM-O), và qwen_vl_utils
# đọc video qua torchvision -> pyav. Nên bên bỏ đi phải là OpenCV.
import av
import torch
from PIL import Image  # noqa: F401 - av.VideoFrame.to_image() cần Pillow
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.video_utils import VideoMetadata
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


# ==========================================
# CAC MO HINH THI GIAC DUNG DUOC
# ==========================================
# Đều là bản gốc chính thức của Qwen, tải sẵn trong AI Hub cục bộ (không dùng bản convert MLX
# cộng đồng — đã thử và bị lỗi vùng nhận diện thị giác không đáng tin).
AIHUB_HUB = _AIHUB_HUB

VL_MODELS = [
    {
        "key": "qwen2.5-vl-7b-awq",
        "repo": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        "label": "Qwen2.5-VL 7B (AWQ 4-bit) — mô tả tốt hơn, rất chậm",
        "note": "Khoảng 0,23 token/giây trên máy này: một video ngắn mất ~20 phút. "
                "Chậm vì trọng số int4 phải giải nén lại trong mỗi lần forward.",
        # Đo thực trên máy này, dùng để ước tính thời gian chạy (xem _estimate_seconds).
        "seconds_per_frame": 1.4,
        "seconds_per_token": 4.3,
    },
    {
        "key": "qwen2.5-vl-3b-bf16",
        "repo": "Qwen/Qwen2.5-VL-3B-Instruct",
        "label": "Qwen2.5-VL 3B (bf16) — nhanh hơn nhiều, mô tả sơ hơn",
        "note": "Khoảng 15,6 token/giây trên máy này — nhanh hơn bản 7B khoảng 67 lần, "
                "một video ngắn xong trong khoảng một phút. Nhanh vì trọng số để nguyên "
                "bf16, không phải giải nén lại mỗi lần chạy. Đổi lại mô hình nhỏ hơn nên "
                "mô tả thường ngắn và ít chi tiết hơn bản 7B.",
        "seconds_per_frame": 0.51,
        "seconds_per_token": 0.064,
    },
]


def _aihub_snapshot(repo_id):
    """Đường dẫn snapshot của repo trong AI Hub; None nếu chưa tải về.

    Không ghim cứng mã snapshot như trước: mã đó đổi mỗi lần repo trên HuggingFace có bản
    mới, ghim vào code thì sau một lần tải lại là đứt đường dẫn.
    """
    base = os.path.join(AIHUB_HUB, "models--" + repo_id.replace("/", "--"), "snapshots")
    if not os.path.isdir(base):
        return None
    snaps = [os.path.join(base, d) for d in sorted(os.listdir(base))]
    snaps = [d for d in snaps if os.path.isdir(d)]
    return max(snaps, key=os.path.getmtime) if snaps else None


def model_entry(key):
    for m in VL_MODELS:
        if m["key"] == key:
            return m
    raise ValueError(f"Không có mô hình nào tên {key!r}.")


def model_path(key):
    """Đường dẫn cục bộ nếu đã tải ĐỦ; nếu chưa thì trả về repo id để HuggingFace tải nốt.

    Quan trọng là "đủ" chứ không phải "có thư mục": trỏ vào một snapshot đang tải dở thì
    transformers nạp thiếu trọng số, còn trả về repo id thì nó tải tiếp phần còn thiếu.
    """
    entry = model_entry(key)
    if model_is_local(key):
        return _aihub_snapshot(entry["repo"])
    return entry["repo"]


def model_is_local(key):
    """True chỉ khi snapshot có ĐỦ trọng số, không phải chỉ có thư mục.

    Trong lúc tải, HuggingFace đã tạo sẵn thư mục snapshot và các file nhỏ (config, tokenizer)
    nhưng trọng số còn nằm ở dạng .incomplete và chưa được liên kết vào. Nếu chỉ kiểm tra
    "có thư mục không" thì giao diện báo đã tải xong, người dùng chọn vào, và model chết giữa
    chừng vì thiếu trọng số thay vì tải nốt phần còn lại.
    """
    snapshot = _aihub_snapshot(model_entry(key)["repo"])
    if snapshot is None:
        return False
    index = os.path.join(snapshot, "model.safetensors.index.json")
    if os.path.isfile(index):
        try:
            with open(index, encoding="utf-8") as f:
                shards = set(json.load(f).get("weight_map", {}).values())
        except (OSError, ValueError):
            return False
        return bool(shards) and all(os.path.exists(os.path.join(snapshot, n)) for n in shards)
    return any(n.endswith(".safetensors") for n in os.listdir(snapshot))


DEFAULT_VL_MODEL = os.environ.get("VL_MODEL", VL_MODELS[0]["key"])

# Giữ lại tên cũ cho những chỗ đã dùng, và cho phép trỏ thẳng vào một thư mục bất kỳ bằng
# biến môi trường VL_MODEL_PATH (ưu tiên cao hơn VL_MODEL).
VL_MODEL_PATH = os.environ.get("VL_MODEL_PATH") or model_path(DEFAULT_VL_MODEL)

PATCH_PIXELS = 14 * 14                                   # một patch thị giác = 14x14 px
VIDEO_MIN_PIXELS_PER_FRAME = int(128 * 28 * 28 * 1.05)   # sàn min_pixels/frame của qwen_vl_utils
FRAME_FACTOR = 2                                         # qwen_vl_utils làm tròn số frame theo bội số này
# Trần/sàn số khung hình mỗi lượt suy luận, lấy đúng theo qwen_vl_utils (FPS_MIN_FRAMES /
# FPS_MAX_FRAMES). Ứng dụng tự giải mã video nên phải tự giữ hai mốc này: bỏ qua thì một
# video dài lấy mẫu ở fps cao sẽ nhét hàng nghìn khung hình vào một lượt và tràn bộ nhớ.
MIN_SAMPLED_FRAMES = 4
MAX_SAMPLED_FRAMES = 768
# Độ dài tối đa của phần chữ mô hình sinh ra. Trên máy này mỗi token mất khoảng 4 giây (xem
# ghi chú tốc độ ở _estimate_seconds trong web_app.py), nên con số này là trần thời gian chạy
# chứ không chỉ là trần độ dài: 1024 token tương đương hơn một tiếng cho MỘT đoạn video.
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
DEFAULT_MAX_PIXELS = 360 * 420                           # độ phân giải mỗi frame khi lấy mẫu video
# Ảnh tĩnh không đi qua qwen_vl_utils nên không có sàn min_pixels như video, nhưng vẫn cần một
# trần để tránh vision tower tràn bộ nhớ hợp nhất với ảnh chụp gốc (chụp điện thoại có thể
# 4000x3000 trở lên) — đã tái hiện được lỗi tràn bộ nhớ MPS khi thử với ảnh ~1.9 triệu px.
DEFAULT_IMAGE_MAX_PIXELS = 1280 * 28 * 28

# RAM hợp nhất trên Mac chỉ 16GB (chia sẻ với OS + toàn bộ ứng dụng), nên video dài vẫn cần
# chia đoạn để không giữ quá nhiều khung hình trong một lượt suy luận. Khác bản CUDA cũ (ngân
# sách tính theo công thức attention bậc hai của SDPA "math backend"), ở đây dùng một mốc thời
# lượng cố định đơn giản, đủ an toàn cho máy 16GB và chỉnh được qua biến môi trường.
DEFAULT_MAX_VIDEO_CHUNK_SECONDS = float(os.environ.get("MAX_VIDEO_CHUNK_SECONDS", "90"))


def video_duration(video_path):
    """Độ dài video theo giây; trả về 0.0 nếu không đọc được metadata.

    Đọc thẳng độ dài trong metadata thay vì suy ra từ (số frame / fps) như bản dùng OpenCV
    trước đây: với video fps biến thiên (VFR - hay gặp ở video quay bằng điện thoại và video
    tải từ mạng) thì fps trung bình nhân số frame ra một con số lệch hẳn.
    """
    try:
        with av.open(video_path) as container:
            if container.duration is not None:
                return container.duration / av.time_base
            stream = container.streams.video[0]
            if stream.duration is not None and stream.time_base:
                return float(stream.duration * stream.time_base)
    except (av.FFmpegError, IndexError, OSError):
        pass  # file hỏng, không phải video, hoặc không có luồng hình
    return 0.0


def _frame_at(container, stream, target_sec):
    """Khung hình đầu tiên tại hoặc sau mốc target_sec; None nếu không lấy được.

    seek() của ffmpeg chỉ nhảy được tới KEYFRAME gần nhất TRƯỚC mốc cần, nên sau khi seek
    vẫn phải giải mã tiếp tới đúng mốc. Bỏ bước này thì mọi mốc trong cùng một GOP đều trả
    về y hệt một khung hình - với video x264 mặc định (GOP 250 frame) thì cả video ngắn chỉ
    có đúng một keyframe, và bốn "khung hình tiêu biểu" hoá ra là bốn bản sao của frame đầu.
    """
    try:
        container.seek(int(target_sec / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            if float(frame.pts * stream.time_base) >= target_sec - 1e-3:
                return frame
    except (av.FFmpegError, StopIteration):
        pass
    return None


def _round_to_factor(value, factor=FRAME_FACTOR):
    return max(factor, int(round(value / factor)) * factor)


def sample_video_frames(video_path, fps, start=None, end=None):
    """Giải mã video bằng av và trả về danh sách khung hình PIL đã lấy mẫu.

    Trước đây chỗ này đưa thẳng đường dẫn video cho qwen_vl_utils, nhưng bộ đọc video của nó
    gọi torchvision.io.read_video - hàm đã bị gỡ khỏi torchvision 0.29, nên mọi lần phân tích
    video đều chết với AttributeError. Tự giải mã bằng av vừa hết phụ thuộc vào hàm đó, vừa
    dùng lại đúng bản ffmpeg đã có sẵn trong tiến trình (xem ghi chú ở phần import).

    Lấy mẫu theo MỐC THỜI GIAN trải đều trên đoạn, không phải "cứ 1/fps giây lấy một khung":
    khi số khung bị chặn ở MAX_SAMPLED_FRAMES, cách trải đều vẫn phủ hết đoạn video, còn cách
    kia sẽ dừng giữa chừng và bỏ trắng phần đuôi.
    """
    duration = video_duration(video_path)
    if duration <= 0:
        return []
    seg_start = 0.0 if start is None else max(0.0, float(start))
    seg_end = duration if end is None else min(duration, float(end))
    span = seg_end - seg_start
    if span <= 0:
        return []

    nframes = _round_to_factor(span * fps)
    nframes = min(max(nframes, MIN_SAMPLED_FRAMES), MAX_SAMPLED_FRAMES)
    targets = [seg_start + span * i / nframes for i in range(nframes)]

    frames = []
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            container.seek(int(seg_start / stream.time_base), stream=stream)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                t = float(frame.pts * stream.time_base)
                if t > seg_end + 1e-3 and frames:
                    break
                # while chứ không phải if: khi fps lấy mẫu cao hơn fps thật của video, nhiều
                # mốc rơi vào cùng một khung hình - lặp lại khung đó, giống linspace của
                # qwen_vl_utils, thay vì lệch pha dần rồi thiếu khung ở cuối.
                while len(frames) < nframes and t + 1e-3 >= targets[len(frames)]:
                    frames.append(frame.to_image())
                if len(frames) >= nframes:
                    break
    except (av.FFmpegError, IndexError, OSError):
        return frames

    # Giải mã kết thúc sớm (file cắt dở, metadata báo dài hơn thực tế): bù cho đủ số khung
    # bằng khung cuối, đúng cách qwen_vl_utils bù khi nhận một danh sách khung hình.
    while frames and len(frames) < nframes:
        frames.append(frames[-1])
    return frames


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


def plan_video_chunks(video_path, fps, chunk_seconds=DEFAULT_MAX_VIDEO_CHUNK_SECONDS):
    """Chia video thành các đoạn đủ ngắn để không giữ quá nhiều khung hình trong một lượt.

    Trả về (danh sách (start, end) theo giây, số frame lấy mẫu ước tính mỗi đoạn).
    Tách khỏi class để giao diện web ước tính được số đoạn trước khi nạp model."""
    duration = video_duration(video_path)
    frames_per_chunk = max(FRAME_FACTOR, int(chunk_seconds * fps))
    frames_per_chunk -= frames_per_chunk % FRAME_FACTOR
    frames_per_chunk = max(FRAME_FACTOR, frames_per_chunk)

    if duration <= chunk_seconds:
        return [(None, None)], frames_per_chunk

    chunks = []
    start = 0.0
    while start < duration - 0.1:
        chunks.append((start, min(start + chunk_seconds, duration)))
        start += chunk_seconds
    return chunks, frames_per_chunk


def _device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class _GenerationProgress:
    """Streamer tối giản cho model.generate(): đếm chữ sinh ra và đẩy tiến trình ra giao diện.

    Không có nó thì cả lượt suy luận - trên máy này có thể hơn hai mươi phút - thanh tiến
    trình đứng yên một chỗ, không phân biệt được "đang chạy" với "đã treo".

    generate() gọi put() lần đầu với toàn bộ prompt, sau đó mỗi lần một token mới.
    """

    def __init__(self, report, start_pct, end_pct, max_new_tokens,
                 bar_interval=1.0, text_interval=20.0):
        self.report = report
        self.start_pct = start_pct
        self.end_pct = end_pct
        self.max_new_tokens = max(1, max_new_tokens)
        self.bar_interval = bar_interval
        self.text_interval = text_interval
        self.count = 0
        self._got_prompt = False
        self._t0 = time.time()
        self._last_bar = 0.0
        self._last_text = time.time()

    def put(self, value):
        if not self._got_prompt:
            self._got_prompt = True   # lần đầu là prompt, không phải chữ sinh ra
            return
        try:
            self.count += int(value.numel())
        except AttributeError:
            self.count += 1

        now = time.time()
        if now - self._last_bar < self.bar_interval:
            return
        self._last_bar = now
        ratio = min(1.0, self.count / self.max_new_tokens)
        percent = self.start_pct + (self.end_pct - self.start_pct) * ratio

        if now - self._last_text >= self.text_interval:
            self._last_text = now
            speed = self.count / max(1e-6, now - self._t0)
            self.report(
                f"    ... đã sinh {self.count} token ({speed:.2f} token/giây, "
                f"đã chạy {now - self._t0:.0f} giây)",
                percent,
            )
        else:
            self.report(None, percent)   # chỉ nhích thanh, không thêm dòng log

    def end(self):
        pass


class LocalVisionAnalyzer:

    def __init__(self, model_id=VL_MODEL_PATH, storage_dir="./storage",
                 progress_callback=None):
        self.storage_dir = storage_dir
        self.progress_callback = progress_callback
        os.makedirs(self.storage_dir, exist_ok=True)
        self.db = VisionStorageDB(os.path.join(self.storage_dir, "analysis_history.db"))

        self._report(f"[*] Đang tải mô hình {model_id}...", 2)
        self.model = self._load_model(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self._report("[+] Khởi tạo mô hình thành công!\n", 8)

    def _load_model(self, model_id):
        """Nạp model AWQ qua transformers/gptqmodel, chạy trên MPS (không cần CUDA).

        Vá một lỗi khớp đường dẫn module đã xác minh trực tiếp: checkpoint AWQ gốc của Qwen
        khai báo quantization_config.modules_to_not_convert=["visual"], nhưng ở các bản
        transformers mới, submodule thị giác nằm dưới "model.visual...." (có tiền tố "model.").
        Hàm khớp mẫu của transformers so khớp từ đầu chuỗi nên "visual" (không tiền tố) không
        khớp được "model.visual...." — hậu quả là TOÀN BỘ vision tower bị coi là cần lượng tử
        hoá, không tìm thấy trọng số nén tương ứng trong checkpoint (vốn lưu ở dạng thường vì
        đã được loại trừ), nên bị nạp NGẪU NHIÊN thay vì trọng số thật đã huấn luyện — model
        chạy được nhưng "mù", trả lời sai hoàn toàn về nội dung ảnh dù không báo lỗi gì.
        Đã kiểm chứng bằng cách so LOAD REPORT của transformers trước/sau khi vá: trước vá có
        hàng chục dòng MISSING/UNEXPECTED cho model.visual.*, sau vá sạch hoàn toàn."""
        config = AutoConfig.from_pretrained(model_id)
        qcfg = getattr(config, "quantization_config", None)
        if qcfg:
            qcfg["modules_to_not_convert"] = ["model.visual", "visual", "lm_head"]

        # bfloat16 thay vì float16: fp16 tràn số (inf/nan) trong attention của vision tower
        # trên backend MPS với checkpoint này, gây lỗi ngay cả khi tự nó không liên quan tới
        # bộ nhớ. bfloat16 có dải giá trị rộng hơn nên không gặp lỗi này (đã kiểm chứng trực
        # tiếp: cùng ảnh, fp16 lỗi "probability tensor contains inf/nan", bfloat16 chạy đúng).
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, config=config, dtype=torch.bfloat16
        )
        return model.to(_device())

    def _report(self, message, percent=None):
        """In tiến trình ra console và đẩy cho giao diện web (nếu có).

        message=None nghĩa là "chỉ nhích thanh tiến trình, đừng thêm dòng log". Dùng khi cập
        nhật liên tục lúc mô hình đang sinh chữ: nếu dòng nào cũng in ra thì một lượt chạy dài
        sẽ đẻ ra hàng trăm dòng giống nhau, đúng cái cảm giác "máy đang lặp vô hạn".
        """
        if message is not None:
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

    def _generate(self, messages, video_metadata=None, progress=None):
        """Hàm suy luận chung.

        video_metadata cho processor biết các khung hình truyền vào cách nhau bao lâu. Thiếu
        nó, processor mặc định coi video gốc 24 fps và tính second_per_grid_ts sai hẳn - mô
        hình sẽ tưởng 12 khung hình trải trong nửa giây thay vì 6 giây, nên mọi mốc thời gian
        nó nói ra đều lệch.
        """
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        extra = {"video_metadata": video_metadata} if video_metadata else {}
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **extra
        ).to(self.model.device)

        streamer = None
        if progress is not None:
            start_pct, end_pct = progress
            streamer = _GenerationProgress(self._report, start_pct, end_pct, MAX_NEW_TOKENS)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, streamer=streamer,
                repetition_penalty=1.05,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        del inputs, generated_ids, generated_ids_trimmed
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return response[0]

    def _plan_video_chunks(self, video_path, fps):
        return plan_video_chunks(video_path, fps)

    def _analyze_video_segment(self, video_path, prompt, fps, max_pixels, start=None, end=None,
                               progress=None):
        """Chạy suy luận trên toàn bộ video (start/end = None) hoặc trên một đoạn thời gian."""
        frames = sample_video_frames(video_path, fps, start=start, end=end)
        if not frames:
            raise RuntimeError(f"Không giải mã được khung hình nào từ video: {video_path}")

        video_content = {
            "type": "video",
            "video": frames,           # Khung hình đã giải mã sẵn, không đưa đường dẫn nữa
            "max_pixels": max_pixels,  # Giới hạn độ phân giải để tránh tràn bộ nhớ
        }
        messages = [
            {
                "role": "user",
                "content": [video_content, {"type": "text", "text": prompt}],
            }
        ]

        # fps THỰC TẾ của chuỗi khung hình, không phải fps người dùng chọn: khi đoạn video dài
        # và số khung bị chặn ở MAX_SAMPLED_FRAMES thì hai con số này khác nhau, và mốc thời
        # gian mô hình đọc ra phải theo con số thực tế.
        span = (end - start) if (start is not None and end is not None) else video_duration(video_path)
        effective_fps = (len(frames) / span) if span > 0 else fps
        metadata = VideoMetadata(
            total_num_frames=len(frames),
            fps=effective_fps,
            frames_indices=list(range(len(frames))),
            duration=span,
        )
        return self._generate(messages, video_metadata=[metadata], progress=progress)

    def _synthesize_timeline(self, segment_reports, prompt, progress=None):
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
        return self._generate(messages, progress=progress)

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

    def analyze_image(self, image_path: str, prompt: str, max_pixels=DEFAULT_IMAGE_MAX_PIXELS):
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
                    {"type": "image", "image": image_path, "max_pixels": max_pixels},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        self._report(f"[*] Đang phân tích ảnh: {os.path.basename(image_path)}...", 20)
        result = self._generate(messages, progress=(20, 88))

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
        """Trích một số khung hình tiêu biểu từ video để lưu lại kèm text.

        Lấy mẫu theo MỐC THỜI GIAN trải đều, không theo chỉ số frame: seek theo thời gian là
        thứ PyAV (và ffmpeg nói chung) làm đúng, còn "frame thứ n" không phải khái niệm chắc
        chắn với video fps biến thiên.
        """
        duration = video_duration(video_path)
        if duration <= 0:
            return []

        saved_frames = []
        try:
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                for i in range(num_frames):
                    # Trải đều trong video, tránh đúng mốc cuối vì ở đó hay không còn frame
                    target = duration * i / num_frames
                    frame = _frame_at(container, stream, target)
                    if frame is None:
                        break  # hết frame hoặc seek hỏng - giữ những gì đã lấy được
                    frame_filename = f"keyframe_{i + 1:02d}.jpg"
                    frame.to_image().save(os.path.join(output_dir, frame_filename), quality=95)
                    saved_frames.append(frame_filename)
        except (av.FFmpegError, IndexError, OSError):
            return saved_frames  # không mở được file thì trả về phần đã có, không làm hỏng cả job
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

        # 2. Chạy suy luận video qua Qwen2.5-VL, chia đoạn nếu video dài
        chunks, frames_per_chunk = self._plan_video_chunks(video_path, fps)
        segment_reports = []

        duration = video_duration(video_path)
        if len(chunks) == 1:
            self._report(f"[*] Đang phân tích video: {os.path.basename(video_path)} (FPS lấy mẫu: {fps})...", 20)
            result = self._analyze_video_segment(video_path, prompt + timeline_instruction(0, duration),
                                                 fps, max_pixels,
                                                 progress=(20, 88))
        else:
            self._report(
                f"[*] Video dài, chia thành {len(chunks)} đoạn "
                f"(khoảng {frames_per_chunk} khung/đoạn)...",
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
                seg_lo = 15 + (i - 1) * 65 / len(chunks)
                seg_hi = 15 + i * 65 / len(chunks)
                segment_text = self._analyze_video_segment(
                    video_path, segment_prompt, fps, max_pixels, start=start, end=end,
                    progress=(seg_lo, seg_hi),
                )
                segment_reports.append((start, end, segment_text))

            self._report(f"[*] Đang tổng hợp {len(segment_reports)} đoạn thành báo cáo cuối...", 80)
            result = self._synthesize_timeline(segment_reports, prompt, progress=(80, 90))
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
        model_id=VL_MODEL_PATH,
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
            max_pixels=360 * 420,  # Giữ ở mức này để tối ưu bộ nhớ
            num_keyframes_saved=4  # Số khung hình trích xuất để lưu làm bằng chứng trong báo cáo
        )
        print("KẾT QUẢ PHÂN TÍCH VIDEO:")
        print(vid_res["result"])
    else:
        print(f"[!] Vui lòng đặt một video '{sample_video}' cùng thư mục để thử nghiệm video.")
