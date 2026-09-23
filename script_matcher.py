"""Bộ não khớp kịch bản: Qwen3-Reranker-4B.

Qwen3-Reranker là mô hình CHỈ ĐỌC CHỮ (text-only cross-encoder). Nó không nhìn được
ảnh/video. Vì vậy đường ống gồm hai tầng nối tiếp nhau:

    Tầng 1 (đã có)   - Qwen2.5-VL đọc ảnh/video  ->  mô tả bằng chữ, lưu ở vision_storage/
    Tầng 2 (file này) - Qwen3-Reranker chấm điểm từng cặp
                        (một dòng kịch bản, mô tả một cảnh quay) -> chọn + sắp xếp

Cách reranker chấm điểm: nó là một causal LM được huấn luyện để trả lời đúng một từ
"yes" hoặc "no". Ta đưa cặp (query, document) vào, đọc logit của token "yes" và "no"
ở vị trí cuối cùng, softmax trên hai token đó -> xác suất 0..1 = độ khớp.
"""

import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Bản GPTQ int4 (compressed-tensors) có sẵn trong AI Hub cục bộ — đã kiểm chứng chạy đúng
# trên MPS bằng transformers (weights được compressed-tensors tự giải nén khi cần, không đòi
# CUDA). Đổi model bằng biến môi trường RERANKER_MODEL_ID nếu muốn dùng bản khác.
RERANKER_MODEL_ID = os.environ.get(
    "RERANKER_MODEL_ID",
    "/Users/mac/.aihub/models/hf/hub/models--boboliu--Qwen3-Reranker-4B-W4A16-G128/"
    "snapshots/84d3701aa6a3a7e581389c666ec340f91e7936c6",
)

# Khung prompt chính thức của Qwen3-Reranker. Không được đổi chữ trong PREFIX/SUFFIX:
# mô hình được huấn luyện đúng định dạng này, sai một chi tiết là điểm lệch hẳn.
PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    '<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

DEFAULT_INSTRUCTION = (
    "Cho một dòng kịch bản và mô tả nội dung một cảnh quay có sẵn. "
    "Hãy đánh giá cảnh quay đó có dùng làm hình ảnh minh hoạ cho dòng kịch bản này được hay không. "
    "Ưu tiên sự trùng khớp về chủ thể, hành động, bối cảnh và cảm xúc."
)

MAX_LENGTH = 4096          # đủ cho một dòng kịch bản + một mô tả cảnh đã rút gọn
MAX_DOC_CHARS = 1600       # cắt bớt mô tả quá dài để giữ tốc độ và độ tập trung
LOW_SCORE = 0.30           # dưới ngưỡng này coi như "chưa có cảnh phù hợp"


# ==========================================
# TÁCH KỊCH BẢN
# ==========================================
# Dấu phẩy nằm trong tập dấu vì gõ nhầm "2," thay cho "2." rất hay gặp; thiếu nó thì dòng
# đó không được nhận là cảnh mới và bị gộp lặng lẽ vào cảnh phía trên.
MARKER_RE = re.compile(r"^\s*(?:cảnh|canh|scene|shot|c)?\s*\d+\s*[\.\):\-,–]\s+", re.I)


def split_script(text):
    """Tách kịch bản thành danh sách cảnh.

    Ba kiểu viết đều nhận ra được, xét theo thứ tự ưu tiên:
      1. Có đánh số ("1.", "Cảnh 2:", "Shot 3 -") -> mỗi số là một cảnh, kể cả khi
         giữa chúng có dòng trống hay một cảnh trải trên nhiều dòng.
      2. Không đánh số nhưng có dòng trống -> mỗi khối cách nhau bằng dòng trống là một cảnh.
      3. Không có gì -> mỗi dòng không rỗng là một cảnh.

    Kiểm tra đánh số trước dòng trống là điều bắt buộc: một kịch bản đánh số liền tù tì
    rồi chừa một dòng trống ở giữa sẽ bị kiểu tách theo dòng trống gộp nhầm nhiều cảnh
    thành một khối.
    """
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []

    lines = [ln.strip() for ln in text.split("\n")]
    marked = [i for i, ln in enumerate(lines) if ln and MARKER_RE.match(ln)]

    if len(marked) >= 2:
        # Gom mỗi dòng đánh số cùng các dòng mô tả đi kèm phía sau nó.
        blocks = []
        for pos, start in enumerate(marked):
            end = marked[pos + 1] if pos + 1 < len(marked) else len(lines)
            block = "\n".join(ln for ln in lines[start:end] if ln).strip()
            if block:
                blocks.append(block)
    else:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        if len(blocks) <= 1:
            blocks = [ln for ln in lines if ln]

    scenes = []
    for i, block in enumerate(blocks, 1):
        clean = MARKER_RE.sub("", block)
        clean = re.sub(r"^\s*[-*•]\s*", "", clean).strip()
        scenes.append({"index": i, "text": clean or block, "raw": block})
    return scenes


# ==========================================
# GOM ỨNG VIÊN TỪ KẾT QUẢ ĐÃ PHÂN TÍCH
# ==========================================
def _thumb_for(images, seg_index, seg_total):
    """Chọn ảnh đại diện cho một ứng viên. Video có nhiều keyframe -> lấy cái gần
    vị trí của đoạn đó nhất, để nhìn thumbnail là đoán được đúng cảnh."""
    pics = [n for n in (images or []) if re.search(r"\.(jpg|jpeg|png)$", n, re.I)]
    if not pics:
        return None
    if seg_index is None or not seg_total:
        return pics[0]
    pos = int((seg_index - 0.5) / seg_total * len(pics))
    return pics[min(max(pos, 0), len(pics) - 1)]


def collect_candidates(storage_dir, granularity="segment"):
    """Đọc mọi báo cáo đã lưu trong vision_storage/ thành danh sách cảnh quay ứng viên.

    granularity = "segment": mỗi đoạn video là một ứng viên riêng (cắt cảnh mịn hơn).
    granularity = "file":    mỗi file là một ứng viên duy nhất.
    """
    candidates = []
    if not os.path.isdir(storage_dir):
        return candidates

    for stem in sorted(os.listdir(storage_dir)):
        task_dir = os.path.join(storage_dir, stem)
        json_path = os.path.join(task_dir, f"{stem}.json")
        if not os.path.isdir(task_dir) or not os.path.isfile(json_path):
            continue
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        images = data.get("images", [])
        segments = data.get("segments", []) if granularity == "segment" else []
        common = {
            "stem": stem,
            "media_type": data.get("media_type", "video"),
            "source_file": data.get("source_file", stem),
        }

        if segments:
            for seg in segments:
                candidates.append({
                    **common,
                    "id": f"{stem}#seg{seg['index']}",
                    "segment_index": seg["index"],
                    "start_sec": seg.get("start_sec"),
                    "end_sec": seg.get("end_sec"),
                    "text": (seg.get("text") or "").strip(),
                    "thumbnail": _thumb_for(images, seg["index"], len(segments)),
                })
        else:
            candidates.append({
                **common,
                "id": stem,
                "segment_index": None,
                "start_sec": None,
                "end_sec": None,
                "text": (data.get("result") or "").strip(),
                "thumbnail": _thumb_for(images, None, None),
            })

    return [c for c in candidates if c["text"]]


def build_result(model_name, scenes, candidates, plan, instruction=None, reuse="segment", shortlisted=None):
    """Đóng gói kết quả xếp cảnh theo đúng khuôn mà plan_to_markdown() và giao diện mong đợi."""
    return {
        "model": model_name,
        "instruction": instruction,
        "reuse": reuse,
        "num_scenes": len(scenes),
        "num_candidates": len(candidates),
        "shortlisted": shortlisted,
        "plan": plan,
    }


# ==========================================
# XẾP CẢNH VÀO KỊCH BẢN
# ==========================================
def assign(scores, scenes, candidates, reuse="segment", top_k=3):
    """Ghép mỗi cảnh kịch bản với một cảnh quay, dựa trên ma trận điểm.

    reuse = "free"    : một cảnh quay được dùng lại bao nhiêu lần cũng được.
    reuse = "segment" : mỗi đoạn chỉ dùng một lần (vẫn cho phép hai đoạn cùng một file).
    reuse = "file"    : mỗi file nguồn chỉ dùng một lần.

    Với hai chế độ không tái sử dụng, ta xếp tham lam theo điểm toàn cục: duyệt mọi cặp
    (cảnh kịch bản, cảnh quay) theo điểm giảm dần và chốt cặp nào còn trống cả hai phía.
    Cách này tránh được lỗi của kiểu "duyệt lần lượt từng dòng": dòng đầu tiên chiếm mất
    cảnh quay mà dòng sau cần đến hơn nhiều.
    """
    n_scenes, n_cands = len(scenes), len(candidates)
    results = [None] * n_scenes

    def alternatives(si, exclude_id=None):
        order = sorted(range(n_cands), key=lambda ci: scores[si][ci], reverse=True)
        out = []
        for ci in order:
            if candidates[ci]["id"] == exclude_id:
                continue
            out.append({"candidate": candidates[ci], "score": round(float(scores[si][ci]), 4)})
            if len(out) >= top_k:
                break
        return out

    if reuse == "free":
        for si in range(n_scenes):
            ci = max(range(n_cands), key=lambda c: scores[si][c])
            results[si] = (ci, scores[si][ci])
    else:
        pairs = sorted(
            ((scores[si][ci], si, ci) for si in range(n_scenes) for ci in range(n_cands)),
            reverse=True,
        )
        used_keys = set()
        for score, si, ci in pairs:
            if results[si] is not None:
                continue
            key = candidates[ci]["stem"] if reuse == "file" else candidates[ci]["id"]
            if key in used_keys:
                continue
            results[si] = (ci, score)
            used_keys.add(key)

    plan = []
    for si, scene in enumerate(scenes):
        picked = results[si]
        if picked is None:
            # Hết cảnh quay chưa dùng: báo thiếu thay vì lặng lẽ bỏ trống.
            plan.append({
                "scene_index": scene["index"],
                "scene_text": scene["text"],
                "match": None,
                "score": None,
                "confidence": "thieu_canh",
                "alternatives": alternatives(si),
            })
            continue
        ci, score = picked
        score = float(score)
        plan.append({
            "scene_index": scene["index"],
            "scene_text": scene["text"],
            "match": candidates[ci],
            "score": round(score, 4),
            # Điểm âm = cảnh quay này bị tầng lọc nhanh gạt ra, reranker chưa hề chấm nó;
            # nó chỉ được lấy để trám chỗ. Nói thẳng thay vì gọi chung là "yếu".
            "confidence": (
                "chua_cham" if score < 0
                else "tot" if score >= 0.6
                else "tam" if score >= LOW_SCORE
                else "yeu"
            ),
            "alternatives": alternatives(si, exclude_id=candidates[ci]["id"]),
        })
    return plan


# ==========================================
# MÔ HÌNH
# ==========================================
class ScriptMatcher:
    """Bọc Qwen3-Reranker. Nạp chậm (chỉ khi thực sự chấm điểm) và trả lại VRAM được."""

    def __init__(self, model_id=RERANKER_MODEL_ID, progress_callback=None):
        self.model_id = model_id
        self.progress_callback = progress_callback
        self.model = None
        self.tokenizer = None

    def _report(self, message, percent=None):
        print(message, flush=True)
        if self.progress_callback:
            self.progress_callback(message, percent)

    def load(self):
        if self.model is not None:
            return
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._report(f"[*] Đang nạp bộ não khớp kịch bản {self.model_id} lên {device}...", 5)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        ).to(device).eval()

        # Hai token quyết định điểm số. Lấy id một lần để không phải tra cứu mỗi batch.
        self.token_yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_no = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self._report("[*] Đã nạp xong bộ não khớp kịch bản.", 10)

    def unload(self):
        """Trả bộ nhớ lại cho mô hình thị giác. 16GB RAM hợp nhất không đủ cho cả hai cùng lúc."""
        if self.model is None:
            return
        del self.model
        self.model = None
        self.tokenizer = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _format_pair(self, query, document, instruction):
        doc = document[:MAX_DOC_CHARS]
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    @torch.no_grad()
    def score(self, query, documents, instruction=DEFAULT_INSTRUCTION, batch_size=4):
        """Trả về xác suất khớp (0..1) của từng document so với một query."""
        self.load()
        out = []
        for start in range(0, len(documents), batch_size):
            chunk = documents[start:start + batch_size]
            pairs = [self._format_pair(query, d, instruction) for d in chunk]

            budget = MAX_LENGTH - len(self.prefix_ids) - len(self.suffix_ids)
            enc = self.tokenizer(
                pairs,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=budget,
            )
            enc["input_ids"] = [self.prefix_ids + ids + self.suffix_ids for ids in enc["input_ids"]]
            enc = self.tokenizer.pad(enc, padding=True, return_tensors="pt", max_length=MAX_LENGTH)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}

            # padding_side="left" nên vị trí -1 luôn là token thật cuối cùng của mỗi dòng.
            logits = self.model(**enc).logits[:, -1, :]
            two = torch.stack([logits[:, self.token_no], logits[:, self.token_yes]], dim=1).float()
            probs = torch.nn.functional.log_softmax(two, dim=1).exp()[:, 1]
            out.extend(probs.tolist())

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return out

    def match_script(self, script_text, candidates, instruction=DEFAULT_INSTRUCTION,
                     reuse="segment", top_k=3, shortlist=None, fallback_scores=None):
        """Chấm điểm (kịch bản x cảnh quay) rồi xếp cảnh theo đúng thứ tự kịch bản.

        shortlist (tuỳ chọn): với mỗi cảnh kịch bản, danh sách chỉ số cảnh quay đáng chấm
        — do tầng nhúng đa phương thức lọc sẵn. Có nó thì số lần chạy reranker giảm từ
        (số cảnh × cả kho) xuống (số cảnh × top-K), tiết kiệm rất nhiều khi kho lớn.

        fallback_scores: ma trận cosine của tầng nhúng, dùng cho những cảnh quay KHÔNG
        nằm trong danh sách ngắn. Chúng được dìm xuống vùng âm (-1 + 0.01×cos) nên không
        bao giờ vượt mặt một cặp đã qua reranker, nhưng vẫn giữ đúng thứ tự tương đối để
        còn chỗ trám khi chế độ "không dùng lại" làm cạn ứng viên.
        """
        scenes = split_script(script_text)
        if not scenes:
            raise ValueError("Kịch bản trống. Hãy nhập ít nhất một dòng.")
        if not candidates:
            raise ValueError(
                "Chưa có cảnh quay nào đã phân tích. Hãy chạy phân tích ảnh/video ở tab đầu tiên trước."
            )

        self.load()
        documents = [c["text"] for c in candidates]
        scores = []
        for i, scene in enumerate(scenes):
            picked = None if shortlist is None else shortlist[i]
            n_scored = len(documents) if picked is None else len(picked)
            self._report(
                f"[*] Đang chấm điểm cảnh {i + 1}/{len(scenes)} với {n_scored} cảnh quay...",
                10 + 85 * (i + 1) / len(scenes),
            )
            if picked is None:
                scores.append(self.score(scene["text"], documents, instruction))
                continue

            sub = self.score(scene["text"], [documents[j] for j in picked], instruction)
            base = fallback_scores[i] if fallback_scores else [0.0] * len(documents)
            row = [-1.0 + 0.01 * base[j] for j in range(len(documents))]
            for j, value in zip(picked, sub):
                row[j] = value
            scores.append(row)

        plan = assign(scores, scenes, candidates, reuse=reuse, top_k=top_k)
        self._report("[*] Đã xếp xong kịch bản.", 100)
        return build_result(
            self.model_id, scenes, candidates, plan, instruction=instruction, reuse=reuse,
            shortlisted=None if shortlist is None else len(shortlist[0]),
        )


# ==========================================
# XUẤT KẾ HOẠCH DỰNG
# ==========================================
def plan_to_markdown(result, title="Kế hoạch dựng theo kịch bản"):
    lines = [f"# {title}", ""]
    lines.append(f"- **Mô hình khớp:** `{result['model']}`")
    lines.append(f"- **Số cảnh kịch bản:** {result['num_scenes']}")
    lines.append(f"- **Số cảnh quay ứng viên:** {result['num_candidates']}")
    lines.append(f"- **Chế độ tái sử dụng:** {result['reuse']}")
    lines.append("")
    lines.append("| # | Dòng kịch bản | Cảnh quay được chọn | Thời điểm | Điểm khớp |")
    lines.append("|---|---|---|---|---|")
    for item in result["plan"]:
        m = item["match"]
        if m is None:
            lines.append(f"| {item['scene_index']} | {item['scene_text']} | _CHƯA CÓ CẢNH PHÙ HỢP_ | | |")
            continue
        when = ""
        if m.get("start_sec") is not None:
            when = f"{m['start_sec']:.0f}s - {m['end_sec']:.0f}s"
        lines.append(
            f"| {item['scene_index']} | {item['scene_text']} | {m['source_file']} | {when} | "
            f"{item['score']:.3f} ({item['confidence']}) |"
        )

    lines.append("")
    lines.append("## Chi tiết từng cảnh")
    for item in result["plan"]:
        lines.append("")
        lines.append(f"### Cảnh {item['scene_index']}: {item['scene_text']}")
        m = item["match"]
        if m is None:
            lines.append("")
            lines.append("**Chưa có cảnh quay nào phù hợp** — cần quay bổ sung.")
        else:
            lines.append("")
            lines.append(f"- **Chọn:** {m['source_file']} (`{m['id']}`) — điểm {item['score']:.3f}")
            lines.append(f"- **Mô tả cảnh quay:** {m['text'][:600]}")
        if item["alternatives"]:
            lines.append("- **Phương án thay thế:**")
            for alt in item["alternatives"]:
                lines.append(
                    f"  - {alt['candidate']['source_file']} (`{alt['candidate']['id']}`) — điểm {alt['score']:.3f}"
                )
    return "\n".join(lines) + "\n"
