"""Cầu nối từ app chính sang sidecar nhúng đa phương thức (embedding_sidecar/server.py).

Chỉ dùng thư viện chuẩn của Python: app chính bị khoá phiên bản chặt, thêm dependency
mới là thêm rủi ro không đáng có cho một việc chỉ là gọi HTTP.

Vai trò trong đường ống cascade:

    Kho cảnh quay (có thể hàng trăm)
        │  nhúng ảnh + mô tả  ->  vector, cache lại trên đĩa
        ▼
    Lọc nhanh top-K bằng cosine (mili-giây, không tốn GPU sau lần nhúng đầu)
        ▼
    Qwen3-Reranker chấm kỹ từng cặp trong danh sách ngắn (chính xác, nhưng chậm)

Không có sidecar thì mọi thứ vẫn chạy: app tự quay về chấm toàn bộ bằng reranker.
"""

import hashlib
import json
import math
import os
import urllib.error
import urllib.request

SIDECAR_URL = os.environ.get("EMBEDDING_SIDECAR_URL", "http://127.0.0.1:8011")
CACHE_VERSION = 1


def _post(path, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SIDECAR_URL + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def health(timeout=1.5):
    """Trả về thông tin sidecar, hoặc None nếu chưa bật. Không bao giờ ném lỗi."""
    try:
        with urllib.request.urlopen(SIDECAR_URL + "/health", timeout=timeout) as resp:
            return json.load(resp)
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def unload(timeout=30):
    """Bảo sidecar nhả VRAM. Gọi trước khi nạp mô hình lớn ở app chính."""
    try:
        return _post("/unload", {}, timeout)
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def embed(items, timeout=900, batch_size=4):
    """items: danh sách dict {text?, image?, video?, instruction?} -> danh sách vector.

    Kiem tra so luong vector tra ve khop voi so muc da gui: vendor script cua
    Qwen3-VL-Embedding co the am tham gop CA MOT LO ve mot muc "NULL" gia neu chi MOT
    item trong lo loi (vd: video khong doc duoc), lam so vector tra ve it hon han so
    yeu cau. Neu khong bat o day, cac ham goi ben tren (embed_candidates,
    embed_raw_candidates, embed_scenes) se gan nham vector cho sai candidate/scene ma
    khong bao loi gi - vector cua vi tri i se thanh mot gia tri hoan toan khong lien quan
    thay vi bao loi ngay, va co the lam sai ket qua khop kich ban ma khong ai biet."""
    if not items:
        return []
    result = _post("/embed", {"items": items, "batch_size": batch_size}, timeout)
    embeddings = result["embeddings"]
    if len(embeddings) != len(items):
        raise RuntimeError(
            f"Sidecar trả về {len(embeddings)} vector cho {len(items)} mục đã gửi — số lượng "
            f"không khớp. Thường do một ảnh/video trong lô bị lỗi khi đọc (file hỏng, codec lạ) "
            f"khiến sidecar gộp cả lô lại. Xem log console của sidecar để biết mục nào gây lỗi."
        )
    return embeddings


# ==========================================
# CACHE VECTOR CỦA CẢNH QUAY
# ==========================================
def _cache_path(storage_dir):
    return os.path.join(storage_dir, "_embeddings", "cache.json")


def _load_cache(storage_dir):
    path = _cache_path(storage_dir)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") == CACHE_VERSION:
            return data.get("items", {})
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_cache(storage_dir, items):
    path = _cache_path(storage_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": CACHE_VERSION, "items": items}, f)


def _fingerprint(candidate, image_path):
    """Đổi khoá cache khi nội dung đổi: mô tả được phân tích lại, hoặc ảnh bị thay."""
    h = hashlib.sha1()
    h.update(candidate["id"].encode("utf-8"))
    h.update(candidate["text"][:2000].encode("utf-8"))
    if image_path and os.path.isfile(image_path):
        h.update(str(os.path.getmtime(image_path)).encode("utf-8"))
    return h.hexdigest()


def embed_candidates(candidates, storage_dir, report=None):
    """Nhúng từng cảnh quay (ảnh đại diện + mô tả), tận dụng cache trên đĩa.

    Ảnh đại diện là thứ khiến tầng này hơn hẳn so khớp thuần chữ: vector mang nội dung
    hình ảnh thật, không chỉ những gì mô tả kịp diễn đạt thành lời.
    """
    cache = _load_cache(storage_dir)
    vectors = [None] * len(candidates)
    todo, todo_idx = [], []

    for i, c in enumerate(candidates):
        image_path = None
        if c.get("thumbnail"):
            image_path = os.path.join(storage_dir, c["stem"], c["thumbnail"])
            if not os.path.isfile(image_path):
                image_path = None
        key = _fingerprint(c, image_path)
        hit = cache.get(key)
        if hit:
            vectors[i] = hit
            continue
        item = {"text": c["text"][:1200]}
        if image_path:
            item["image"] = image_path
        todo.append((key, item))
        todo_idx.append(i)

    if todo:
        if report:
            report(f"[*] Đang nhúng {len(todo)} cảnh quay mới (còn lại lấy từ cache)...", None)
        new_vectors = embed([item for _, item in todo])
        for (key, _), idx, vec in zip(todo, todo_idx, new_vectors):
            vectors[idx] = vec
            cache[key] = vec
        _save_cache(storage_dir, cache)
    elif report:
        report("[*] Mọi cảnh quay đã có vector trong cache — bỏ qua bước nhúng.", None)

    return vectors


# ==========================================
# CACHE VECTOR CỦA FILE THÔ (che độ "Chỉ Embedding" — bỏ qua Qwen2.5-VL)
# ==========================================
# Cache riêng với _cache_path() ở trên: vân tay ở đây dựa trên đường dẫn + mtime + kích
# thước file gốc, không dựa trên một mô tả bằng chữ nào (vì không có).
def _raw_cache_path(storage_dir):
    return os.path.join(storage_dir, "_embeddings_raw", "cache.json")


def _load_raw_cache(storage_dir):
    path = _raw_cache_path(storage_dir)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") == CACHE_VERSION:
            return data.get("items", {})
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_raw_cache(storage_dir, items):
    path = _raw_cache_path(storage_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": CACHE_VERSION, "items": items}, f)


def _fingerprint_raw(path):
    h = hashlib.sha1()
    h.update(path.encode("utf-8"))
    try:
        st = os.stat(path)
        h.update(str(st.st_mtime).encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
    except OSError:
        pass
    return h.hexdigest()


def embed_raw_candidates(candidates, storage_dir, report=None):
    """Nhúng THẲNG ảnh/video thô trong media_input/, không cần mô tả bằng chữ.

    Dùng cho chế độ "Chỉ Embedding": mỗi candidate phải có "media_path" (đường dẫn file
    gốc) và "media_type" ("image"/"video") — xem script_matcher.collect_raw_candidates().
    """
    cache = _load_raw_cache(storage_dir)
    vectors = [None] * len(candidates)
    todo, todo_idx = [], []

    for i, c in enumerate(candidates):
        path = c["media_path"]
        key = _fingerprint_raw(path)
        hit = cache.get(key)
        if hit:
            vectors[i] = hit
            continue
        item = {"image": path} if c["media_type"] == "image" else {"video": path}
        todo.append((key, item))
        todo_idx.append(i)

    if todo:
        if report:
            report(f"[*] Đang nhúng {len(todo)} file mới (còn lại lấy từ cache)...", None)
        # batch_size=1: nhung video tho (khong phai thumbnail nho) tung file mot. Nhung
        # nhieu video cung luc trong mot batch tung gay OutOfMemoryError (13.44 GiB) o
        # attention cua vision tower - chua ro Qwen3-VL co xu ly rieng tung video trong
        # batch bang cu_seqlens hay khong, nen chon phuong an chac chan an toan.
        new_vectors = embed([item for _, item in todo], batch_size=1)
        for (key, _), idx, vec in zip(todo, todo_idx, new_vectors):
            vectors[idx] = vec
            cache[key] = vec
        _save_raw_cache(storage_dir, cache)
    elif report:
        report("[*] Mọi file đã có vector trong cache — bỏ qua bước nhúng.", None)

    return vectors


def embed_scenes(scenes, instruction=None):
    items = [{"text": s["text"]} for s in scenes]
    if instruction:
        for item in items:
            item["instruction"] = instruction
    return embed(items)


def cosine_matrix(scene_vectors, candidate_vectors):
    """Vector trả về từ sidecar đã chuẩn hoá L2, nhưng vẫn chia lại chuẩn cho chắc."""
    def norm(v):
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    svs = [norm(v) for v in scene_vectors]
    cvs = [norm(v) for v in candidate_vectors]
    return [[sum(a * b for a, b in zip(sv, cv)) for cv in cvs] for sv in svs]


def shortlist(cos_scores, top_k):
    """Với mỗi cảnh kịch bản, lấy chỉ số của top_k cảnh quay giống nhất."""
    return [
        sorted(range(len(row)), key=lambda ci: row[ci], reverse=True)[:top_k]
        for row in cos_scores
    ]
