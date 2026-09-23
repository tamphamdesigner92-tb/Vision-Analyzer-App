"""Hỏi đáp về nội dung phim - không cần model embedding.

    1. LLM sinh từ khoá cho câu hỏi (tiếng Việt + ngôn ngữ gốc của phim) -> tra FTS5 trong index.sqlite.
    2. LLM đọc tóm tắt các chương (vài nghìn token cho cả phim) -> chọn chương liên quan.
    3. Đi xuống cây chương -> đoạn -> cảnh, nạp ĐẦY ĐỦ file cảnh + lời thoại của các nhánh trúng
       (cảnh trúng từ khoá được ưu tiên), trong giới hạn token của hồ sơ phần cứng.
    4. LLM trả lời kèm mốc [hh:mm:ss] và id cảnh để bấm mở.
Mỗi câu hỏi được lưu vào 08_hoi_dap/qa_<thời gian>.json (câu hỏi, câu trả lời, các cảnh đã dùng).
"""

import datetime
import time

from film import index as film_index
from film import store
from film.project import now_iso, read_json, write_json_atomic
from film.stage_synthesis import SYSTEM, render_scene

KEYWORD_SCHEMA = {"type": "object", "properties": {"tu_khoa": {"type": "array", "items": {"type": "string", "maxLength": 60},
                                                             "minItems": 1, "maxItems": 10}},
                  "required": ["tu_khoa"]}


def _chapter_schema(ids):
    return {"type": "object", "properties": {"chuong": {"type": "array", "items": {"type": "string", "enum": ids}, "maxItems": 4}},
            "required": ["chuong"]}


def _answer_schema(scene_ids):
    return {"type": "object", "properties": {
        "tra_loi": {"type": "string", "maxLength": 4000},
        "canh_trich_dan": {"type": "array", "items": {"type": "string", "enum": scene_ids or ["khong_co"]}, "maxItems": 12},
        "do_tin_cay": {"type": "string", "enum": ["cao", "vua", "thap"]}},
        "required": ["tra_loi", "canh_trich_dan", "do_tin_cay"]}


def answer(project, server, question, max_tokens_ctx=40000, report=None):
    say = report or (lambda m: None)
    t0 = time.time()
    lang = (read_json(project.path("01_loi_thoai", "dialogue.json"), {}) or {}).get("lang") or "không rõ"

    say("Tìm từ khoá...")
    kw, _ = server.chat(
        f"Câu hỏi về một bộ phim: \"{question}\"\nLời thoại của phim bằng ngôn ngữ: {lang}.\n"
        "Liệt kê 4–10 từ khoá ngắn để tìm trong mô tả cảnh (tiếng Việt) và trong lời thoại (ngôn ngữ gốc): "
        "tên nhân vật, đồ vật, địa điểm, hành động.", KEYWORD_SCHEMA, max_tokens=300, system=SYSTEM, fail_dir=project.path("10_nhat_ky"))
    hits = []
    for k in kw["tu_khoa"][:10]:
        hits += film_index.search(project, k, limit=15)
    hit_scenes = []
    for h in sorted(hits, key=lambda h: -h["score"]):
        sid = h["node_id"]
        if sid and sid.startswith("scene_") and sid not in hit_scenes:
            hit_scenes.append(sid)

    chapters = [store.read_node(project, "chapter", c) for c in store.list_nodes(project, "chapter")]
    say("Chọn chương liên quan...")
    picked = []
    if chapters:
        out, _ = server.chat(
            f"Câu hỏi: \"{question}\"\n\nTóm tắt các chương của phim:\n\n"
            + "\n\n".join(f"{c['id']} [{store.hms(c['start'], False)}–{store.hms(c['end'], False)}] {c['tieu_de']}: {c['tom_tat']}"
                          for c in chapters)
            + "\n\nChọn các chương (tối đa 4) có khả năng chứa câu trả lời.",
            _chapter_schema([c["id"] for c in chapters]), max_tokens=200, system=SYSTEM, fail_dir=project.path("10_nhat_ky"))
        picked = out["chuong"][:4]

    # Cảnh ứng viên: trúng từ khoá trước, rồi toàn bộ cảnh của các chương được chọn.
    seqs = {q: store.read_node(project, "seq", q) for q in store.list_nodes(project, "seq")}
    ordered = list(hit_scenes)
    for c in chapters:
        if c["id"] in picked:
            for q in c["children"]:
                ordered += [s for s in seqs[q]["children"] if s not in ordered]
    if not ordered:
        ordered = store.list_nodes(project, "scene")

    context, used, budget = [], [], max_tokens_ctx
    for sid in ordered:
        sc = store.read_node(project, "scene", sid)
        text = render_scene(sc)
        cost = len(text) // 3            # ước lượng thô, đủ để giữ trong giới hạn
        if cost > budget:
            continue
        context.append((sc["start"], text))
        used.append(sid)
        budget -= cost
    context.sort()
    chapter_notes = "\n".join(f"- {c['id']} [{store.hms(c['start'], False)}]: {c['tieu_de']}" for c in chapters)

    say(f"Trả lời dựa trên {len(used)} cảnh...")
    out, timing = server.chat(
        f"Câu hỏi về phim \"{project.name}\": \"{question}\"\n\nMục lục chương:\n{chapter_notes}\n\n"
        "Tư liệu các cảnh liên quan (xếp theo thời gian):\n\n" + "\n\n".join(t for _, t in context)
        + "\n\nTrả lời bằng tiếng Việt, dựa HOÀN TOÀN vào tư liệu trên. Mỗi ý quan trọng ghi kèm mốc "
          "[hh:mm:ss] của cảnh chứng minh. Nếu tư liệu không đủ để trả lời thì nói rõ là không tìm thấy.",
        _answer_schema(used), max_tokens=2500, system=SYSTEM, fail_dir=project.path("10_nhat_ky"))

    record = {
        "question": question, "answer": out["tra_loi"], "confidence": out["do_tin_cay"],
        "cited_scenes": [{"id": s, "start_hms": store.hms(store.read_node(project, "scene", s)["start"], False)}
                         for s in out["canh_trich_dan"] if s in used],
        "keywords": kw["tu_khoa"], "chapters_picked": picked, "scenes_used": used,
        "created_at": now_iso(), "seconds": round(time.time() - t0, 1),
        "prompt_tokens": timing.get("prompt_n"),
    }
    name = f"qa_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json_atomic(project.path("08_hoi_dap", name), record)
    record["file"] = f"08_hoi_dap/{name}"
    return record
