"""Bước [5] Tổng hợp: cảnh -> ĐOẠN (tầng 2) -> CHƯƠNG (tầng 3) -> CẢ PHIM (tầng 4) + sổ nhân vật.

"Bộ não" là Qwen3-30B-A3B qua llama-server (xem llm_client.py). Mọi đầu ra đều bị ép theo JSON
schema; các trường trỏ về tầng dưới (id cảnh, id đoạn, id nhân vật) dùng enum trong schema nên
model KHÔNG THỂ bịa ra một id không tồn tại.

Chạy theo từng chương, mỗi chương là một checkpoint:
  - Trong một chương: sinh từng đoạn (05_doan/seq_XXX.json), rồi tóm tắt chương (06_chuong/ch_XX.json).
  - Sau mỗi chương, lưu "câu chuyện tới giờ" + sổ nhân vật vào 07_phim/_state.json, nên dừng
    giữa chừng thì chạy lại tiếp đúng chương đang dở, và các đoạn đã có file được dùng lại.
  - File đoạn/chương không bao giờ bị ghi đè (store.write_node).
Cuối cùng: 07_phim/film.json + film_report.md + characters.json, 11_xuat/ (cho tab 2, .srt),
và dựng lại chỉ mục 09_chi_muc/index.sqlite.
"""

import json
import os
import re
import shutil
import time

from film import index as film_index
from film import stage_io, store
from film.project import now_iso, read_json, write_json_atomic

SYNTH_VERSION = "synth-v2"
SEQ_TARGET_SEC, SEQ_MAX_SCENES = 240, 6
CH_TARGET_SEC, CH_MAX_SEQS = 780, 5
SYSTEM = ("Bạn là biên tập viên phim cẩn thận. Luôn viết bằng tiếng Việt, chính xác, trung thực, không bịa "
          "chi tiết không có trong tư liệu. Chỉ dùng tên nhân vật khi tên đó xuất hiện trong lời thoại hoặc "
          "chữ trên hình; chưa biết tên thì gọi bằng đặc điểm nhận dạng (vd: 'người đàn ông áo khoác đen'). "
          "Mốc thời gian luôn ghi dạng hh:mm:ss theo giờ phim. KHÔNG bao giờ ghi mã nội bộ (scene_0001, "
          "seq_001, ch_01, nv_01…) vào tiêu đề hay văn kể chuyện - đó chỉ là mã để trỏ, người đọc không cần.")


# ==========================================
# CHIA NHÓM
# ==========================================
def group(items, target_sec, max_items):
    """Gộp các mục liên tiếp (có start/end) thành nhóm ~target_sec giây, tối đa max_items mục."""
    groups, cur = [], []
    for it in items:
        cur.append(it)
        if cur[-1]["end"] - cur[0]["start"] >= target_sec or len(cur) >= max_items:
            groups.append(cur)
            cur = []
    if cur:
        if groups and cur[-1]["end"] - cur[0]["start"] < target_sec / 3:
            groups[-1].extend(cur)     # mẩu cuối quá ngắn thì nhập vào nhóm trước
        else:
            groups.append(cur)
    return groups


def parse_hms(text, lo, hi, fallback):
    """"hh:mm:ss" / "mm:ss" -> giây, kẹp trong [lo, hi]; hỏng thì dùng fallback."""
    m = re.match(r"^\s*(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,]\d+)?\s*$", str(text or ""))
    if not m:
        return fallback
    t = int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    return fallback if not (lo - 2 <= t <= hi + 2) else min(max(t, lo), hi)


# ==========================================
# DỰNG PROMPT
# ==========================================
def render_scene(sc):
    c = sc["card"]
    rows = [f"### {sc['id']} [{store.hms(sc['start'], False)}–{store.hms(sc['end'], False)}]"]
    if c.get("boi_canh"):
        rows.append(f"Bối cảnh: {c['boi_canh']}")
    rows.append(f"Mô tả: {c.get('mo_ta', '')}")
    if c.get("nhan_vat"):
        rows.append("Nhân vật: " + "; ".join(
            f"{n.get('goi_la', '?')} ({n.get('ngoai_hinh', '')}; {n.get('hanh_dong', '')})" for n in c["nhan_vat"]))
    if sc.get("events"):
        rows.append("Sự kiện: " + "; ".join(f"[{e['t_hms'][:8]}] {e['mo_ta']}" for e in sc["events"]))
    if c.get("chu_tren_man_hinh"):
        rows.append(f"Chữ trên hình: {c['chu_tren_man_hinh']}")
    if c.get("am_thanh_khong_khi"):
        rows.append(f"Không khí: {c['am_thanh_khong_khi']}")
    if sc.get("dialogue"):
        rows.append("Lời thoại (nguyên văn):")
        rows += [f"[{store.hms(d['start'], False)}] {d['text']}" for d in sc["dialogue"]]
    return "\n".join(rows)


def render_registry(registry):
    if not registry:
        return "(chưa có)"
    return "\n".join(f"- {c['id']}: {c['ten']}" + (f" (còn gọi: {', '.join(c['bi_danh'])})" if c.get("bi_danh") else "")
                     + f" — {c.get('mo_ta', '')}; vai trò: {c.get('vai_tro', '')}" for c in registry)


def _s(max_len):
    """Chuỗi có trần độ dài - không có trần thì model đôi khi viết mãi không dừng."""
    return {"type": "string", "maxLength": max_len}


def _a(item, max_items, min_items=0):
    """Mảng có trần số phần tử - chặn kiểu lỗi model lặp lại một mục hàng trăm lần."""
    return {"type": "array", "items": item, "minItems": min_items, "maxItems": max_items}


def _enum(values):
    return {"type": "string", "enum": values}


def seq_schema(scene_ids):
    return {
        "type": "object",
        "properties": {
            "tieu_de": _s(120),
            "tom_tat": _s(1800),
            "su_kien": _a({"type": "object", "properties": {
                "moc": _s(12), "canh": _enum(scene_ids), "mo_ta": _s(300)},
                "required": ["moc", "canh", "mo_ta"]}, 12, 1),
            "nhan_vat": _a({"type": "object", "properties": {
                "ten": _s(80), "vai_tro_trong_doan": _s(200),
                "canh": _a(_enum(scene_ids), len(scene_ids))},
                "required": ["ten", "vai_tro_trong_doan", "canh"]}, 10),
            "dia_diem": _a(_s(100), 6),
            "buoc_ngoat": _s(400),
        },
        "required": ["tieu_de", "tom_tat", "su_kien", "nhan_vat", "dia_diem", "buoc_ngoat"],
    }


def chapter_schema(seq_ids, char_ids):
    return {
        "type": "object",
        "properties": {
            "tieu_de": _s(120),
            "tom_tat": _s(2400),
            "diem_chinh": _a({"type": "object", "properties": {
                "moc": _s(12), "doan": _enum(seq_ids), "mo_ta": _s(300)},
                "required": ["moc", "doan", "mo_ta"]}, 10, 1),
            "buoc_ngoat": _s(500),
            "cau_chuyen_toi_gio": _s(2000),
            "nhan_vat": _a({"type": "object", "properties": {
                "id": _enum(char_ids + ["moi"]),
                "ten": _s(80), "bi_danh": _a(_s(80), 4),
                "mo_ta_nhan_dang": _s(300), "vai_tro": _s(200),
                "doan": _a(_enum(seq_ids), len(seq_ids))},
                "required": ["id", "ten", "bi_danh", "mo_ta_nhan_dang", "vai_tro", "doan"]}, 15),
        },
        "required": ["tieu_de", "tom_tat", "diem_chinh", "buoc_ngoat", "cau_chuyen_toi_gio", "nhan_vat"],
    }


def film_schema(char_ids):
    return {
        "type": "object",
        "properties": {
            "tom_tat_ngan": _s(900),
            "tom_tat_day_du": _s(6000),
            "the_loai": _s(120),
            "khong_khi": _s(200),
            "cau_truc": _a({"type": "object", "properties": {
                "phan": _s(80), "tu": _s(12), "den": _s(12), "mo_ta": _s(600)},
                "required": ["phan", "tu", "den", "mo_ta"]}, 8, 1),
            "tuyen_nhan_vat": _a({"type": "object", "properties": {
                "id": _enum(char_ids or ["khong_co"]), "dien_bien": _s(800)},
                "required": ["id", "dien_bien"]}, 15),
            "chu_de": _a(_s(150), 8),
        },
        "required": ["tom_tat_ngan", "tom_tat_day_du", "the_loai", "khong_khi", "cau_truc", "tuyen_nhan_vat", "chu_de"],
    }


# ==========================================
# SỔ NHÂN VẬT
# ==========================================
def _is_alias(text):
    """Bí danh là CÁCH GỌI ngắn ("áo hoodie vàng", "Mike"), không phải câu tả hành động mà model
    hay nhét nhầm vào ("ngồi bên máy giặt, hút thuốc…")."""
    return bool(text) and len(text) <= 40 and "," not in text


def merge_characters(registry, updates, seq_children):
    """Gộp cập nhật của một chương vào sổ nhân vật. "moi" -> cấp id mới nv_XX."""
    by_id = {c["id"]: c for c in registry}
    for u in updates:
        seqs = list(dict.fromkeys(u.get("doan", [])))
        scenes = [s for q in seqs for s in seq_children.get(q, [])]
        cid = u.get("id")
        if cid not in by_id:
            cid = f"nv_{len(registry) + 1:02d}"
            c = {"id": cid, "ten": u["ten"], "bi_danh": [], "mo_ta": u.get("mo_ta_nhan_dang", ""),
                 "vai_tro": u.get("vai_tro", ""), "xuat_hien": []}
            registry.append(c)
            by_id[cid] = c
        c = by_id[cid]
        if u.get("ten") and u["ten"] != c["ten"]:
            if c["ten"] not in c["bi_danh"]:
                c["bi_danh"].append(c["ten"])     # tên cũ (thường là mô tả) thành bí danh
            c["ten"] = u["ten"]
        for alias in u.get("bi_danh", []):
            if _is_alias(alias) and alias != c["ten"] and alias not in c["bi_danh"]:
                c["bi_danh"].append(alias)
        if len(u.get("mo_ta_nhan_dang", "")) > len(c.get("mo_ta", "")):
            c["mo_ta"] = u["mo_ta_nhan_dang"]
        if u.get("vai_tro"):
            c["vai_tro"] = u["vai_tro"]
        for nid in seqs + scenes:
            if nid not in c["xuat_hien"]:
                c["xuat_hien"].append(nid)
    return registry


def dedupe_characters(server, project, registry, fail_dir):
    """Một lượt LLM tìm những mục trong sổ nhân vật thực ra là cùng một người, rồi gộp lại.

    Mô tả của model thị giác thay đổi theo góc máy/ánh sáng, nên cùng một người dễ thành 2–3
    mục ("áo vàng", "hoodie vàng", "tóc xoăn áo vàng"). Gộp: giữ id đầu, nhập bí danh + nơi
    xuất hiện của các id kia vào, rồi xoá các id kia khỏi sổ."""
    ids = [c["id"] for c in registry]
    schema = {"type": "object", "properties": {"nhom_trung": _a({"type": "object", "properties": {
        "giu": _enum(ids), "gop_vao": _a(_enum(ids), len(ids), 1), "ly_do": _s(200)},
        "required": ["giu", "gop_vao", "ly_do"]}, len(ids))}, "required": ["nhom_trung"]}
    listing = "\n".join(
        f"- {c['id']}: {c['ten']}" + (f" (còn gọi: {', '.join(c['bi_danh'])})" if c.get("bi_danh") else "")
        + f" — {c.get('mo_ta', '')}; xuất hiện ở: {', '.join(x for x in c['xuat_hien'] if x.startswith('scene_'))}"
        for c in registry)
    out, _ = server.chat(
        f"Sổ nhân vật của phim \"{project.name}\" có thể đang đếm trùng một người thành nhiều mục, vì mỗi "
        f"cảnh mô tả ngoại hình hơi khác nhau:\n\n{listing}\n\nHãy tìm các nhóm mục là CÙNG MỘT NGƯỜI (trang "
        f"phục/ngoại hình giống nhau, xuất hiện ở các cảnh nối tiếp, cùng vai trò). Mỗi nhóm: \"giu\" là mục "
        f"giữ lại, \"gop_vao\" là các mục nhập vào nó. Không chắc thì KHÔNG gộp. Không có nhóm nào thì trả mảng rỗng.",
        schema, max_tokens=2048, system=SYSTEM, fail_dir=fail_dir)
    by_id = {c["id"]: c for c in registry}
    removed, merged = set(), []
    for g in out["nhom_trung"]:
        keep = by_id.get(g["giu"])
        if keep is None or g["giu"] in removed:
            continue
        for other_id in g["gop_vao"]:
            other = by_id.get(other_id)
            if other is None or other_id == g["giu"] or other_id in removed:
                continue
            for alias in [other["ten"]] + other.get("bi_danh", []):
                if _is_alias(alias) and alias != keep["ten"] and alias not in keep["bi_danh"]:
                    keep["bi_danh"].append(alias)
            for nid in other["xuat_hien"]:
                if nid not in keep["xuat_hien"]:
                    keep["xuat_hien"].append(nid)
            removed.add(other_id)
            merged.append({"giu": g["giu"], "gop": other_id, "ly_do": g["ly_do"]})
    return [c for c in registry if c["id"] not in removed], merged


# ==========================================
# XUẤT KẾT QUẢ
# ==========================================
def write_report(project, film, chapters, seqs, scenes, registry):
    lines = [f"# {project.name}", "",
             f"- **Thời lượng:** {store.hms(scenes[-1]['end'], False) if scenes else '?'}",
             f"- **Thể loại:** {film.get('the_loai', '')} · **Không khí:** {film.get('khong_khi', '')}",
             f"- **Cảnh / đoạn / chương:** {len(scenes)} / {len(seqs)} / {len(chapters)}",
             f"- **Tạo lúc:** {now_iso()} · {SYNTH_VERSION}", "",
             "## Tóm tắt", "", film.get("tom_tat_ngan", ""), "", film.get("tom_tat_day_du", ""), "",
             "## Cấu trúc", ""]
    lines += [f"- **{p['phan']}** ({p['tu']}–{p['den']}): {p['mo_ta']}" for p in film.get("cau_truc", [])]
    lines += ["", "## Nhân vật", ""]
    for c in registry:
        arc = next((t["dien_bien"] for t in film.get("tuyen_nhan_vat", []) if t["id"] == c["id"]), "")
        alias = f" (còn gọi: {', '.join(c['bi_danh'])})" if c.get("bi_danh") else ""
        lines.append(f"- **{c['ten']}**{alias} — {c.get('mo_ta', '')}. Vai trò: {c.get('vai_tro', '')}. {arc}")
    if film.get("chu_de"):
        lines += ["", "## Chủ đề", ""] + [f"- {t}" for t in film["chu_de"]]
    lines += ["", "## Mục lục theo thời gian", ""]
    seq_by_id = {s["id"]: s for s in seqs}
    scene_by_id = {s["id"]: s for s in scenes}
    for ch in chapters:
        lines += [f"### {ch['id']} · {ch['tieu_de']} ({store.hms(ch['start'], False)}–{store.hms(ch['end'], False)})",
                  "", ch["tom_tat"], ""]
        lines += [f"- `{e['t_hms'][:8]}` {e['mo_ta']}" for e in ch.get("su_kien", [])]
        for sid in ch["children"]:
            sq = seq_by_id[sid]
            lines += ["", f"#### {sq['id']} · {sq['tieu_de']} ({store.hms(sq['start'], False)}–{store.hms(sq['end'], False)})",
                      "", sq["tom_tat"], ""]
            lines += [f"- `{e['t_hms'][:8]}` {e['mo_ta']} ({e['canh']})" for e in sq.get("su_kien", [])]
            lines.append("")
            lines += [f"  - {scid} `{store.hms(scene_by_id[scid]['start'], False)}` "
                      f"{(scene_by_id[scid]['card'].get('mo_ta') or '')[:160]}" for scid in sq["children"]]
        lines.append("")
    tmp = project.path("07_phim", "film_report.md.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, project.path("07_phim", "film_report.md"))


def export_for_tab2(project, scenes):
    """11_xuat/<tên>.json theo đúng khuôn segments/images mà tab 2 (khớp kịch bản) đọc."""
    safe = re.sub(r"[^\w\-. ]", "_", project.name).strip() or "phim"
    data = {
        "source_file": project.data["source"]["file_name"],
        "media_type": "video",
        "project_id": project.id,
        "images": [sc["keyframes"][1]["file"] for sc in scenes if sc.get("keyframes")],
        "segments": [{"index": i, "scene_id": sc["id"], "start_sec": sc["start"], "end_sec": sc["end"],
                      "text": " ".join(p for p in [sc["card"].get("mo_ta", ""), sc["card"].get("y_chinh_loi_thoai", "")] if p)}
                     for i, sc in enumerate(scenes, 1)],
    }
    write_json_atomic(project.path("11_xuat", f"{safe}.json"), data)
    srt = project.path("01_loi_thoai", "dialogue.srt")
    if os.path.isfile(srt):
        shutil.copy2(srt, project.path("11_xuat", f"{safe}.srt"))
    shutil.copy2(project.path("07_phim", "film_report.md"), project.path("11_xuat", f"{safe}_bao_cao.md"))


# ==========================================
# CHẠY
# ==========================================
def main(project, profile, args):
    from film.llm_client import MODEL_NAME, LlamaServer

    scenes = [s for s in store.all_scenes(project) if s]
    plan = read_json(project.path("02_phan_canh", "scene_plan.json"), {}) or {}
    if not scenes or len(scenes) < len(plan.get("scenes", [])):
        raise RuntimeError("Bước mô tả cảnh chưa xong hết — chưa tổng hợp được.")
    scene_by_id = {s["id"]: s for s in scenes}

    seq_groups = group(scenes, SEQ_TARGET_SEC, SEQ_MAX_SCENES)
    seq_meta = [{"id": store.seq_id(i), "index": i, "children": [s["id"] for s in g],
                 "start": g[0]["start"], "end": g[-1]["end"]} for i, g in enumerate(seq_groups, 1)]
    ch_groups = group(seq_meta, CH_TARGET_SEC, CH_MAX_SEQS)
    ch_meta = [{"id": store.chapter_id(i), "index": i, "children": [q["id"] for q in g],
                "start": g[0]["start"], "end": g[-1]["end"]} for i, g in enumerate(ch_groups, 1)]
    seq_by_id = {q["id"]: q for q in seq_meta}
    seq_children = {q["id"]: q["children"] for q in seq_meta}
    total_units = len(seq_meta) + len(ch_meta) + 1

    state_path = project.path("07_phim", "_state.json")
    state = read_json(state_path, {}) or {}
    if state.get("version") != SYNTH_VERSION:
        state = {"version": SYNTH_VERSION, "done_chapters": [], "story": "", "registry": []}

    def current(level, nid):
        """Đã có bản theo ĐÚNG phiên bản tổng hợp hiện tại chưa (bản cũ hơn vẫn giữ nguyên)."""
        node = store.read_node(project, level, nid)
        return bool(node) and node.get("complete") and node.get("provenance", {}).get("version") == SYNTH_VERSION

    def write(level, nid, data):
        store.write_node(project, level, nid, data, new_revision=store.has_node(project, level, nid))

    def done_units():
        return (sum(current("seq", q["id"]) for q in seq_meta)
                + sum(current("chapter", c["id"]) for c in ch_meta))

    stage_io.progress(2, f"{len(scenes)} cảnh -> {len(seq_meta)} đoạn -> {len(ch_meta)} chương. Nạp {MODEL_NAME}...")
    server = LlamaServer(profile, project.log_path("llama-server.log"), project.path("tmp"))
    fail_dir = project.path("10_nhat_ky")
    server.start()
    try:
        stage_io.model_loaded(5, "Đã nạp model tổng hợp.")
        for ch in ch_meta:
            if ch["id"] in state["done_chapters"] and current("chapter", ch["id"]):
                continue
            seq_nodes = []
            for qid in ch["children"]:
                q = seq_by_id[qid]
                if current("seq", qid):
                    seq_nodes.append(store.read_node(project, "seq", qid))
                    continue
                t0 = time.time()
                earlier = "\n".join(f"- {s['id']}: {s['tom_tat']}" for s in seq_nodes)
                prompt = (
                    f"Phim: \"{project.name}\". Tóm tắt ĐOẠN {qid} ({store.hms(q['start'], False)}–"
                    f"{store.hms(q['end'], False)}), gồm {len(q['children'])} cảnh liên tiếp.\n\n"
                    f"Câu chuyện từ đầu phim tới trước chương này:\n{state['story'] or '(đây là đầu phim)'}\n\n"
                    + (f"Các đoạn trước trong cùng chương:\n{earlier}\n\n" if earlier else "")
                    + f"Sổ nhân vật hiện có:\n{render_registry(state['registry'])}\n\n"
                    f"Tư liệu các cảnh trong đoạn:\n\n"
                    + "\n\n".join(render_scene(scene_by_id[s]) for s in q["children"])
                    + "\n\nYêu cầu: \"tieu_de\" là một tiêu đề ngắn gợi tả nội dung đoạn (không phải mã đoạn); "
                      "\"tom_tat\" 5–10 câu kể lại diễn biến của đoạn, nối với câu chuyện trước đó; "
                      "\"su_kien\" là các sự kiện quan trọng theo thứ tự thời gian (mốc hh:mm:ss, cảnh chứa sự kiện); "
                      "\"nhan_vat\" là những người xuất hiện trong đoạn (dùng đúng tên như trong sổ nếu là người đã biết); "
                      "\"buoc_ngoat\" để chuỗi rỗng nếu đoạn không có bước ngoặt.")
                out, timing = server.chat(prompt, seq_schema(q["children"]), max_tokens=3072, system=SYSTEM, fail_dir=fail_dir)
                events = []
                for i, e in enumerate(out["su_kien"], 1):
                    sc = scene_by_id[e["canh"]]
                    t = parse_hms(e["moc"], sc["start"], sc["end"], sc["start"])
                    events.append({"id": store.event_id(qid, i), "t": t, "t_hms": store.hms(t),
                                   "canh": e["canh"], "mo_ta": e["mo_ta"]})
                node = {**q, "complete": True, "level": "seq", "tieu_de": out["tieu_de"], "tom_tat": out["tom_tat"],
                        "su_kien": events, "nhan_vat": out["nhan_vat"], "dia_diem": out["dia_diem"],
                        "buoc_ngoat": out["buoc_ngoat"], "parent": ch["id"],
                        "start_hms": store.hms(q["start"]), "end_hms": store.hms(q["end"]),
                        "provenance": {"model": MODEL_NAME, "version": SYNTH_VERSION, "created_at": now_iso(),
                                       "seconds": round(time.time() - t0, 1),
                                       "prompt_tokens": timing.get("prompt_n"), "output_tokens": timing.get("predicted_n")}}
                write("seq", qid, node)
                seq_nodes.append(store.read_node(project, "seq", qid))
                stage_io.progress(5 + 90 * done_units() / total_units,
                                  f"{qid} ({store.hms(q['start'], False)}): {out['tieu_de']} — {time.time() - t0:.0f}s")
                stage_io.check_pause(project, "synthesis")

            t0 = time.time()
            char_ids = [c["id"] for c in state["registry"]]
            prompt = (
                f"Phim: \"{project.name}\". Tóm tắt CHƯƠNG {ch['id']} ({store.hms(ch['start'], False)}–"
                f"{store.hms(ch['end'], False)}), gồm {len(seq_nodes)} đoạn.\n\n"
                f"Câu chuyện từ đầu phim tới trước chương này:\n{state['story'] or '(đây là đầu phim)'}\n\n"
                f"Sổ nhân vật hiện có (id: tên — nhận dạng):\n{render_registry(state['registry'])}\n\n"
                "Các đoạn trong chương:\n\n"
                + "\n\n".join(
                    f"### {s['id']} [{store.hms(s['start'], False)}–{store.hms(s['end'], False)}] {s['tieu_de']}\n"
                    f"{s['tom_tat']}\nSự kiện: " + "; ".join(f"[{e['t_hms'][:8]}] {e['mo_ta']}" for e in s["su_kien"])
                    + "\nNhân vật: " + "; ".join(f"{n['ten']} ({n['vai_tro_trong_doan']})" for n in s["nhan_vat"])
                    for s in seq_nodes)
                + "\n\nYêu cầu: \"tieu_de\" là tiêu đề ngắn gợi tả nội dung chương (không ghi \"Chương\" hay mã chương); "
                  "\"tom_tat\" 6–12 câu; \"diem_chinh\" là các mốc quan trọng nhất của chương; "
                  "\"cau_chuyen_toi_gio\" tóm tắt TOÀN BỘ câu chuyện từ đầu phim tới hết chương này, tối đa 250 từ; "
                  "\"nhan_vat\": mọi nhân vật xuất hiện trong chương — người đã có trong sổ thì dùng đúng id đó "
                  "(và cập nhật tên nếu giờ mới biết tên thật), người mới thì id là \"moi\". "
                  "Mô tả hình ảnh ở từng cảnh thường KHÁC NHAU một chút cho cùng một người (vd: \"áo vàng\", "
                  "\"áo hoodie màu vàng\", \"tóc xoăn mặc hoodie vàng\" ở các cảnh liền nhau): gộp thành MỘT nhân "
                  "vật, các cách gọi kia đưa vào \"bi_danh\". Chỉ tạo nhân vật riêng khi chắc chắn là người khác.")
            out, timing = server.chat(prompt, chapter_schema(ch["children"], char_ids), max_tokens=4096, system=SYSTEM, fail_dir=fail_dir)
            events = []
            for i, e in enumerate(out["diem_chinh"], 1):
                sq = seq_by_id[e["doan"]]
                t = parse_hms(e["moc"], sq["start"], sq["end"], sq["start"])
                events.append({"id": store.event_id(ch["id"], i), "t": t, "t_hms": store.hms(t), "doan": e["doan"],
                               "canh": next((s for s in sq["children"] if scene_by_id[s]["start"] <= t < scene_by_id[s]["end"]),
                                            sq["children"][0]), "mo_ta": e["mo_ta"]})
            state["registry"] = merge_characters(state["registry"], out["nhan_vat"], seq_children)
            write("chapter", ch["id"], {
                **ch, "complete": True, "level": "chapter", "tieu_de": out["tieu_de"], "tom_tat": out["tom_tat"],
                "su_kien": events, "buoc_ngoat": out["buoc_ngoat"], "cau_chuyen_toi_gio": out["cau_chuyen_toi_gio"],
                "nhan_vat": [u["id"] if u["id"] != "moi" else u["ten"] for u in out["nhan_vat"]],
                "start_hms": store.hms(ch["start"]), "end_hms": store.hms(ch["end"]),
                "provenance": {"model": MODEL_NAME, "version": SYNTH_VERSION, "created_at": now_iso(),
                               "seconds": round(time.time() - t0, 1),
                               "prompt_tokens": timing.get("prompt_n"), "output_tokens": timing.get("predicted_n")}})
            state["story"] = out["cau_chuyen_toi_gio"]
            state["done_chapters"].append(ch["id"])
            write_json_atomic(state_path, state)
            write_json_atomic(project.path("07_phim", "characters.json"),
                              {"version": SYNTH_VERSION, "characters": state["registry"]})
            stage_io.progress(5 + 90 * done_units() / total_units,
                              f"{ch['id']}: {out['tieu_de']} — {time.time() - t0:.0f}s")
            stage_io.check_pause(project, "synthesis")

        chapters = [store.read_node(project, "chapter", c["id"]) for c in ch_meta]
        seqs = [store.read_node(project, "seq", q["id"]) for q in seq_meta]
        if len(state["registry"]) > 1 and not state.get("deduped"):
            stage_io.progress(95, f"Rà {len(state['registry'])} nhân vật để gộp người bị đếm trùng...")
            state["registry"], merged = dedupe_characters(server, project, state["registry"], fail_dir)
            state["deduped"] = True
            state["merged"] = merged
            write_json_atomic(state_path, state)
            write_json_atomic(project.path("07_phim", "characters.json"),
                              {"version": SYNTH_VERSION, "characters": state["registry"], "merged": merged})
        char_ids = [c["id"] for c in state["registry"]]
        stage_io.progress(96, "Tổng hợp cả phim...")
        prompt = (
            f"Phim: \"{project.name}\", dài {store.hms(scenes[-1]['end'], False)}.\n\n"
            f"Sổ nhân vật:\n{render_registry(state['registry'])}\n\nCác chương:\n\n"
            + "\n\n".join(f"### {c['id']} [{store.hms(c['start'], False)}–{store.hms(c['end'], False)}] {c['tieu_de']}\n"
                          f"{c['tom_tat']}\nBước ngoặt: {c['buoc_ngoat'] or '(không)'}" for c in chapters)
            + "\n\nYêu cầu: \"tom_tat_ngan\" 3–5 câu; \"tom_tat_day_du\" 3–6 đoạn văn kể lại toàn bộ phim; "
              "\"cau_truc\" chia phim thành các phần lớn (mở đầu, phát triển, cao trào, kết…) kèm mốc hh:mm:ss; "
              "\"tuyen_nhan_vat\" diễn biến của từng nhân vật quan trọng qua cả phim.")
        film, timing = server.chat(prompt, film_schema(char_ids), max_tokens=6144, system=SYSTEM, fail_dir=fail_dir)
    finally:
        server.stop()

    film.update({"version": SYNTH_VERSION, "model": MODEL_NAME, "created_at": now_iso(),
                 "chapters": [{"id": c["id"], "tieu_de": c["tieu_de"], "start": c["start"], "end": c["end"],
                               "start_hms": c["start_hms"], "end_hms": c["end_hms"]} for c in chapters]})
    write_json_atomic(project.path("07_phim", "film.json"), film)
    write_report(project, film, chapters, seqs, scenes, state["registry"])
    export_for_tab2(project, scenes)
    stage_io.progress(99, "Dựng chỉ mục tra cứu...")
    film_index.rebuild(project)
    stage_io.progress(100, f"Xong: {len(seqs)} đoạn, {len(chapters)} chương, {len(state['registry'])} nhân vật.")
    return {"seqs": len(seqs), "chapters": len(chapters), "characters": len(state["registry"])}


if __name__ == "__main__":
    stage_io.run_stage("synthesis", main)
