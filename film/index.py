"""Chỉ mục tra cứu 09_chi_muc/index.sqlite - DẪN XUẤT từ các file JSON, xoá đi dựng lại được.

Bảng:
  nodes        mọi cảnh / đoạn / chương (+ phim): cha–con, mốc, tiêu đề, tóm tắt, đường dẫn JSON
  events       từng sự kiện có id riêng (scene_0001.e03, seq_001.e02, ch_01.e01…)
  dialogue     từng câu thoại, gắn với cảnh chứa nó
  characters   sổ nhân vật; appearances: nhân vật xuất hiện ở nút nào
  fts          tìm kiếm toàn văn trên mô tả / sự kiện / lời thoại / tóm tắt

Tìm tiếng Việt không dấu: FTS5 "unicode61 remove_diacritics 2" bỏ dấu thanh và dấu mũ, nhưng
"đ" là một CHỮ riêng (U+0111) chứ không phải "d" có dấu - nên cả văn bản lẫn câu tìm đều được
gập đ->d trước. Nhờ vậy gõ "canh sat", "đường" hay "duong" đều ra.
"""

import os
import pathlib
import re
import sqlite3
import unicodedata

from film import store
from film.project import read_json

SCHEMA = """
CREATE TABLE nodes (id TEXT PRIMARY KEY, level TEXT, parent TEXT, idx INTEGER, start REAL, end REAL,
                    title TEXT, summary TEXT, path TEXT, revision INTEGER);
CREATE TABLE events (id TEXT PRIMARY KEY, node_id TEXT, level TEXT, t REAL, text TEXT, scene_id TEXT);
CREATE TABLE dialogue (scene_id TEXT, start REAL, end REAL, text TEXT, lang TEXT);
CREATE TABLE characters (id TEXT PRIMARY KEY, name TEXT, aliases TEXT, description TEXT, role TEXT);
CREATE TABLE appearances (character_id TEXT, node_id TEXT);
CREATE VIRTUAL TABLE fts USING fts5(kind UNINDEXED, ref_id UNINDEXED, node_id UNINDEXED, t UNINDEXED,
                                    body UNINDEXED, folded, tokenize = 'unicode61 remove_diacritics 2');
CREATE INDEX idx_nodes_parent ON nodes(parent);
CREATE INDEX idx_events_node ON events(node_id);
CREATE INDEX idx_dialogue_scene ON dialogue(scene_id);
"""


def fold(text):
    return (text or "").replace("đ", "d").replace("Đ", "D")


def index_path(project):
    return project.path("09_chi_muc", "index.sqlite")


def _scene_text(sc):
    c = sc.get("card", {})
    parts = [c.get("mo_ta", ""), c.get("boi_canh", ""), c.get("chu_tren_man_hinh", ""),
             c.get("am_thanh_khong_khi", ""), c.get("y_chinh_loi_thoai", "")]
    parts += [f"{n.get('goi_la', '')} {n.get('ngoai_hinh', '')} {n.get('hanh_dong', '')}" for n in c.get("nhan_vat", [])]
    return " ".join(p for p in parts if p)


def rebuild(project):
    """Dựng lại toàn bộ chỉ mục vào file tạm rồi thay thế - đang đọc dở cũng không thấy bản hỏng."""
    final = index_path(project)
    tmp = final + ".building"
    if os.path.exists(tmp):
        os.remove(tmp)
    db = sqlite3.connect(tmp)
    db.executescript(SCHEMA)

    def add_fts(kind, ref_id, node_id, t, body):
        if body:
            db.execute("INSERT INTO fts VALUES (?,?,?,?,?,?)", (kind, ref_id, node_id, t, body, fold(body)))

    seq_of_scene, ch_of_seq = {}, {}
    for level in ("chapter", "seq"):
        for nid in store.list_nodes(project, level):
            node = store.read_node(project, level, nid)
            for child in node.get("children", []):
                (ch_of_seq if level == "chapter" else seq_of_scene)[child] = nid

    for sid in store.list_nodes(project, "scene"):
        sc = store.read_node(project, "scene", sid)
        if not sc:
            continue
        db.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (sid, "scene", seq_of_scene.get(sid), sc.get("index"), sc["start"], sc["end"],
                    None, sc["card"].get("mo_ta"), _node_rel(project, "scene", sid), sc.get("revision")))
        add_fts("scene", sid, sid, sc["start"], _scene_text(sc))
        for ev in sc.get("events", []):
            db.execute("INSERT INTO events VALUES (?,?,?,?,?,?)", (ev["id"], sid, "scene", ev["t"], ev["mo_ta"], sid))
            add_fts("event", ev["id"], sid, ev["t"], ev["mo_ta"])
        for d in sc.get("dialogue", []):
            db.execute("INSERT INTO dialogue VALUES (?,?,?,?,?)", (sid, d["start"], d["end"], d["text"], d.get("lang")))
            add_fts("dialogue", sid, sid, d["start"], d["text"])

    for level, parent_map in (("seq", ch_of_seq), ("chapter", {})):
        for nid in store.list_nodes(project, level):
            node = store.read_node(project, level, nid)
            db.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (nid, level, parent_map.get(nid), node.get("index"), node["start"], node["end"],
                        node.get("tieu_de"), node.get("tom_tat"), _node_rel(project, level, nid), node.get("revision")))
            add_fts(level, nid, nid, node["start"], f"{node.get('tieu_de', '')} {node.get('tom_tat', '')}")
            for ev in node.get("su_kien", []):
                db.execute("INSERT INTO events VALUES (?,?,?,?,?,?)",
                           (ev["id"], nid, level, ev.get("t"), ev["mo_ta"], ev.get("canh")))
                add_fts("event", ev["id"], ev.get("canh") or nid, ev.get("t"), ev["mo_ta"])

    chars = read_json(project.path("07_phim", "characters.json"), {}) or {}
    for ch in chars.get("characters", []):
        db.execute("INSERT INTO characters VALUES (?,?,?,?,?)",
                   (ch["id"], ch.get("ten"), ", ".join(ch.get("bi_danh", [])), ch.get("mo_ta"), ch.get("vai_tro")))
        for nid in ch.get("xuat_hien", []):
            db.execute("INSERT INTO appearances VALUES (?,?)", (ch["id"], nid))
        add_fts("character", ch["id"], None, None,
                f"{ch.get('ten', '')} {' '.join(ch.get('bi_danh', []))} {ch.get('mo_ta', '')} {ch.get('vai_tro', '')}")
    db.commit()
    db.close()
    os.replace(tmp, final)
    return final


def _node_rel(project, level, nid):
    revs = store._revisions(project, level, nid)
    return project.rel(revs[-1][1]) if revs else None


def _connect(project):
    path = index_path(project)
    if not os.path.isfile(path):
        rebuild(project)
    # as_uri() mã hoá đúng dấu cách / ký tự tiếng Việt trong đường dẫn thư mục dự án.
    return sqlite3.connect(pathlib.Path(path).as_uri() + "?mode=ro", uri=True)


def _ascii(word):
    return unicodedata.normalize("NFD", fold(word)).encode("ascii", "ignore").decode().lower()


# Chỉ bỏ ở chế độ khớp-một-phần (OR): ở đó "không"/"có"… sẽ kéo về gần như mọi cảnh.
_STOP = {_ascii(w) for w in (
    "không có là và của một những các được trong với cho này đó thì đã đang sẽ rồi mà nào gì ai "
    "the a an of to in and is are was were it on at by for with").split()}


def _fts_query(q, mode):
    tokens = [t for t in re.findall(r"\w+", fold(q).lower()) if len(t) > 1 or t.isdigit()]
    if mode == "any":
        tokens = [t for t in tokens if _ascii(t) not in _STOP]
    if not tokens:
        return None
    return (" AND " if mode == "all" else " OR ").join(f'"{t}"' for t in tokens)


def search(project, q, limit=30):
    """Tìm trong mô tả, sự kiện, lời thoại, tóm tắt, nhân vật. Ưu tiên khớp đủ mọi từ; không
    có kết quả thì nới ra khớp một phần (OR)."""
    db = _connect(project)
    try:
        for mode in ("all", "any"):
            expr = _fts_query(q, mode)
            if not expr:
                return []
            rows = db.execute(
                "SELECT kind, ref_id, node_id, t, body, bm25(fts) FROM fts WHERE fts MATCH ? "
                "ORDER BY bm25(fts) LIMIT ?", (f"folded : ({expr})", limit)).fetchall()
            if rows:
                return [{"kind": k, "ref_id": r, "node_id": n, "t": t, "t_hms": store.hms(t) if t is not None else None,
                         "text": body[:300], "score": round(-s, 3), "match": mode} for k, r, n, t, body, s in rows]
        return []
    finally:
        db.close()


def tree(project):
    """Cây chương -> đoạn -> cảnh (chỉ tiêu đề + mốc, đủ để vẽ trục thời gian)."""
    db = _connect(project)
    try:
        rows = db.execute("SELECT id, level, parent, start, end, title, summary FROM nodes ORDER BY start").fetchall()
    finally:
        db.close()
    nodes = {r[0]: {"id": r[0], "level": r[1], "parent": r[2], "start": r[3], "end": r[4],
                    "start_hms": store.hms(r[3], False), "end_hms": store.hms(r[4], False),
                    "title": r[5], "summary": (r[6] or "")[:240], "children": []} for r in rows}
    roots = []
    for n in nodes.values():
        parent = nodes.get(n["parent"]) if n["parent"] else None
        (parent["children"] if parent else roots).append(n)
    return roots
