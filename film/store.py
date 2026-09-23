"""Lưu trữ theo tầng bên trong thư mục dự án.

    TẦNG 1  03_canh/scene_0001.json     cảnh nhỏ (20–90 giây) - đầy đủ, chi tiết, KHÔNG BAO GIỜ XOÁ
    TẦNG 2  05_doan/seq_001.json        đoạn (vài cảnh)
    TẦNG 3  06_chuong/ch_01.json        chương (vài đoạn)
    TẦNG 4  07_phim/film.json           cả phim

Phân tích lại một cảnh (đổi model, đổi prompt) ghi thành scene_0001.r2.json, r3… - bản cũ vẫn
còn nguyên. Hàm đọc mặc định lấy bản mới nhất. File JSON là nguồn gốc duy nhất; chỉ mục SQLite
(index.py) chỉ dẫn xuất từ đây.
"""

import glob
import os
import re

from film.project import read_json, write_json_atomic

LEVEL_DIRS = {"scene": "03_canh", "seq": "05_doan", "chapter": "06_chuong"}
_REV_RE = re.compile(r"^(?P<id>[a-z]+_\d+)(?:\.r(?P<rev>\d+))?\.json$")


def scene_id(i):
    return f"scene_{i:04d}"


def seq_id(i):
    return f"seq_{i:03d}"


def chapter_id(i):
    return f"ch_{i:02d}"


def event_id(node_id, i):
    return f"{node_id}.e{i:02d}"


def hms(sec, ms=True):
    """Giây -> "hh:mm:ss.mmm" (hoặc "hh:mm:ss")."""
    sec = max(0.0, float(sec or 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}" if ms else f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


def _revisions(project, level, node_id):
    """[(rev, path)] tăng dần. Bản gốc là rev 1."""
    out = []
    for path in glob.glob(project.path(LEVEL_DIRS[level], f"{node_id}*.json")):
        m = _REV_RE.match(os.path.basename(path))
        if m and m.group("id") == node_id:
            out.append((int(m.group("rev") or 1), path))
    return sorted(out)


def read_node(project, level, node_id, rev=None):
    revs = _revisions(project, level, node_id)
    if not revs:
        return None
    if rev is None:
        return read_json(revs[-1][1])
    for r, path in revs:
        if r == rev:
            return read_json(path)
    return None


def has_node(project, level, node_id):
    """Có ít nhất một bản hợp lệ (đọc được, đủ trường cốt lõi) - dùng làm checkpoint."""
    data = read_node(project, level, node_id)
    return isinstance(data, dict) and data.get("id") == node_id and data.get("complete") is True


def write_node(project, level, node_id, data, new_revision=False):
    """Ghi một nút. new_revision=False: chỉ ghi khi chưa có bản nào (chạy tiếp từ checkpoint
    không bao giờ đè lên kết quả cũ). new_revision=True: ghi thêm bản .rN mới."""
    revs = _revisions(project, level, node_id)
    folder = project.path(LEVEL_DIRS[level])
    if revs and not new_revision:
        raise FileExistsError(f"{node_id} đã có ({os.path.basename(revs[-1][1])}); dùng new_revision=True.")
    rev = (revs[-1][0] + 1) if revs else 1
    name = f"{node_id}.json" if rev == 1 else f"{node_id}.r{rev}.json"
    data = {**data, "id": node_id, "revision": rev}
    write_json_atomic(os.path.join(folder, name), data)
    return rev


def list_nodes(project, level):
    """Id của mọi nút ở một tầng, theo thứ tự."""
    ids = set()
    for path in glob.glob(project.path(LEVEL_DIRS[level], "*.json")):
        m = _REV_RE.match(os.path.basename(path))
        if m:
            ids.add(m.group("id"))
    return sorted(ids)


def all_scenes(project):
    return [read_node(project, "scene", sid) for sid in list_nodes(project, "scene")]
