"""Phụ đề có sẵn: tìm, đọc, chuẩn hoá. Có phụ đề thì bỏ qua hẳn bước bóc băng Whisper.

Thứ tự tìm: file .srt cùng tên phim (kể cả tên.vi.srt, tên.en.srt…) -> phụ đề dạng chữ nhúng
trong file phim (ffmpeg tách ra). Phụ đề dạng ảnh (PGS/VobSub) không đọc được bằng chữ nên
bỏ qua - khi đó vẫn phải bóc băng.
"""

import glob
import json
import os
import re
import subprocess
import unicodedata

TEXT_SUB_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}
_TIME = r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
_CUE_RE = re.compile(_TIME + r"\s*-->\s*" + _TIME)
_TAG_RE = re.compile(r"<[^>]+>|\{\\[^}]*\}")


def find_sidecar_srts(video_path):
    """Các file .srt đi kèm phim: <tên>.srt và <tên>.<mã ngôn ngữ>.srt."""
    base = os.path.splitext(video_path)[0]
    found = []
    exact = base + ".srt"
    if os.path.isfile(exact):
        found.append(exact)
    for p in sorted(glob.glob(glob.escape(base) + ".*.srt")):
        if p not in found:
            found.append(p)
    return found


def lang_from_filename(path):
    """"phim.vi.srt" -> "vi"; không có mã thì None."""
    m = re.search(r"\.([a-z]{2,3}(?:-[A-Za-z]{2,4})?)\.srt$", os.path.basename(path), re.I)
    return m.group(1).lower() if m else None


def list_embedded(video_path):
    """Các luồng phụ đề dạng chữ nhúng trong phim: [{index, codec, lang, title}]."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s", "-show_entries",
         "stream=index,codec_name:stream_tags=language,title", "-of", "json", video_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        streams = json.loads(out.stdout or "{}").get("streams", [])
    except ValueError:
        return []
    subs = []
    for n, s in enumerate(streams):
        if s.get("codec_name") in TEXT_SUB_CODECS:
            tags = s.get("tags", {})
            subs.append({"order": n, "index": s["index"], "codec": s.get("codec_name"),
                         "lang": tags.get("language"), "title": tags.get("title")})
    return subs


def extract_embedded(video_path, sub_order, out_path):
    """Tách luồng phụ đề thứ sub_order (thứ tự trong các luồng phụ đề) ra .srt."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", video_path, "-map", f"0:s:{sub_order}",
         "-c:s", "srt", out_path],
        check=True, capture_output=True,
    )
    return out_path


def _decode(raw):
    """Đoán mã hoá: BOM -> UTF-16 không BOM -> UTF-8 -> cp1258 (tiếng Việt Windows cũ) -> cp1252."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace"), "utf-8-sig"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace"), "utf-16"
    if raw[:200].count(b"\x00") > 20:   # UTF-16 không có BOM: cứ mỗi ký tự ASCII lại một byte 0
        le = raw[1::2].count(b"\x00") > raw[0::2].count(b"\x00")
        return raw.decode("utf-16-le" if le else "utf-16-be", errors="replace"), "utf-16"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    for enc in ("cp1258", "cp1252"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), "latin-1"


def _secs(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def read_srt(path, duration=None):
    """Đọc .srt -> (danh sách {start, end, text}, mã hoá đã dùng).

    Bỏ thẻ định dạng (<i>, {\\an8}…), gộp nhiều dòng của một câu, bỏ dòng rỗng, sắp theo
    thời gian, và cắt các mốc vượt quá độ dài phim (phụ đề của bản dựng khác thường bị lệch)."""
    with open(path, "rb") as f:
        text, enc = _decode(f.read())
    # cp1258 (và một số file UTF-8 gõ bằng bộ gõ cũ) lưu dấu thanh TÁCH RỜI khỏi chữ: "ả" là
    # "a" + dấu hỏi riêng. Không gộp lại thành NFC thì tìm "cảnh sát" sẽ không khớp.
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for block in re.split(r"\n\s*\n", text):
        rows = [r for r in block.split("\n") if r.strip()]
        for i, row in enumerate(rows):
            m = _CUE_RE.search(row)
            if not m:
                continue
            start, end = _secs(*m.groups()[:4]), _secs(*m.groups()[4:])
            body = " ".join(_TAG_RE.sub("", r).strip() for r in rows[i + 1:]).strip()
            body = re.sub(r"\s+", " ", body)
            if body and end > start:
                lines.append({"start": round(start, 3), "end": round(end, 3), "text": body})
            break
    lines.sort(key=lambda x: x["start"])
    if duration:
        lines = [dict(ln, end=min(ln["end"], duration)) for ln in lines if ln["start"] < duration]
    return lines, enc


def _srt_time(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, ln in enumerate(lines, 1):
            f.write(f"{i}\n{_srt_time(ln['start'])} --> {_srt_time(ln['end'])}\n{ln['text']}\n\n")
