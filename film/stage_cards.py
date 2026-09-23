"""Bước [4] Mô tả cảnh: mỗi cảnh trong kế hoạch -> một file 03_canh/scene_XXXX.json đầy đủ.

Đây là bước lâu nhất (hàng giờ cho một phim dài) nên:
  - Checkpoint theo TỪNG CẢNH: cảnh nào đã có file hợp lệ thì bỏ qua. Tạm dừng / sập máy /
    tắt app rồi chạy lại đều đi tiếp đúng chỗ.
  - File cảnh không bao giờ bị ghi đè hay xoá (store.write_node).
  - Kiểm tra lệnh tạm dừng sau mỗi cảnh (vừa lưu xong mới dừng).

Model nhận: khung hình lấy mẫu đều trong cảnh + LỜI THOẠI của đúng khoảng thời gian đó + tóm
tắt 2 cảnh liền trước (để giữ mạch truyện), và trả về một thẻ cảnh JSON bằng tiếng Việt.

NHIỀU LƯỢT NHÌN: số khung hình một lượt chịu được phụ thuộc VRAM đang trống (ứng dụng khác
như Blender, trình duyệt… có thể chiếm vài GB). Thay vì hạ mật độ khung hình - cảnh 60 giây mà
chỉ nhìn 8 khung là bỏ sót hành động - cảnh được chia thành vài đoạn nhỏ, mỗi đoạn một lượt
nhìn ghi lại quan sát kèm mốc giây, rồi một lượt CHỈ CÓ CHỮ gộp các quan sát thành thẻ cảnh.

Mốc thời gian: model ghi theo giây tính từ đầu ĐOẠN nó đang xem (việc dễ nhất cho model); code
tự cộng ra mốc trong cảnh, rồi ra giờ phim tuyệt đối.
"""

import json
import math
import os
import re
import time

from film import stage_io, store
from film.project import now_iso, read_json

PROMPT_VERSION = "card-v3"
KEYFRAME_SIDE = 640
MIN_FRAMES_PER_PASS = 6
MAX_PASSES = 6
CARD_FIELDS = ["mo_ta", "boi_canh", "nhan_vat", "su_kien", "chu_tren_man_hinh",
               "am_thanh_khong_khi", "y_chinh_loi_thoai"]
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_OBS_TIME = re.compile(r"\[\s*giây\s*([\d.]+)\s*\]", re.IGNORECASE)

LANGUAGE_RULE = ("Toàn bộ câu trả lời viết bằng TIẾNG VIỆT. Không chen chữ Trung Quốc, Nhật, Hàn hay "
                 "tiếng Anh vào câu (trừ tên riêng và lời thoại trích nguyên văn).")


def scene_dialogue(lines, start, end):
    out = []
    for ln in lines:
        if ln["end"] > start and ln["start"] < end:
            out.append({"start": ln["start"], "end": ln["end"],
                        "rel_start": round(max(0.0, ln["start"] - start), 2),
                        "rel_end": round(min(end, ln["end"]) - start, 2),
                        "text": ln["text"], "lang": ln.get("lang")})
    return out


def _dialogue_block(dialogue, offset=0.0, title="Lời thoại trong cảnh"):
    if not dialogue:
        return "\nKhông có lời thoại."
    rows = [f"\n{title} (mốc tính bằng giây, giữ nguyên ngôn ngữ gốc):"]
    rows += [f"[{d['rel_start'] - offset:.1f}–{d['rel_end'] - offset:.1f}] {d['text']}" for d in dialogue]
    return "\n".join(rows)


def _context_block(previous):
    if not previous:
        return ""
    rows = ["\nDiễn biến ngay trước cảnh này (để giữ mạch truyện, KHÔNG mô tả lại):"]
    rows += [f"- {p['id']} ({store.hms(p['start'], False)}): {p['summary']}" for p in previous]
    return "\n".join(rows)


def _schema_block(dur, has_dialogue):
    dialogue_rule = ("Cảnh CÓ lời thoại: \"y_chinh_loi_thoai\" BẮT BUỘC tóm tắt BẰNG TIẾNG VIỆT (dịch ý, "
                     "không chép nguyên văn) ai nói gì, với ai, nhằm mục đích gì."
                     if has_dialogue else "Cảnh không có lời thoại: để \"y_chinh_loi_thoai\" là chuỗi rỗng.")
    return f"""
Hãy trả về DUY NHẤT một đối tượng JSON hợp lệ (không kèm chữ nào khác, không dùng ```):
{{
  "mo_ta": "mô tả chi tiết chuyện gì đang diễn ra trong cảnh, 4–8 câu",
  "boi_canh": "địa điểm, trong nhà/ngoài trời, thời gian trong ngày, thời tiết",
  "nhan_vat": [{{"goi_la": "tên nếu được gọi trong lời thoại, nếu không thì mô tả ngắn như 'người đàn ông áo khoác đen'", "ngoai_hinh": "...", "hanh_dong": "..."}}],
  "su_kien": [{{"giay": <số giây tính từ đầu cảnh, từ 0 đến {dur:.0f}>, "mo_ta": "một hành động hoặc thay đổi cụ thể"}}],
  "chu_tren_man_hinh": "chữ đọc được trên hình (biển hiệu, màn hình, chữ cứng), không có thì để chuỗi rỗng",
  "am_thanh_khong_khi": "không khí, cảm xúc, nhịp độ của cảnh",
  "y_chinh_loi_thoai": "..."
}}
Quy định:
- {LANGUAGE_RULE}
- {dialogue_rule}
- "su_kien": 3–8 sự kiện xếp theo thời gian, mỗi sự kiện là một việc KHÁC NHAU; không lặp lại cùng một hành động.
- Chỉ ghi những gì thấy trong hình hoặc nghe trong lời thoại, không bịa thêm."""


def single_pass_prompt(title, scene, n_frames, dialogue, previous):
    dur = scene["end"] - scene["start"]
    return (f"Bạn đang xem MỘT CẢNH trong phim \"{title}\", từ {store.hms(scene['start'], False)} đến "
            f"{store.hms(scene['end'], False)} (dài {dur:.0f} giây), qua {n_frames} khung hình lấy mẫu đều."
            + _context_block(previous) + _dialogue_block(dialogue) + _schema_block(dur, bool(dialogue)))


def observe_prompt(title, k, n, a, b, n_frames, dialogue):
    seg = [d for d in dialogue if d["rel_end"] > a and d["rel_start"] < b]
    return (f"Đây là đoạn {k}/{n} của một cảnh trong phim \"{title}\", dài {b - a:.0f} giây, xem qua "
            f"{n_frames} khung hình lấy mẫu đều."
            + _dialogue_block(seg, offset=a, title="Lời thoại trong đoạn này")
            + f"""

Hãy liệt kê theo thứ tự thời gian những gì NHÌN THẤY trong đoạn: ai xuất hiện (ngoại hình, trang phục),
họ làm gì, bối cảnh, chữ đọc được trên hình. Mỗi dòng bắt đầu bằng mốc "[giây X]", X tính từ đầu ĐOẠN
này (từ 0 đến {b - a:.0f}). Không lặp lại cùng một hành động. {LANGUAGE_RULE}""")


def merge_prompt(title, scene, observations, dialogue, previous):
    dur = scene["end"] - scene["start"]
    return (f"Dưới đây là các ghi chép quan sát MỘT CẢNH trong phim \"{title}\", từ "
            f"{store.hms(scene['start'], False)} đến {store.hms(scene['end'], False)} (dài {dur:.0f} giây). "
            f"Cảnh được xem thành {len(observations)} đoạn liên tiếp; mốc [giây X] đã được đổi ra giây "
            f"tính từ đầu CẢNH."
            + _context_block(previous)
            + "\n\nGhi chép quan sát:\n" + "\n\n".join(observations)
            + _dialogue_block(dialogue)
            + "\n\nHãy gộp các ghi chép thành MỘT thẻ cảnh thống nhất (nối liền diễn biến giữa các đoạn, "
              "bỏ phần trùng lặp)." + _schema_block(dur, bool(dialogue)))


def fix_language_prompt(raw):
    return ("Đoạn JSON dưới đây có chỗ chen chữ nước ngoài (Trung/Nhật/Hàn/Anh) vào câu tiếng Việt. "
            "Hãy viết lại ĐÚNG khuôn JSON đó, giữ nguyên nội dung và các con số, nhưng mọi câu đều bằng "
            "tiếng Việt (tên riêng và lời thoại trích nguyên văn thì giữ). Chỉ trả về JSON.\n\n" + raw)


def _shift_times(text, offset):
    return _OBS_TIME.sub(lambda m: f"[giây {float(m.group(1)) + offset:.0f}]", text)


def parse_card(raw):
    """Tách JSON từ câu trả lời của model. Không tách được thì trả về None (giữ nguyên bản thô)."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    lo, hi = text.find("{"), text.rfind("}")
    if lo < 0 or hi <= lo:
        return None
    try:
        data = json.loads(text[lo:hi + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def has_foreign_script(parsed):
    return bool(parsed) and bool(_CJK.search(json.dumps(parsed, ensure_ascii=False)))


def normalize_card(parsed, raw, scene):
    """Đưa thẻ về đúng khuôn và đổi mốc sự kiện sang giờ phim tuyệt đối."""
    dur = scene["end"] - scene["start"]
    card = {k: (parsed or {}).get(k) for k in CARD_FIELDS}
    if not card["mo_ta"]:
        card["mo_ta"] = raw.strip() if parsed is None else ""
    for k in ("boi_canh", "chu_tren_man_hinh", "am_thanh_khong_khi", "y_chinh_loi_thoai"):
        v = card[k]
        card[k] = v if isinstance(v, str) else ("" if v is None else json.dumps(v, ensure_ascii=False))
    card["nhan_vat"] = [n for n in (card["nhan_vat"] or []) if isinstance(n, dict)]
    events, seen = [], set()
    for ev in [e for e in (card.pop("su_kien") or []) if isinstance(e, dict)]:
        desc = str(ev.get("mo_ta", "")).strip()
        if not desc or desc.lower() in seen:      # model 7B đôi khi vẫn lặp y hệt một câu
            continue
        seen.add(desc.lower())
        try:
            rel = float(ev.get("giay", 0))
        except (TypeError, ValueError):
            rel = 0.0
        t = scene["start"] + min(max(rel, 0.0), dur)
        events.append({"id": store.event_id(scene["id"], len(events) + 1), "t": round(t, 2),
                       "t_hms": store.hms(t), "mo_ta": desc})
    events.sort(key=lambda e: e["t"])
    for i, e in enumerate(events, 1):
        e["id"] = store.event_id(scene["id"], i)
    return card, events


def save_keyframes(project, scene_id, frames, times):
    """3 khung tiêu biểu (1/6, 1/2, 5/6 cảnh), cạnh dài 640px để không phình ổ đĩa."""
    out = []
    n = len(frames)
    for k, pos in enumerate((1 / 6, 1 / 2, 5 / 6), 1):
        i = min(n - 1, int(pos * n))
        img = frames[i].copy()
        img.thumbnail((KEYFRAME_SIDE, KEYFRAME_SIDE))
        path = project.path("04_keyframes", f"{scene_id}_k{k}.jpg")
        tmp = path + ".tmp.jpg"
        img.convert("RGB").save(tmp, "JPEG", quality=85)
        os.replace(tmp, path)
        out.append({"file": project.rel(path), "t": times[i]})
    return out


def describe_scene(backend, project, profile, title, scene, dialogue, previous):
    """Trả về (raw cuối, parsed, bản ghi các lượt, khung hình, mốc khung)."""
    from vision_analyzer_app import sample_frames

    dur = scene["end"] - scene["start"]
    want = max(4, min(profile["vision_max_frames"], int(round(dur * profile["vision_fps"]))))
    per_pass = backend.max_frames()
    passes_log = []

    if per_pass >= want:
        want -= want % 2
        frames, times = sample_frames(project.source_path, scene["start"], scene["end"], want)
        prompt = single_pass_prompt(title, scene, len(frames), dialogue, previous)
        raw = backend.describe(frames, len(frames) / max(dur, 1e-3), prompt)
        passes_log.append({"kind": "nhin", "start": scene["start"], "end": scene["end"],
                           "frames": len(frames), "output": raw})
        all_frames, all_times = frames, times
    else:
        per_pass = max(MIN_FRAMES_PER_PASS, per_pass)
        n = min(MAX_PASSES, math.ceil(want / per_pass))
        per_pass = max(MIN_FRAMES_PER_PASS, min(per_pass, math.ceil(want / n)))
        per_pass -= per_pass % 2
        observations, all_frames, all_times = [], [], []
        for k in range(n):
            a, b = dur * k / n, dur * (k + 1) / n
            frames, times = sample_frames(project.source_path, scene["start"] + a, scene["start"] + b, per_pass)
            text = backend.describe(frames, len(frames) / max(b - a, 1e-3),
                                    observe_prompt(title, k + 1, n, a, b, len(frames), dialogue),
                                    max_new_tokens=700)
            passes_log.append({"kind": "nhin", "start": round(scene["start"] + a, 3),
                               "end": round(scene["start"] + b, 3), "frames": len(frames), "output": text})
            observations.append(f"--- Đoạn {k + 1} (giây {a:.0f}–{b:.0f} của cảnh) ---\n{_shift_times(text, a)}")
            all_frames += frames
            all_times += times
        raw = backend.text(merge_prompt(title, scene, observations, dialogue, previous))
        passes_log.append({"kind": "gop", "output": raw})

    parsed = parse_card(raw)
    if has_foreign_script(parsed):
        fixed = backend.text(fix_language_prompt(raw))
        passes_log.append({"kind": "sua_ngon_ngu", "output": fixed})
        fixed_parsed = parse_card(fixed)
        if fixed_parsed and not has_foreign_script(fixed_parsed):
            raw, parsed = fixed, fixed_parsed
    return raw, parsed, passes_log, all_frames, all_times


def main(project, profile, args):
    from film import vision_backends

    plan = read_json(project.path("02_phan_canh", "scene_plan.json"))
    if not plan:
        raise RuntimeError("Chưa có kế hoạch cảnh (02_phan_canh/scene_plan.json).")
    scenes = plan["scenes"]
    # Phân tích lại (settings.cards_reanalyze): làm lại những cảnh mà bản mới nhất được tạo bằng
    # phiên bản prompt khác bản hiện tại, ghi thành .rN mới - bản cũ vẫn giữ nguyên. Vẫn chạy
    # tiếp được từ checkpoint: cảnh nào đã có bản theo prompt hiện tại thì bỏ qua.
    reanalyze = bool(project.data.get("settings", {}).get("cards_reanalyze"))

    def needs_work(s):
        if not store.has_node(project, "scene", s["id"]):
            return True
        if reanalyze:
            latest = store.read_node(project, "scene", s["id"])
            return latest.get("provenance", {}).get("prompt_version") != PROMPT_VERSION
        return False

    todo = [s for s in scenes if needs_work(s)]
    if not todo:
        stage_io.progress(100, f"Cả {len(scenes)} cảnh đều đã có file — không còn gì để làm.")
        return {"scenes": len(scenes), "new": 0}

    dialogue_doc = (read_json(project.path("01_loi_thoai", "dialogue.json"), {}) or {})
    lines = dialogue_doc.get("lines", [])
    done_before = len(scenes) - len(todo)
    stage_io.progress(100 * done_before / len(scenes),
                      f"{done_before}/{len(scenes)} cảnh đã có sẵn — nạp model thị giác cho {len(todo)} cảnh còn lại...")

    backend = vision_backends.create(profile, project.path("tmp", "vl"), project.path("10_nhat_ky"))
    stage_io.model_loaded(message="Đã nạp model thị giác.")
    parse_errors, t_scene = 0, []

    try:
        for n, scene in enumerate(todo, 1):
            t0 = time.time()
            previous = []
            for k in (scene["index"] - 2, scene["index"] - 1):
                prev = store.read_node(project, "scene", store.scene_id(k)) if k >= 1 else None
                if prev:
                    previous.append({"id": prev["id"], "start": prev["start"],
                                     "summary": (prev["card"]["mo_ta"] or "")[:300]})
            dlg = scene_dialogue(lines, scene["start"], scene["end"])
            raw, parsed, passes_log, frames, times = describe_scene(
                backend, project, profile, project.name, scene, dlg, previous)
            if not frames:
                raise RuntimeError(f"Không đọc được khung hình nào của {scene['id']} ({scene['start_hms']}).")
            parse_errors += parsed is None
            card, events = normalize_card(parsed, raw, scene)
            keyframes = save_keyframes(project, scene["id"], frames, times)
            dur = scene["end"] - scene["start"]

            existed = store.has_node(project, "scene", scene["id"])
            store.write_node(project, "scene", scene["id"], new_revision=existed, data={
                "complete": True,
                "index": scene["index"],
                "start": scene["start"], "end": scene["end"],
                "start_hms": store.hms(scene["start"]), "end_hms": store.hms(scene["end"]),
                "duration": round(dur, 3),
                "shots": scene["shots"],
                "frames": {"count": len(frames), "sample_fps": round(len(frames) / max(dur, 1e-3), 3),
                           "times": times, "max_pixels": backend.max_pixels,
                           "passes": sum(1 for p in passes_log if p["kind"] == "nhin")},
                "keyframes": keyframes,
                "dialogue": [{**d, "source": dialogue_doc.get("source")} for d in dlg],
                "card": card,
                "events": events,
                "vision": {"raw": raw, "parsed_ok": parsed is not None, "passes": passes_log},
                "context_used": [p["id"] for p in previous],
                "provenance": {"model": backend.model, "backend": backend.name, "profile": profile["name"],
                               "prompt_version": PROMPT_VERSION, "created_at": now_iso(),
                               "seconds": round(time.time() - t0, 1)},
            })
            t_scene.append(time.time() - t0)
            avg = sum(t_scene) / len(t_scene)
            left = (len(todo) - n) * avg
            done = done_before + n
            n_look = sum(1 for p in passes_log if p["kind"] == "nhin")
            stage_io.progress(100 * done / len(scenes),
                              f"{scene['id']} ({scene['start_hms'][:8]}) xong: {len(frames)} khung / {n_look} lượt nhìn, "
                              f"{t_scene[-1]:.0f}s — {done}/{len(scenes)} cảnh, còn khoảng {left / 60:.0f} phút",
                              eta_seconds=round(left), sec_per_scene=round(avg, 1))
            stage_io.check_pause(project, "cards")
    finally:
        backend.close()   # llama-server của backend GGUF phải tắt cả khi bị tạm dừng giữa chừng

    return {"scenes": len(scenes), "new": len(todo), "parse_errors": parse_errors,
            "avg_sec_per_scene": round(sum(t_scene) / len(t_scene), 1)}


if __name__ == "__main__":
    stage_io.run_stage("cards", main)
