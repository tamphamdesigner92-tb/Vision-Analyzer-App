"""Bước [3] Cắt cảnh: tìm ranh giới shot (PySceneDetect) rồi gộp thành CẢNH 20–90 giây.

Cảnh là đơn vị lưu trữ tầng 1 (mỗi cảnh một file JSON) và là đơn vị mô tả của bước 4, nên:
  - Gộp nhiều shot ngắn thành một cảnh đủ dài để có nội dung (>= scene_min_sec).
  - Shot quá dài (một cú máy dài, cảnh hội thoại tĩnh) bị chẻ ra, và điểm chẻ được dời vào
    khoảng lặng gần nhất giữa hai câu thoại để không cắt ngang câu.
  - Kế hoạch cảnh CỐ ĐỊNH sau khi bước 4 đã chạy: id cảnh (scene_0001…) là khoá trỏ từ mọi
    tầng trên xuống, đổi ranh giới là gãy hết liên kết. Bước này từ chối chạy lại khi đã có
    file cảnh trong 03_canh/.

Đầu ra: 02_phan_canh/shots.json, 02_phan_canh/scene_plan.json
"""

import math

from film import stage_io, store
from film.project import read_json, write_json_atomic

CHUNK_SEC = 60          # xử lý theo từng phút để báo tiến độ và nhận lệnh tạm dừng
GAP_SEARCH_SEC = 15     # tìm khoảng lặng trong phạm vi ±15 giây quanh điểm chẻ lý tưởng


def detect_shots(path, duration, threads, project):
    import cv2
    from scenedetect import AdaptiveDetector, SceneManager, open_video

    cv2.setNumThreads(threads)
    video = open_video(path, backend="opencv")
    sm = SceneManager()
    sm.auto_downscale = True
    sm.add_detector(AdaptiveDetector())
    t = 0.0
    while t < duration:
        t = min(duration, t + CHUNK_SEC)
        sm.detect_scenes(video, end_time=t)
        stage_io.progress(5 + 80 * t / max(duration, 1),
                          f"Tìm ranh giới shot: {store.hms(t, ms=False)} / {store.hms(duration, ms=False)}")
        stage_io.check_pause(project, "scenes")
    shots = [(s.seconds, e.seconds) for s, e in sm.get_scene_list(start_in_scene=True)]
    return shots or [(0.0, duration)]


def _silence_points(lines):
    """Điểm giữa các khoảng lặng giữa hai câu thoại liên tiếp (>= 0.3 giây)."""
    pts = []
    for a, b in zip(lines, lines[1:]):
        if b["start"] - a["end"] >= 0.3:
            pts.append((a["end"] + b["start"]) / 2)
    return pts


def _best_cut(ideal, silences, lines, lo, hi):
    near = [p for p in silences if lo < p < hi and abs(p - ideal) <= GAP_SEARCH_SEC]
    if near:
        return min(near, key=lambda p: abs(p - ideal))
    for ln in lines:                       # rơi giữa một câu: dời tới cuối câu đó
        if ln["start"] < ideal < ln["end"] and ln["end"] < hi:
            return ln["end"]
    return ideal


def split_long(start, end, max_len, target, silences, lines):
    length = end - start
    if length <= max_len:
        return [(start, end)]
    n = math.ceil(length / target)
    cuts, prev = [], start
    for k in range(1, n):
        cut = _best_cut(start + k * length / n, silences, lines, prev + 1, end - 1)
        cuts.append(cut)
        prev = cut
    bounds = [start] + cuts + [end]
    return list(zip(bounds, bounds[1:]))


def group_scenes(shots, lines, min_len, max_len):
    """shots [(start, end)] -> cảnh [(start, end, [chỉ số shot])]."""
    target = (min_len + max_len) / 2
    silences = _silence_points(lines)
    pieces = []
    for i, (s, e) in enumerate(shots):
        for ps, pe in split_long(s, e, max_len, target, silences, lines):
            pieces.append((ps, pe, i))

    scenes, cur = [], None
    for s, e, i in pieces:
        if cur is None:
            cur = [s, e, [i]]
            continue
        cur_len = cur[1] - cur[0]
        if cur_len >= min_len and (cur_len + (e - s) > max_len or cur_len >= target):
            scenes.append(cur)
            cur = [s, e, [i]]
        else:
            cur[1] = e
            if i not in cur[2]:
                cur[2].append(i)
    if cur:
        # Mẩu cuối quá ngắn thì nhập vào cảnh trước, miễn không phình quá xa max_len.
        if scenes and cur[1] - cur[0] < min_len / 2 and cur[1] - scenes[-1][0] <= max_len * 1.2:
            scenes[-1][1] = cur[1]
            scenes[-1][2].extend(i for i in cur[2] if i not in scenes[-1][2])
        else:
            scenes.append(cur)
    return [(round(s, 3), round(e, 3), idx) for s, e, idx in scenes]


def main(project, profile, args):
    if store.list_nodes(project, "scene"):
        raise RuntimeError("Đã có file cảnh trong 03_canh/ — không cắt cảnh lại để khỏi gãy liên kết "
                           "giữa các tầng. Tạo dự án mới nếu muốn chia cảnh khác.")
    info = read_json(project.path("00_nguon", "probe.json"), {})
    duration = info.get("duration_sec") or project.data["source"].get("duration_sec") or 0
    dialogue = read_json(project.path("01_loi_thoai", "dialogue.json"), {}) or {}
    lines = dialogue.get("lines", [])

    stage_io.model_loaded(2, "Mở phim để tìm ranh giới shot...")
    shots = detect_shots(project.source_path, duration, profile["torch_threads"], project)
    write_json_atomic(project.path("02_phan_canh", "shots.json"),
                      {"detector": "AdaptiveDetector", "shots": [{"i": i, "start": round(s, 3), "end": round(e, 3)}
                                                                 for i, (s, e) in enumerate(shots)]})

    stage_io.progress(90, f"Tìm được {len(shots)} shot — gộp thành cảnh...")
    groups = group_scenes(shots, lines, profile["scene_min_sec"], profile["scene_max_sec"])
    plan = {
        "min_sec": profile["scene_min_sec"],
        "max_sec": profile["scene_max_sec"],
        "profile": profile["name"],
        "scenes": [{"id": store.scene_id(n), "index": n, "start": s, "end": e, "shots": idx,
                    "start_hms": store.hms(s), "end_hms": store.hms(e)}
                   for n, (s, e, idx) in enumerate(groups, 1)],
    }
    write_json_atomic(project.path("02_phan_canh", "scene_plan.json"), plan)
    lens = [e - s for s, e, _ in groups]
    stage_io.progress(100, f"{len(groups)} cảnh (dài {min(lens):.0f}–{max(lens):.0f} giây, "
                           f"trung bình {sum(lens) / len(lens):.0f} giây).")
    return {"shots": len(shots), "scenes": len(groups)}


if __name__ == "__main__":
    stage_io.run_stage("scenes", main)
