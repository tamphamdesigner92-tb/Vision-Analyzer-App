"""Bước [1] Chuẩn bị: đọc thông tin phim, tìm lời thoại có sẵn, tách audio nếu cần bóc băng.

Có phụ đề (.srt cùng tên, hoặc phụ đề chữ nhúng trong phim) thì dùng luôn làm lời thoại và
BỎ QUA hẳn bước bóc băng - không tách audio. Đặt settings.force_whisper = true trong dự án nếu
nghi phụ đề lệch/thiếu và muốn bóc băng lại bằng Whisper.

Đầu ra:
  00_nguon/probe.json                   thông tin phim (thời lượng, fps, các luồng)
  00_nguon/<phụ đề gốc>.srt             bản chép phụ đề gốc (để dự án tự chứa đủ)
  01_loi_thoai/dialogue.json            lời thoại chuẩn hoá: {source, lang, lines:[{start,end,text}]}
  01_loi_thoai/dialogue.srt
  01_loi_thoai/audio.wav                (chỉ khi phải bóc băng) 16kHz mono
"""

import json
import os
import shutil
import subprocess

from film import stage_io, subtitles
from film.project import write_json_atomic


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration,size,format_name:stream=index,codec_type,codec_name,width,height,"
         "avg_frame_rate,channels,sample_rate:stream_tags=language,title",
         "-of", "json", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    )
    data = json.loads(out.stdout)
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    fps = None
    if video.get("avg_frame_rate") and video["avg_frame_rate"] != "0/0":
        num, den = video["avg_frame_rate"].split("/")
        fps = float(num) / float(den) if float(den) else None
    return {
        "duration_sec": float(fmt.get("duration") or 0),
        "size": int(fmt.get("size") or 0),
        "container": fmt.get("format_name"),
        "width": video.get("width"),
        "height": video.get("height"),
        "fps": fps,
        "audio_streams": [s for s in streams if s.get("codec_type") == "audio"],
        "subtitle_streams": [s for s in streams if s.get("codec_type") == "subtitle"],
    }


def extract_audio(src, wav):
    """16kHz mono - đúng thứ Whisper cần. Ghi ra file tạm rồi đổi tên để không để lại WAV dở."""
    tmp = wav + ".part.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", tmp], check=True, capture_output=True)
    os.replace(tmp, wav)


def main(project, profile, args):
    src = project.source_path
    status = project.check_source()
    if status != "ok":
        raise RuntimeError(
            "Không tìm thấy phim nguồn." if status == "missing"
            else "File phim nguồn đã thay đổi so với lúc tạo dự án."
        )

    stage_io.model_loaded(5, "Đọc thông tin phim (ffprobe)...")
    info = probe(src)
    write_json_atomic(project.path("00_nguon", "probe.json"), info)
    duration = info["duration_sec"]
    settings = project.data.get("settings", {})

    dialogue = None
    if not settings.get("force_whisper"):
        stage_io.progress(20, "Tìm phụ đề có sẵn...")
        for srt in subtitles.find_sidecar_srts(src):
            lines, enc = subtitles.read_srt(srt, duration)
            if lines:
                copy = project.path("00_nguon", os.path.basename(srt))
                shutil.copy2(srt, copy)
                dialogue = {"source": "srt", "source_file": project.rel(copy), "encoding": enc,
                            "lang": subtitles.lang_from_filename(srt), "lines": lines}
                break
        if dialogue is None:
            for sub in subtitles.list_embedded(src):
                out = project.path("00_nguon", f"embedded_s{sub['order']}.srt")
                try:
                    subtitles.extract_embedded(src, sub["order"], out)
                except subprocess.CalledProcessError:
                    continue
                lines, enc = subtitles.read_srt(out, duration)
                if lines:
                    dialogue = {"source": "embedded", "source_file": project.rel(out), "encoding": enc,
                                "lang": sub.get("lang"), "lines": lines}
                    break

    if dialogue is not None:
        write_json_atomic(project.path("01_loi_thoai", "dialogue.json"), dialogue)
        subtitles.write_srt(dialogue["lines"], project.path("01_loi_thoai", "dialogue.srt"))
        stage_io.progress(100, f"Dùng phụ đề có sẵn {dialogue['source_file']} "
                               f"({len(dialogue['lines'])} câu) — bỏ qua bóc băng.")
        return {"dialogue_source": dialogue["source"], "subtitle_file": dialogue["source_file"],
                "lines": len(dialogue["lines"]), "duration_sec": duration}

    if not info["audio_streams"]:
        write_json_atomic(project.path("01_loi_thoai", "dialogue.json"),
                          {"source": "none", "lang": None, "lines": []})
        stage_io.progress(100, "Phim không có tiếng — không có lời thoại để bóc băng.")
        return {"dialogue_source": "none", "duration_sec": duration}

    stage_io.progress(40, "Không có phụ đề — tách audio 16kHz mono cho bước bóc băng...")
    extract_audio(src, project.path("01_loi_thoai", "audio.wav"))
    stage_io.progress(100, "Đã tách audio.")
    return {"dialogue_source": "whisper", "duration_sec": duration}


if __name__ == "__main__":
    stage_io.run_stage("prepare", main)
