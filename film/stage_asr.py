"""Bước [2] Bóc băng bằng faster-whisper large-v3 - chỉ chạy khi phim không có phụ đề.

Model: Systran/faster-whisper-large-v3 nằm ở cache Hugging Face MẶC ĐỊNH (~/.cache/huggingface/hub)
- cùng chỗ mà faster-whisper, WhisperX… của các ứng dụng khác tự tìm, nên dùng chung được, không
tải lại. Máy đã có ~/.cache/whisper/large-v3.pt thì tools/convert_whisper_pt.py chuyển đổi và đặt
thẳng vào đó. Hồ sơ "high" chạy float16, "low" (GTX 1060) chạy int8.

Phim nhiều thứ tiếng:
  - Nhận diện ngôn ngữ trên 8 khung 30 giây rải đều khắp phim chứ không chỉ 30 giây đầu (đầu
    phim hay là nhạc/logo - nhận nhầm ngôn ngữ ngay từ đó là sai cả phim).
  - Nếu không có ngôn ngữ nào chiếm áp đảo -> bật multilingual (nhận diện lại theo từng đoạn).
  - settings.language trong dự án ép cứng một ngôn ngữ nếu tự nhận sai.

Đầu ra: 01_loi_thoai/dialogue.json + dialogue.srt (cùng khuôn với phụ đề có sẵn, thêm mốc từng từ).
"""

import os
import re

from film import stage_io, subtitles
from film.project import write_json_atomic

WHISPER_REPO = "Systran/faster-whisper-large-v3"
SAMPLE_RATE = 16000
LANG_WINDOWS = 8
DOMINANT_SHARE = 0.6
MAX_LINE_SEC = 7.0
_SENT_END = re.compile(r"[.?!…。？！]$")


MIN_SPEECH_SEC = 4.0     # vùng có ít tiếng nói hơn mức này thì không cho bầu (nhạc, tiếng động)
MIN_LANG_PROB = 0.6


def whisper_model_path():
    """Thư mục model: WHISPER_MODEL_DIR nếu có đặt, không thì bản trong cache Hugging Face mặc định.

    Chỉ tìm trong máy (local_files_only): thiếu model thì báo rõ để người dùng chủ động tải hoặc
    chuyển đổi, chứ không âm thầm tải 3GB giữa lúc đang chạy phim."""
    env = os.environ.get("WHISPER_MODEL_DIR")
    if env:
        return env
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(WHISPER_REPO, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Chưa có {WHISPER_REPO} trong cache Hugging Face — xem docs/PHIM_DAI.md "
                           f"(tải về, hoặc chuyển đổi từ ~/.cache/whisper/large-v3.pt).") from exc


def detect_languages(model, audio):
    """Bầu chọn ngôn ngữ trên LANG_WINDOWS vùng rải đều khắp phim, CHỈ trên phần có tiếng nói.

    Không bầu thẳng trên khung 30 giây thô: khung nào nhiều nhạc là Whisper đoán bừa một ngôn
    ngữ hiếm (hay gặp "nn" - Na Uy Nynorsk) với độ tin cậy vừa phải, đủ kéo cả phim sang chế
    độ nhiều thứ tiếng. Nên: VAD tìm các đoạn có tiếng nói, mỗi vùng gom tối đa 30 giây tiếng
    nói của vùng đó, phiếu được cân theo số giây tiếng nói thật.
    Trả về (ngôn ngữ chính, tỉ lệ phiếu, bảng phiếu)."""
    import numpy as np
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    speech = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=500))
    if not speech:
        return None, 0.0, {}
    total = len(audio)
    votes = {}
    for r in range(LANG_WINDOWS):
        lo, hi = r * total / LANG_WINDOWS, (r + 1) * total / LANG_WINDOWS
        pieces, got = [], 0
        for ch in speech:
            if ch["end"] <= lo or ch["start"] >= hi:
                continue
            s, e = int(max(ch["start"], lo)), int(min(ch["end"], hi))
            take = min(e - s, 30 * SAMPLE_RATE - got)
            if take <= 0:
                break
            pieces.append(audio[s:s + take])
            got += take
        secs = got / SAMPLE_RATE
        if secs < MIN_SPEECH_SEC:
            continue
        try:
            lang, prob, _ = model.detect_language(audio=np.concatenate(pieces))
        except Exception:  # noqa: BLE001
            continue
        if lang and prob >= MIN_LANG_PROB:
            votes[lang] = votes.get(lang, 0.0) + prob * secs
    if not votes:
        return None, 0.0, {}
    top = max(votes, key=votes.get)
    return top, votes[top] / sum(votes.values()), {k: round(v, 1) for k, v in votes.items()}


def _is_hallucination(seg, prev_texts):
    """Whisper hay "bịa" câu ở đoạn nhạc/im lặng: nén lặp cao, xác suất không-lời cao, lặp y hệt."""
    if seg.compression_ratio > 2.4:
        return True
    if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
        return True
    text = seg.text.strip()
    return bool(text) and prev_texts.count(text) >= 2


def _split_long(seg, lang):
    """Chặt đoạn dài (chế độ batched hay gộp nhiều câu thành một đoạn ~15 giây) theo dấu câu,
    dùng mốc thời gian từng từ - để lời thoại khớp sát cảnh hơn."""
    words = [w for w in (seg.words or []) if w.word.strip()]
    base = {"lang": lang}
    if seg.end - seg.start <= MAX_LINE_SEC or not words:
        return [{**base, "start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text.strip(),
                 "words": [[w.word, round(w.start, 3), round(w.end, 3), round(w.probability, 3)] for w in words]}]
    lines, cur = [], []
    for w in words:
        cur.append(w)
        long_enough = cur[-1].end - cur[0].start >= 1.5
        if (_SENT_END.search(w.word.strip()) and long_enough) or cur[-1].end - cur[0].start >= MAX_LINE_SEC * 1.5:
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    return [{**base, "start": round(ws[0].start, 3), "end": round(ws[-1].end, 3),
             "text": "".join(w.word for w in ws).strip(),
             "words": [[w.word, round(w.start, 3), round(w.end, 3), round(w.probability, 3)] for w in ws]}
            for ws in lines]


def main(project, profile, args):
    import torch  # noqa: F401 - nạp DLL cuDNN/cuBLAS của torch trước để ctranslate2 dùng chung
    from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio

    wav = project.path("01_loi_thoai", "audio.wav")
    if not os.path.isfile(wav):
        raise RuntimeError("Chưa có audio.wav — bước Chuẩn bị chưa tách audio.")
    compute = profile["whisper_compute"]
    stage_io.progress(2, f"Nạp Whisper large-v3 ({compute})...")
    model = WhisperModel(whisper_model_path(), device="cuda", compute_type=compute,
                         cpu_threads=profile["torch_threads"])

    stage_io.model_loaded(4, "Đã nạp Whisper.")
    settings = project.data.get("settings", {})
    forced = (settings.get("language") or "auto").lower()
    stage_io.progress(5, "Đọc audio...")
    audio = decode_audio(wav, sampling_rate=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE

    if forced != "auto":
        lang, share, votes, multilingual = forced, 1.0, {}, False
        stage_io.progress(8, f"Dùng ngôn ngữ đã chọn: {lang}")
    else:
        stage_io.progress(8, f"Nhận diện ngôn ngữ trên {LANG_WINDOWS} đoạn rải khắp phim...")
        lang, share, votes = detect_languages(model, audio)
        multilingual = lang is None or share < DOMINANT_SHARE
        stage_io.progress(12, f"Ngôn ngữ: {votes or 'không rõ'} -> "
                              f"{'nhiều thứ tiếng (nhận diện theo từng đoạn)' if multilingual else lang}")

    pipe = BatchedInferencePipeline(model)
    segments, info = pipe.transcribe(
        audio, language=None if multilingual else lang, multilingual=multilingual,
        batch_size=8 if profile["name"] == "high" else 4,
        vad_filter=True, word_timestamps=True, condition_on_previous_text=False,
    )
    lines, dropped, recent = [], 0, []
    for seg in segments:
        stage_io.check_pause(project, "asr")
        if _is_hallucination(seg, recent):
            dropped += 1
            continue
        recent = (recent + [seg.text.strip()])[-5:]
        seg_lang = getattr(seg, "language", None) or info.language
        lines.extend(_split_long(seg, seg_lang))
        stage_io.progress(12 + 85 * min(1.0, seg.end / max(duration, 1)),
                          f"Đã bóc băng tới {int(seg.end // 60)}:{int(seg.end % 60):02d}")

    dialogue = {
        "source": "whisper",
        "model": "faster-whisper-large-v3",
        "compute_type": compute,
        "lang": info.language if not multilingual else "multi",
        "language_votes": votes,
        "multilingual": multilingual,
        "dropped_hallucinations": dropped,
        "lines": lines,
    }
    write_json_atomic(project.path("01_loi_thoai", "dialogue.json"), dialogue)
    subtitles.write_srt(lines, project.path("01_loi_thoai", "dialogue.srt"))
    stage_io.progress(100, f"Xong: {len(lines)} câu, bỏ {dropped} đoạn nghi bịa chữ.")
    return {"lines": len(lines), "lang": dialogue["lang"], "dropped": dropped}


if __name__ == "__main__":
    stage_io.run_stage("asr", main)
