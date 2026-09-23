"""Chuyển file Whisper định dạng openai-whisper (.pt) sang định dạng faster-whisper (CTranslate2).

Dùng khi máy đã có sẵn trọng số (vd: ~/.cache/whisper/large-v3.pt do gói openai-whisper tải
về) để khỏi tải lại hàng GB. Chỉ tải bộ file cấu hình + tokenizer (~5MB, không có trọng số)
của repo Hugging Face tương ứng, vì .pt không mang theo tokenizer lẫn alignment heads - thứ
quyết định độ chính xác mốc thời gian từng từ.

    .venv\\Scripts\\python.exe tools\\convert_whisper_pt.py ^
        --pt %USERPROFILE%\\.cache\\whisper\\large-v3.pt ^
        --hf-config openai/whisper-large-v3 ^
        --out models\\faster-whisper-large-v3
"""

import argparse
import os
import shutil
import sys

import torch
from huggingface_hub import hf_hub_download

# Đổi tên khoá từ openai-whisper sang transformers (theo script convert_openai_to_hf.py
# chính thức của Hugging Face, script đó không đi kèm gói transformers đã cài).
RENAMES = [
    ("blocks", "layers"),
    ("mlp.0", "fc1"),
    ("mlp.2", "fc2"),
    ("mlp_ln", "final_layer_norm"),
    (".attn.query", ".self_attn.q_proj"),
    (".attn.key", ".self_attn.k_proj"),
    (".attn.value", ".self_attn.v_proj"),
    (".attn_ln", ".self_attn_layer_norm"),
    (".attn.out", ".self_attn.out_proj"),
    (".cross_attn.query", ".encoder_attn.q_proj"),
    (".cross_attn.key", ".encoder_attn.k_proj"),
    (".cross_attn.value", ".encoder_attn.v_proj"),
    (".cross_attn_ln", ".encoder_attn_layer_norm"),
    (".cross_attn.out", ".encoder_attn.out_proj"),
    ("decoder.ln.", "decoder.layer_norm."),
    ("encoder.ln.", "encoder.layer_norm."),
    ("token_embedding", "embed_tokens"),
    ("encoder.positional_embedding", "encoder.embed_positions.weight"),
    ("decoder.positional_embedding", "decoder.embed_positions.weight"),
    ("ln_post", "layer_norm"),
]

CONFIG_FILES = [
    "config.json", "generation_config.json", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "added_tokens.json", "normalizer.json",
]


def rename(key):
    for old, new in RENAMES:
        key = key.replace(old, new)
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--hf-config", default="openai/whisper-large-v3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantization", default="float16")
    args = ap.parse_args()

    from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration
    from ctranslate2.converters import TransformersConverter

    out = os.path.abspath(args.out)
    hf_dir = out + "_hf_tmp"
    if os.path.exists(out):
        sys.exit(f"[X] Thư mục đích đã tồn tại: {out}")
    os.makedirs(hf_dir, exist_ok=True)

    print(f"[*] Tải cấu hình + tokenizer của {args.hf_config} (không tải trọng số)...", flush=True)
    for name in CONFIG_FILES:
        try:
            shutil.copy(hf_hub_download(args.hf_config, name), os.path.join(hf_dir, name))
        except Exception as exc:  # noqa: BLE001 - vài file là tuỳ chọn
            print(f"    bỏ qua {name}: {type(exc).__name__}", flush=True)

    print(f"[*] Đọc trọng số {args.pt}...", flush=True)
    ckpt = torch.load(args.pt, map_location="cpu", weights_only=False)
    dims = ckpt["dims"]
    state = {rename(k): v for k, v in ckpt["model_state_dict"].items()}

    config = WhisperConfig.from_pretrained(hf_dir)
    # Đối chiếu kích thước: cấu hình tải về phải đúng là của checkpoint này.
    checks = {
        "d_model": dims["n_audio_state"], "encoder_layers": dims["n_audio_layer"],
        "decoder_layers": dims["n_text_layer"], "num_mel_bins": dims["n_mels"],
        "vocab_size": dims["n_vocab"], "max_source_positions": dims["n_audio_ctx"],
        "max_target_positions": dims["n_text_ctx"],
    }
    wrong = {k: (getattr(config, k), v) for k, v in checks.items() if getattr(config, k) != v}
    if wrong:
        sys.exit(f"[X] Cấu hình {args.hf_config} không khớp checkpoint: {wrong}")

    model = WhisperForConditionalGeneration(config)
    missing, unexpected = model.model.load_state_dict(state, strict=False)
    # proj_out dùng chung trọng số với embed_tokens (tie), nên thiếu nó là đúng.
    missing = [k for k in missing if k != "proj_out.weight"]
    if missing or unexpected:
        sys.exit(f"[X] Lệch khoá trọng số. Thiếu: {missing[:10]} | Thừa: {unexpected[:10]}")
    model.tie_weights()
    model = model.half()
    # BẮT BUỘC: không có dòng này thì save_pretrained() ghi đè generation_config.json chính
    # thức bằng bản mặc định sinh từ config - mất alignment_heads (mốc thời gian từng từ
    # sai) và lang_to_id (không nhận diện được ngôn ngữ).
    model.generation_config = GenerationConfig.from_pretrained(hf_dir)
    print(f"[*] Nạp đủ {sum(p.numel() for p in model.parameters()) / 1e9:.2f} tỷ tham số, "
          f"không thiếu/thừa khoá nào.", flush=True)
    model.save_pretrained(hf_dir, safe_serialization=True)
    del model, state, ckpt

    class _Converter(TransformersConverter):
        # ctranslate2 >= 4.7 gọi from_pretrained(dtype=...) theo API transformers mới, còn
        # transformers 4.51.3 (ghim vì AutoAWQ) chỉ hiểu torch_dtype=.
        def load_model(self, model_class, model_name_or_path, **kwargs):
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            return model_class.from_pretrained(model_name_or_path, **kwargs)

    print(f"[*] Chuyển sang CTranslate2 ({args.quantization}) -> {out}...", flush=True)
    _Converter(
        hf_dir,
        copy_files=["tokenizer.json", "preprocessor_config.json"],
        load_as_float16=True,
    ).convert(out, quantization=args.quantization)

    import json
    with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
        ct2 = json.load(f)
    expected = GenerationConfig.from_pretrained(hf_dir).alignment_heads
    if ct2.get("alignment_heads") != expected or not ct2.get("lang_ids"):
        sys.exit("[X] config.json đầu ra thiếu alignment_heads/lang_ids chính thức - giữ lại "
                 f"{hf_dir} để kiểm tra.")
    print(f"[*] alignment_heads khớp bản chính thức ({len(expected)} head), "
          f"{len(ct2['lang_ids'])} ngôn ngữ.", flush=True)

    shutil.rmtree(hf_dir)   # bản trung gian dạng transformers (~3GB), không cần giữ
    print(f"[OK] Xong: {out}", flush=True)


if __name__ == "__main__":
    main()
