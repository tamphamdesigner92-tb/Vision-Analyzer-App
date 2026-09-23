# Triển khai trên PC cấu hình thấp (GTX 1060 6GB)

Máy mục tiêu: **GTX 1060 6GB (MSI Gaming X) · Ryzen 7 2700X (8 nhân/16 luồng) · 32GB DDR4-3200 · MSI X470 Gaming Plus**.

App tự nhận máy này là hồ sơ **`low`** và đổi cách chạy. Đọc [PHIM_DAI.md](PHIM_DAI.md) để hiểu đường ống chung; tài liệu này chỉ nói những gì **khác** trên máy yếu.

> **Trạng thái:** phần mã cho hồ sơ `low` đã viết và đã kiểm tra logic trên máy chính. Nhưng chưa chạy thật với model GGUF trên GTX 1060. Lần chạy đầu trên PC đó chính là lần kiểm chứng: làm theo mục 7 và gửi lại log nếu có lỗi.

---

## 1. Vì sao máy này chạy khác máy chính

| Giới hạn của GTX 1060 (Pascal, sm 6.1) | Hậu quả | Cách app xử lý |
|---|---|---|
| AutoAWQ cần GPU từ Turing (sm 7.5) trở lên | Qwen2.5-VL-7B-**AWQ** không chạy được | Dùng Qwen2.5-VL-7B **GGUF Q4_K_M** qua llama.cpp |
| Tính fp16 chỉ bằng khoảng 1/64 tốc độ fp32 | Chạy fp16/bf16 cực chậm | Whisper chạy **int8**; model LLM và thị giác chạy GGUF lượng tử |
| Chỉ 6GB VRAM | Không chứa hết model | `--fit` của llama.cpp tự để phần không vừa lại RAM |
| CUDA 13 và PyTorch mới (cu128+) đã bỏ Pascal | Cài bản mới là không nhận GPU | Ghim **torch 2.5.1+cu121** và **llama.cpp bản CUDA 12.x** |
| GPU vừa tính vừa xuất hình ra màn hình | Dễ giật desktop | Chừa 1GB VRAM, batch nhỏ (`-ub 256`), ưu tiên tiến trình thấp |

Thông số hồ sơ `low` (trong [hardware_profile.py](../hardware_profile.py)):

| | Máy chính (`high`) | GTX 1060 (`low`) |
|---|---|---|
| Whisper | large-v3 fp16 | large-v3 **int8** (~2.7GB VRAM) |
| Mô tả cảnh | Qwen2.5-VL-7B AWQ, 1 khung/giây, ≤48 khung | Qwen2.5-VL-7B GGUF, **0.5 khung/giây, ≤12 khung, ảnh cạnh dài 448px** |
| Độ dài cảnh | 20–90 giây | **30–120 giây** (ít cảnh hơn) |
| Tổng hợp / hỏi đáp | Qwen3-30B-A3B, ngữ cảnh 64K | Qwen3-30B-A3B, ngữ cảnh **32K**, hỏi đáp nạp ≤20K token |
| Luồng CPU | 8 | **6** (chừa cho Windows) |

---

## 2. Chuẩn bị máy

1. **Driver NVIDIA** hỗ trợ CUDA 12.4 trở lên (dòng 550 trở lên). Kiểm tra bằng `nvidia-smi`: dòng "CUDA Version" phải ≥ 12.4.
   Không cần cài CUDA Toolkit, vì torch và llama.cpp đã kèm thư viện CUDA.
2. **Python 3.11 x64** (máy chính dùng 3.11.9). Khi cài, tích "Add python.exe to PATH".
3. **ffmpeg** có trong PATH (máy chính để ở `C:\ffmpeg\bin`). Kiểm tra: `ffmpeg -version`.
4. **Git** (nếu lấy code bằng git).
5. **Ổ đĩa:** cần khoảng **30GB trống**, gồm khoảng 27GB model và khoảng 1–2GB cho mỗi dự án phim 3 giờ.

---

## 3. Lấy mã nguồn

```
git clone https://github.com/tamphamdesigner92-tb/Vision-Analyzer-App.git
cd Vision-Analyzer-App
git checkout main
```

Hoặc chép cả thư mục app từ máy chính, **bỏ** `.venv/` (môi trường ảo không chép sang máy khác được) và `projects/`.

---

## 4. Cài môi trường Python

Chạy lần lượt trong thư mục app, đúng thứ tự. Đây cũng là cách cài ghi ở đầu [requirements.txt](../requirements.txt).

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Bỏ qua 2 dòng cài `autoawq` trong requirements, vì AWQ không chạy trên Pascal.

Kiểm tra torch nhận GPU:

```
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Kết quả phải là `True NVIDIA GeForce GTX 1060 6GB (6, 1)`.

---

## 5. Các model cần có trên PC GTX 1060

Máy chính **không** giữ các model riêng của máy yếu. Tải trực tiếp trên PC GTX 1060, hoặc chép các thư mục dùng chung từ máy chính sang để đỡ tải lại.

| # | Model | Dung lượng | Đặt ở đâu | Dùng cho |
|---|---|---|---|---|
| 1 | **llama.cpp b11120, bản CUDA 12.4 cho Windows**: `llama-b11120-bin-win-cuda-12.4-x64.zip` + `cudart-llama-bin-win-cuda-12.4-x64.zip` (GitHub ggml-org/llama.cpp, mục Releases) | ~645MB | giải nén cả hai vào `tools\llama.cpp\bin\` | chạy mọi model GGUF |
| 2 | **Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf** (repo `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF`) | 4.68GB | cache Hugging Face | mô tả cảnh |
| 3 | **mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf** (cùng repo) | 0.85GB | cache Hugging Face | phần "mắt" của model thị giác |
| 4 | **Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf** (repo `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF`) | 18.56GB | cache Hugging Face | tổng hợp + hỏi đáp |
| 5 | **Whisper large-v3 định dạng faster-whisper** (repo `Systran/faster-whisper-large-v3`) | 3.09GB | cache Hugging Face | bóc băng (bỏ qua nếu phim có .srt) |
| 6 | *(tuỳ chọn, cho tab 2)* **Qwen3-Reranker-0.6B** (repo `Qwen/Qwen3-Reranker-0.6B`) | ~1.2GB | cache Hugging Face | khớp kịch bản |

"Cache Hugging Face" là thư mục mặc định `%USERPROFILE%\.cache\huggingface\hub\`, dùng chung cho mọi ứng dụng —
**không** phải thư mục app hay thư mục dự án. PC GTX 1060 đã có ứng dụng khác tải sẵn model nào (vd WhisperX đã
có `Systran/faster-whisper-large-v3`) thì app dùng luôn, khỏi tải lại. Muốn để cache ở ổ khác: biến môi trường
chuẩn `HF_HOME=D:\hf`.

Mục 1, 4, 5 **giống hệt** máy chính. Có thể chép thẳng từ máy chính sang (giữ nguyên cấu trúc thư mục), rồi kiểm tra lại dung lượng file:
- `tools\llama.cpp\bin\` → cùng chỗ trong thư mục app trên PC GTX 1060
- `%USERPROFILE%\.cache\huggingface\hub\models--unsloth--Qwen3-30B-A3B-Instruct-2507-GGUF\` → cùng chỗ
- `%USERPROFILE%\.cache\huggingface\hub\models--Systran--faster-whisper-large-v3\` → cùng chỗ

### Tải trên PC GTX 1060

```
.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download as d; [print(d('ggml-org/Qwen2.5-VL-7B-Instruct-GGUF', f)) for f in ('Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf', 'mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf')]"
.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download as d; print(d('unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF', 'Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf'))"
```

Whisper large-v3, chọn **một** trong hai cách:
- **Máy đã có** `%USERPROFILE%\.cache\whisper\large-v3.pt` (của gói openai-whisper): chuyển đổi, không tải lại 3GB:
  ```
  .venv\Scripts\python.exe tools\convert_whisper_pt.py --pt %USERPROFILE%\.cache\whisper\large-v3.pt
  ```
  Script tự kiểm tra alignment heads và 100 ngôn ngữ, rồi đặt kết quả vào cache Hugging Face dưới tên
  `Systran/faster-whisper-large-v3` khi `model.bin` trùng sha256 với bản gốc (trên máy chính đã trùng).
- **Chưa có gì:** tải bản đã chuyển sẵn:
  ```
  .venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download as s; print(s('Systran/faster-whisper-large-v3'))"
  ```

Chỉ khi thật cần để model ở chỗ khác cache chuẩn: `VL_GGUF_DIR` (thư mục chứa 2 file mục 2–3),
`LLM_GGUF` (đường dẫn file mục 4), `WHISPER_MODEL_DIR` (thư mục mục 5). Mặc định không cần đặt gì.

Tab 2 (khớp kịch bản): model mặc định Qwen3-Reranker-4B cần khoảng 8GB, không vừa 6GB. Trước khi chạy app, đặt:

```
set RERANKER_MODEL_ID=Qwen/Qwen3-Reranker-0.6B
```

---

## 6. Kiểm tra trước khi chạy

```
.venv\Scripts\python.exe hardware_profile.py
```

Phải thấy `"name": "low"` và `"vision_backend": "llamacpp_gguf"`. Nếu ra `high` (không nên xảy ra với GTX 1060), ép bằng `set FILM_PROFILE=low`.

```
tools\llama.cpp\bin\llama-server.exe --version
```

Phải in ra `version: ... (build 11120 ...)`. Nếu báo thiếu DLL thì chưa giải nén gói `cudart-...zip`.

---

## 7. Chạy lần đầu (cũng là lần kiểm chứng)

Nên thử với một phim ngắn trước (vài phút, có tiếng):

1. Mở `start_app.bat` → tab **3 · Phim dài** → **Tạo dự án mới** → chọn phim → **Tạo** → **Chạy**.
2. Theo dõi 5 bước trên giao diện. Chip **VRAM** và **RAM trống / pagefile** ở thanh trạng thái phải ổn định:
   - VRAM không vượt khoảng 5GB (1GB chừa cho màn hình).
   - Pagefile không tăng liên tục.
3. Trong lúc chạy, thử mở một video YouTube để xem máy có giật không.
4. Nếu có lỗi, gửi lại các file trong `<thư mục dự án>\10_nhat_ky\`:
   - `cards.log`, `llama-server-vl.log`: bước mô tả cảnh qua llama.cpp
   - `synthesis.log`, `llama-server.log`: bước tổng hợp
   - `resource.log`: RAM/VRAM/pagefile theo thời gian
   - `llm_hong_*.txt`: nếu model trả về JSON hỏng

Dòng lệnh thay cho giao diện:

```
.venv\Scripts\python.exe -m film.cli create "D:\Phim\phim.mp4" --name "Thử GTX 1060"
.venv\Scripts\python.exe -m film.cli run <id in ra ở dòng trên>
```

---

## 8. Chuyển dự án giữa hai máy

Dự án mang đi được, vì mọi đường dẫn bên trong đều là tương đối:

- Có thể chạy các bước nhẹ trên PC GTX 1060 rồi chép cả thư mục dự án sang máy chính để chạy tiếp các bước nặng, hoặc ngược lại.
- Bên máy nhận: tab 3 → **Mở dự án…** → chọn thư mục dự án.
- Nếu phim nguồn nằm ở đường dẫn khác, app báo "không thấy file phim" và hiện nút **Chọn lại file phim…**.
- **Kế hoạch cảnh** (ranh giới cảnh) giữ nguyên theo máy đã chạy bước Cắt cảnh. Máy kia không cắt lại, để không gãy liên kết giữa các tầng.
- Cảnh đã mô tả không bị làm lại; cảnh còn thiếu được máy mới mô tả tiếp bằng backend của nó.
- Mỗi file cảnh ghi rõ `provenance.backend` (`transformers_awq` hay `llamacpp_gguf`), nên luôn biết cảnh nào do máy nào tạo.

---

## 9. Thời gian dự kiến (ước tính, chưa đo trên GTX 1060)

| Bước | Ước tính cho phim 3 giờ |
|---|---|
| Chuẩn bị + bóc băng (int8) | 30–60 phút (bỏ qua nếu có .srt) |
| Cắt cảnh (CPU Ryzen 2700X) | 20–40 phút |
| Mô tả cảnh (~150–200 cảnh) | 5–10 giờ |
| Tổng hợp (Qwen3-30B-A3B, 8–15 token/giây) | 1–2 giờ |
| **Tổng** | **khoảng 8–15 giờ → 1–2 đêm** |

Bấm **Tạm dừng** lúc cần dùng máy; hôm sau bấm **Tiếp tục** là chạy tiếp đúng cảnh đang dở.

RAM: bước tổng hợp cần nhiều RAM hơn máy chính, vì GTX 1060 chỉ chứa được ít phần model trên GPU. Ước tính **khoảng 16GB trống** trước khi bắt đầu bước này. Hãy đóng trình duyệt nhiều tab, Blender, WSL/Docker (`vmmem`)… App kiểm tra trước và báo rõ còn thiếu bao nhiêu.

---

## 10. Giới hạn đã biết trên PC GTX 1060

- **Tab 1** (phân tích một ảnh/video) vẫn dùng Qwen2.5-VL-7B **AWQ**, nên **không chạy** trên GTX 1060. Chỉ tab 3 có backend GGUF. Muốn tab 1 chạy trên máy yếu thì cần làm thêm.
- **Tab 2** chạy được với `RERANKER_MODEL_ID=Qwen/Qwen3-Reranker-0.6B` (xem mục 5), nhưng chậm hơn máy chính vì Pascal tính fp16 yếu.
- **Model thị giác nhận ảnh rời, không nhận video:** mô tả chuyển động kém hơn một chút so với bản AWQ trên máy chính. Nhãn `[giây X]` trước mỗi ảnh giúp model giữ đúng thứ tự thời gian.

---

## 11. Gặp lỗi thường gặp

| Hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `llama-server thoát ngay khi nạp model` | Sai bản llama.cpp (vd: bản CUDA 13, hoặc bản cũ không có `--fit` / `--load-mode`) | Dùng đúng b11120 **CUDA 12.4**; xem dòng cuối `llama-server*.log` |
| `no kernel image is available` / `CUDA error` | Bản torch/llama.cpp build cho CUDA 12.8+/13 (đã bỏ Pascal) | Cài lại torch cu121, llama.cpp CUDA 12.4 |
| Bước đứng ở "Chưa đủ tài nguyên" | RAM/VRAM trống không đủ | Đóng bớt ứng dụng; app tự chạy tiếp khi đủ |
| "Bước này liên tục làm máy thiếu RAM" | Máy bị căng 2 lần liên tiếp trong cùng bước | Đóng ứng dụng nặng rồi bấm **Tiếp tục** |
| `Chưa có Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | Chưa tải model mục 2–3 | Làm lại mục 5 (hoặc đặt `VL_GGUF_DIR`) |
| Hồ sơ nhận là `high` | GPU khác / đang dùng card khác | `set FILM_PROFILE=low` trước khi chạy |
