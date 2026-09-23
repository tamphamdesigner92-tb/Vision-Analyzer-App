# Phân tích phim dài (tab 3)

Hiểu nội dung cả một bộ phim 2–3 giờ: đoạn nào nói về gì, chuyện gì diễn ra, ai là ai, và hỏi đáp
về phim. Mỗi phim là một **dự án** nằm gọn trong một thư mục.

## Đường ống

```
phim ─► [1] Chuẩn bị ─► [2] Bóc băng ─► [3] Cắt cảnh ─► [4] Mô tả cảnh ─► [5] Tổng hợp
         ffprobe,        faster-whisper   PySceneDetect   Qwen2.5-VL-7B     Qwen3-30B-A3B
         tìm phụ đề      large-v3         -> cảnh 20–90s  + lời thoại       (llama.cpp)
                         (bỏ qua nếu                                         đoạn -> chương -> phim
                          có .srt)                                           + sổ nhân vật + chỉ mục
```

- Mỗi bước chạy trong **một tiến trình con riêng** (mức ưu tiên thấp): xong bước là RAM/VRAM được trả sạch.
- **Checkpoint** theo từng cảnh / từng chương: tạm dừng, tắt máy, sập điện rồi bấm *Tiếp tục* là chạy tiếp đúng chỗ.
- Có file `.srt` cùng tên phim (hoặc phụ đề chữ nhúng trong phim) thì **không bóc băng**.
- Trước mỗi bước kiểm tra RAM/VRAM; trong lúc chạy canh pagefile. Máy căng thì **tạm dừng sau cảnh đang làm**, không huỷ.
- Đóng tab trình duyệt khi đang chạy phim: app **chạy tiếp ở nền**, xong việc mới tự tắt.

## Dự án

- *Tạo dự án mới* → nằm trong thư mục mặc định (`projects/`, đổi được).
- *Lưu dự án…* → **chuyển toàn bộ dữ liệu** sang thư mục bạn chọn. Đang chạy cũng lưu được: tự tạm dừng, chuyển, chạy tiếp ở chỗ mới.
  Khác ổ đĩa: chép – kiểm tra sha256 – rồi mới xoá chỗ cũ; lỗi giữa chừng thì dữ liệu cũ còn nguyên.
- Mọi đường dẫn trong dự án là tương đối: chép cả thư mục sang máy khác vẫn *Mở dự án…* được.

```
<Thư mục dự án>/
  project.json        00_nguon/  01_loi_thoai/  02_phan_canh/
  03_canh/scene_0001.json   (+ .r2.json khi phân tích lại — bản cũ không bao giờ bị xoá)
  04_keyframes/  05_doan/  06_chuong/  07_phim/ (film.json, film_report.md, characters.json)
  08_hoi_dap/  09_chi_muc/index.sqlite  10_nhat_ky/  11_xuat/ (cho tab 2, .srt, báo cáo)
```

## Cài đặt thêm (đã làm trên máy chính)

```
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

- `tools/llama.cpp/bin/` — llama.cpp bản **CUDA 12.x** cho Windows (b11120). Không dùng bản CUDA 13 (bỏ hỗ trợ GTX 10xx).
- Qwen3-30B-A3B-Instruct-2507 Q4_K_M (`unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF`, 18.6 GB) trong cache Hugging Face.
- Whisper large-v3 cho faster-whisper: chuyển từ file openai-whisper có sẵn, không tải lại 3 GB:
  ```
  .venv\Scripts\python.exe tools\convert_whisper_pt.py --pt %USERPROFILE%\.cache\whisper\large-v3.pt --out models\faster-whisper-large-v3
  ```

## Số đo trên máy chính (RTX 4070 Ti SUPER 16 GB, 32 GB RAM)

| Bước | Tài nguyên | Tốc độ |
|---|---|---|
| Bóc băng (fp16) | ~4.6 GB VRAM | ~130× thời gian thực |
| Mô tả cảnh | ~7 GB VRAM + ngân sách khung hình | 45–55 s / cảnh 60 s (50 khung, nhìn nhiều lượt khi VRAM chật) |
| Tổng hợp / hỏi đáp | ~8 GB RAM + GPU còn trống (--fit) | đọc 500–700 token/s, sinh ~50 token/s |

Phim 3 giờ (~200 cảnh): mô tả cảnh khoảng **2.5–3 giờ**, tổng hợp khoảng **20–40 phút**.
Mức VRAM/RAM phụ thuộc ứng dụng khác đang mở (Blender, trình duyệt, WSL…): app tự chờ và báo rõ khi thiếu.

## Máy cấu hình thấp (GTX 1060 6GB)

Hồ sơ `low` dùng model GGUF qua llama.cpp thay cho AWQ — xem [TRIEN_KHAI_GTX1060.md](TRIEN_KHAI_GTX1060.md)
(danh sách model cần tải, cách cài, cách kiểm chứng, giới hạn đã biết).

## Tài liệu khác

- [KE_HOACH_PHIM_DAI.md](KE_HOACH_PHIM_DAI.md) — kế hoạch đã duyệt (M0–M6).
- [KHOP_KICH_BAN.md](KHOP_KICH_BAN.md) — tab 2, khớp kịch bản.

## Dòng lệnh (không cần giao diện)

```
.venv\Scripts\python.exe -m film.cli create "D:\Phim\phim.mkv" --name "Tên phim"
.venv\Scripts\python.exe -m film.cli run <id>
.venv\Scripts\python.exe -m film.cli ask <id> "Câu hỏi 1" "Câu hỏi 2"
.venv\Scripts\python.exe -m film.cli search <id> "canh sat"
.venv\Scripts\python.exe -m film.cli save <id> "D:\PhimDuAn"
```
