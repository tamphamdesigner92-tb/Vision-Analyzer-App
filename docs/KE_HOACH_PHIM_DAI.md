# Kế hoạch: Phân tích phim dài 2–3 giờ trên nhánh `main` (Windows)

## Bối cảnh

Hiện tại app phân tích từng video ngắn: Qwen2.5-VL-7B-AWQ mô tả hình, Qwen3-Reranker khớp kịch bản. Mục tiêu mới là hiểu **cả một bộ phim 2–3 giờ**, trả lời được "đoạn nào đang nói về gì, chuyện gì đang diễn ra". Phải có bóc băng lời thoại (Whisper), không được làm nghẽn RAM hay giật máy, và chất lượng được ưu tiên hơn tốc độ.

**Máy (đã đo):**
- RTX 4070 Ti SUPER 16GB VRAM
- 32GB DDR5-5600
- i7-14700F (20 nhân / 28 luồng)
- ổ C còn trống **82GB**
- đã có ffmpeg, torch 2.5.1+cu121 (cuDNN 9.1), transformers 4.51.3 (ghim cho AutoAWQ), psutil 7.2.2

**Máy thứ hai cần hỗ trợ (cấu hình thấp):** GTX 1060 6GB, Ryzen 7 2700X, 32GB DDR4-3200, mainboard X470 (PCIe 3.0). Xem mục "Cấu hình 2".

**Bạn đã chọn:**
- Phim nhiều thứ tiếng
- Giữ Qwen2.5-VL-7B
- "Bộ não" suy luận: Qwen3-30B-A3B chia RAM + VRAM
- Job phim dài **chạy tiếp ở nền** khi đóng tab

## Trả lời 3 câu hỏi trước khi vào kế hoạch

**1. Có model 1M ngữ cảnh chạy trên máy này không? Không chạy được, và cũng không nên.**
- Có model hỗ trợ 1M (Qwen2.5-7B/14B-Instruct-1M, Qwen3-2507 mở rộng tới 1M). Nhưng tài liệu của Qwen ghi bản 7B cần khoảng **120GB VRAM** để chạy 1M token.
- Lý do là bộ nhớ đệm KV tăng theo độ dài ngữ cảnh: dù đã nén 8-bit, 1M token vẫn tốn cỡ 50–70GB. Máy có 16GB VRAM + 32GB RAM.
- Riêng phần hình ảnh của một phim 3 giờ, lấy 1 khung/giây cũng đã hơn 1M token.
- Model nhét ngữ cảnh cực dài còn hay "quên đoạn giữa".
- → Cách làm đúng là **tóm tắt theo tầng** (cảnh → chương → cả phim) và khi hỏi thì **tra đúng đoạn liên quan**. Toàn bộ văn bản của một phim 3 giờ chỉ khoảng 100K token (lời thoại + mô tả cảnh), nên cách này vừa khả thi vừa cho kết quả tốt hơn.

**2. Nạp model vào RAM, san sẻ với VRAM được không? Được.**
- Dùng **llama.cpp** với model MoE: Qwen3-30B-A3B mỗi token chỉ kích hoạt 3B tham số.
- Cờ `--n-cpu-moe` để các "expert" nằm trong RAM (khoảng 14–16GB), còn attention + KV cache nằm trên GPU (khoảng 6–9GB).
- Tốc độ ước tính 15–25 token/giây, chất lượng cao hơn hẳn model 14B chạy hết trên GPU.
- Model dense (không phải MoE) cũng offload được qua `-ngl`, nhưng chậm hơn nhiều.

**3. Không nghẽn RAM, không giật máy — cách đảm bảo:**
- **Mỗi lúc chỉ một model nặng:** mỗi bước chạy trong **một tiến trình con riêng**. Tiến trình thoát là trả sạch VRAM/RAM, không bị phân mảnh bộ nhớ qua nhiều giờ chạy.
- **Kiểm tra trước khi nạp:** mỗi bước kiểm tra RAM/VRAM trống trước khi nạp model.
- **Canh pagefile:** nếu pagefile (swap của Windows) tăng hoặc RAM trống xuống thấp thì **tạm dừng giữa hai cảnh** và báo lên giao diện.
- **Nhường máy:** tiến trình con chạy ở mức ưu tiên thấp (BELOW_NORMAL) và giới hạn số luồng CPU (khoảng 8/28), để máy vẫn mượt khi bạn làm việc khác.

## Kiến trúc: 5 bước, mỗi bước là một tiến trình con, lưu checkpoint từng đơn vị

```
Phim (media_input/) ─► [1] Chuẩn bị ─► [2] Bóc băng ─► [3] Cắt cảnh ─► [4] Mô tả cảnh ─► [5] Tổng hợp
                        ffprobe        faster-whisper   PySceneDetect   Qwen2.5-VL-7B     Qwen3-30B-A3B
                        tách audio     large-v3         gộp thành cảnh  + lời thoại        (llama.cpp)
                        16kHz mono     (GPU ~4.5GB)     20–90 giây      của cảnh đó        chương → phim
                                                                         (GPU ~7GB+)        (GPU ~8GB, RAM ~16GB)
```

### Dự án: mỗi phim là một dự án, lưu trong thư mục do bạn chọn

**Vòng đời của dự án (giống lưu tài liệu trong Word):**

1. **Tạo dự án mới:** chỉ cần chọn file phim và đặt tên. App tạo ngay thư mục làm việc trong **thư mục mặc định của dự án**:
   - Mặc định là `<thư mục app>/projects/<tên dự án>/` (đã có trong `.gitignore`).
   - Đổi được trong phần Cài đặt (`projects_root` trong `settings.json` của app), ví dụ sang ổ D cho rộng.
   - Mọi bước chạy ghi thẳng vào đây từ phút đầu.
   - Giao diện gắn nhãn **"Chưa lưu — đang ở thư mục mặc định"**.
2. **Lưu dự án (lần đầu) / Lưu thành… (sang chỗ khác):**
   - Bạn chọn một thư mục bất kỳ trên máy (ổ khác cũng được) qua **hộp thoại chọn thư mục của Windows**. Trình duyệt không lấy được đường dẫn thật trên đĩa, nên server bật hộp thoại gốc bằng `tkinter.filedialog`, chạy trong một tiến trình con. Có thêm ô dán đường dẫn bằng tay để dự phòng.
   - App **chuyển toàn bộ dữ liệu** sang đó. Từ lúc ấy mọi bước chạy tiếp ghi vào thư mục mới.
3. **Mở dự án cũ:** chọn một thư mục có `project.json`.

**Cách chuyển dữ liệu an toàn (`project.move_to()`):**
- **Chọn thư mục đích:** thư mục trống thì dùng luôn; nếu đã có file thì tạo thư mục con `<tên dự án>/` bên trong, không bao giờ trộn vào dữ liệu có sẵn.
- **Kiểm tra trước:** ghi được, không nằm trong thư mục app / `.venv`, không nằm bên trong chính dự án, **đủ dung lượng trống**.
  - Ước tính: phim 3 giờ cần khoảng 1–2GB cho keyframe, JSON, chỉ mục, cộng khoảng 350MB audio nếu phải bóc băng.
  - Nếu chọn "Chép phim vào dự án" thì cộng thêm dung lượng file phim.
- **Đang chạy cũng lưu được:**
  - App tự **tạm dừng sau cảnh hiện tại**. Tiến trình con thoát nên không còn file nào bị mở.
  - Server đóng kết nối `index.sqlite` và các file log.
  - Chuyển xong thì **tự chạy tiếp từ checkpoint ở chỗ mới**.
  - Làm được vì mọi đường dẫn trong dự án đều tương đối.
- **Cùng ổ đĩa:** đổi tên thư mục (`os.replace`), xong gần như tức thì và nguyên vẹn một khối.
- **Khác ổ đĩa:**
  - Chép từng file kèm thanh tiến độ, **kiểm tra lại** (số file, kích thước, mã băm) ở đích.
  - **Chỉ khi khớp hết** mới xoá thư mục ở chỗ cũ.
  - Lỗi giữa chừng (đầy ổ, rút ổ cứng ngoài, file bị chương trình khác giữ…) thì dữ liệu cũ **giữ nguyên**, bản chép dở ở đích bị dọn đi, và giao diện báo lý do.
- **Sau khi chuyển:** `projects.json` cập nhật đường dẫn mới; `project.json` ghi lịch sử các nơi đã lưu.
- **Danh sách dự án gần đây:** app chỉ lưu **đường dẫn tới các dự án** trong `projects.json` ở thư mục app, không lưu dữ liệu nào ở đó. Xoá file này cũng không mất dự án.
- **Mang đi được:** mọi đường dẫn bên trong dự án đều là **đường dẫn tương đối**, nên chép hoặc chuyển cả thư mục sang ổ khác hay sang PC khác (ví dụ PC GTX 1060) vẫn mở và chạy tiếp được.
- **File phim nguồn:**
  - Mặc định chỉ ghi lại đường dẫn, kích thước, thời điểm sửa và mã băm nhanh, để không nhân đôi hàng chục GB.
  - Có tuỳ chọn **"Chép phim vào dự án"** để dự án tự chứa đủ.
  - Phim bị chuyển chỗ thì app phát hiện (sai mã băm hoặc mất file) và cho bạn chọn lại đường dẫn.
- **Khoá dự án:** `project.lock` ngăn hai lần chạy cùng lúc trên một dự án.
- **Không ghi ra ngoài:** mọi file tạm, log và file ghi an toàn đều nằm trong dự án, không dùng `%TEMP%` hay `vision_storage/`.

**Cấu trúc thư mục dự án:**

```
<Thư mục dự án>/
  project.json                  tên, phim nguồn, hồ sơ phần cứng, thiết lập, trạng thái từng bước, lịch sử các lần chạy
  project.lock                  (chỉ có khi đang chạy)
  00_nguon/                     phim (nếu chọn chép vào) + phụ đề gốc bạn đưa vào
  01_loi_thoai/                 phụ đề đã chuẩn hoá (subtitles.json + .srt) hoặc transcript Whisper; audio.wav nếu có bóc băng
  02_phan_canh/                 shots.json (ranh giới shot), scene_plan.json (ranh giới cảnh 20–90 giây)
  03_canh/scene_0001.json       TẦNG 1 — cảnh nhỏ, MỘT FILE MỖI CẢNH, không bao giờ xoá
  03_canh/scene_0001.r2.json    phân tích lại (đổi model/prompt) thì ghi bản r2, r3…, KHÔNG ghi đè bản cũ
  04_keyframes/scene_0001_*.jpg
  05_doan/seq_001.json          TẦNG 2 — đoạn (~3–5 phút, gộp vài cảnh): tóm tắt + id các cảnh con
  06_chuong/ch_01.json          TẦNG 3 — chương (~10–15 phút, gộp vài đoạn): tóm tắt + id các đoạn con
  07_phim/                      TẦNG 4 — film.json, film_report.md, characters.json (sổ nhân vật)
  08_hoi_dap/qa_<thời gian>.json  mỗi câu hỏi đã hỏi: câu hỏi, câu trả lời, các id cảnh được trích
  09_chi_muc/index.sqlite       chỉ mục tra cứu (dựng lại được từ các file JSON)
  10_nhat_ky/                   pipeline.log, resource.log, log từng bước (stdout tiến trình con), llama-server.log
  11_xuat/                      file xuất: <tên>.json cho tab 2, .srt, .md, gói .zip
  tmp/                          file tạm của lần chạy đang dở (dọn khi bước xong)
```

### Lưu trữ theo tầng: mỗi cảnh nhỏ một file JSON vĩnh viễn, tầng lớn tóm tắt và đánh chỉ mục

Các file JSON là **nguồn gốc duy nhất**. Chỉ mục SQLite chỉ là bản dẫn xuất từ JSON, xoá đi dựng lại được.

**File của mỗi cảnh nhỏ (`scene_XXXX.json`) lưu đầy đủ:**
- id, cha (`seq_*` / `ch_*`), bắt đầu/kết thúc (giây + `hh:mm:ss.mmm`), các shot bên trong
- mốc thời gian của từng khung hình đã lấy mẫu, đường dẫn keyframe
- **toàn bộ lời thoại** trong khoảng đó: từng dòng kèm mốc, ngôn ngữ, nguồn (`srt` / `whisper`)
- **đầu ra thô của model thị giác** và các trường đã tách:
  - mô tả chi tiết, nhân vật, hành động, bối cảnh
  - chữ trên màn hình, âm thanh / không khí, ý chính lời thoại
- danh sách **sự kiện nhỏ** trong cảnh, mỗi sự kiện có id riêng (`scene_0001.e03`) và mốc giây
- thông tin để truy vết: model, backend, hồ sơ phần cứng, phiên bản prompt, thời gian tạo

**Cách ghi:**
- Ghi an toàn: ghi ra file tạm rồi đổi tên, nên tắt máy giữa chừng không để lại file hỏng.
- Không có đoạn code nào xoá các file này.
- Chạy lại chỉ bỏ qua cảnh đã có file hợp lệ, và đó cũng là checkpoint của bước 4.

**Tầng lớn tóm tắt từ tầng nhỏ (đoạn → chương → phim):**
- Mỗi mục trong bản tóm tắt (sự kiện, xuất hiện của nhân vật, địa điểm, bước ngoặt) có **id riêng**.
- Mỗi mục trỏ về id các cảnh con kèm mốc thời gian. Từ bất kỳ dòng tóm tắt nào cũng lần ngược được về đúng cảnh và đúng câu thoại gốc.

**Chỉ mục `index.sqlite`:**
- **Bảng:** `nodes` (mọi cảnh / đoạn / chương, cha–con, mốc, tóm tắt, đường dẫn JSON), `events`, `dialogue` (từng dòng thoại), `characters` + `appearances`.
- **Tìm kiếm toàn văn:** FTS5 với bộ tách từ `unicode61 remove_diacritics`, nên gõ tiếng Việt không dấu vẫn ra kết quả.
- **API truy xuất bất cứ lúc nào** (không cần chạy model):
  - `GET /api/project/<id>/node/<node_id>`: trả JSON đầy đủ của một cảnh, đoạn hay chương
  - `GET /api/project/<id>/search?q=`: tìm trong lời thoại, mô tả và sự kiện
  - `GET /api/project/<id>/tree`: cây chương → đoạn → cảnh
  - `<id>` là id trong `projects.json`. Server chỉ trả file nằm trong thư mục của dự án đã đăng ký, kiểm tra đường dẫn giống `_safe_media_path` / `_safe_result_dir` hiện có.

Ngoài ra ghi `11_xuat/<tên>.json` theo **đúng khuôn `segments`/`images` hiện có**, dẫn xuất từ các file cảnh. `script_matcher.collect_candidates` (`script_matcher.py:110`) được sửa nhẹ để nhận thêm danh sách thư mục: ngoài `vision_storage/` còn quét `11_xuat/` của các dự án đã đăng ký. Nhờ vậy tab 2 (khớp kịch bản) thấy các cảnh của phim làm ứng viên.

**Phạm vi:** dự án áp dụng cho tab 3 (phim dài). Tab 1 và tab 2 vẫn lưu ở `vision_storage/` như hiện tại. Nếu muốn đưa cả hai tab đó vào dự án thì làm ở một mốc sau.

**Tạm dừng, sập máy và chạy lại đi chung một đường:** tiến trình con bị dừng hay chết thì lần sau chỉ làm tiếp các đơn vị chưa có file.

### Chi tiết từng bước

**[1] Chuẩn bị**
- ffprobe lấy thời lượng, fps, các luồng audio/phụ đề.
- **Tìm phụ đề có sẵn**, theo thứ tự:
  1. file `.srt` cùng tên phim (kể cả `tên.vi.srt`, `tên.en.srt`…)
  2. phụ đề nhúng trong file phim: ffmpeg tách ra `.srt`
- Đọc phụ đề với tự nhận mã hoá: UTF-8 / UTF-8-BOM / UTF-16 / cp1258. Bỏ thẻ định dạng `<i>`, `{\an8}`. Kiểm tra mốc thời gian không vượt quá độ dài phim.
- Có phụ đề → **bỏ qua hẳn bước [2] và không tách audio**. `project.json` ghi "bỏ qua — dùng phụ đề <tên file>", và phụ đề gốc được chép vào `00_nguon/`. Giao diện có ô "Vẫn bóc băng bằng Whisper" nếu bạn nghi phụ đề lệch hoặc thiếu.
- Không có phụ đề → ffmpeg tách audio ra WAV 16kHz mono (khoảng 350MB cho 3 giờ) cho bước [2].

**[2] Bóc băng (chỉ chạy khi không có phụ đề)**
- faster-whisper `large-v3`, không dùng bản turbo vì bản turbo kém hơn với tiếng không phải tiếng Anh. Chạy CUDA fp16, dùng BatchedInferencePipeline.
- Thiết lập: `vad_filter=True`, `word_timestamps=True`, `condition_on_previous_text=False` để tránh lặp câu.
- Bỏ các đoạn có `compression_ratio > 2.4` hoặc `no_speech_prob` cao (bịa chữ).
- **Nhiều thứ tiếng:**
  - Nhận diện ngôn ngữ trên nhiều đoạn rải khắp phim, không chỉ 30 giây đầu (vì hay dính nhạc mở đầu).
  - Có ô ép ngôn ngữ trên giao diện.
  - Lời thoại giữ nguyên ngôn ngữ gốc; báo cáo viết bằng tiếng Việt.
- Ước tính 10–20 phút cho phim 3 giờ.

**[3] Cắt cảnh**
- PySceneDetect `AdaptiveDetector` trên luồng video thu nhỏ (khoảng 15–25 phút cho 3 giờ), ra danh sách shot.
- Gộp các shot thành **cảnh 20–90 giây**. Nếu phải cắt một đoạn dài thì dời điểm cắt vào khoảng lặng gần nhất của lời thoại, để không cắt ngang câu.

**[4] Mô tả cảnh (bước lâu nhất)**
- Dùng lại `LocalVisionAnalyzer` (`vision_analyzer_app.py`). Hàm này đã có sẵn các cơ chế:
  - truyền đúng FPS
  - chia đoạn theo ngân sách VRAM (`_vram_patch_budget`)
  - `timeline_instruction` / `merge_timeline` / `clamp_timeline` để ghi mốc thời gian
- **Đọc khung hình kiểu tua tới, không giải mã cả phim:** chuyển `sample_video_frames` / `_frame_at` (đọc bằng PyAV có tua) từ nhánh `Vision-on-Mac` (`vision_analyzer_app.py:215-239` bên đó) sang `main`. Cách đọc hiện tại của `qwen_vl_utils` có thể giải mã lại từ đầu phim cho mỗi cảnh.
- **Prompt mỗi cảnh gồm:** khung hình (mặc định 1 khung/giây, tối đa khoảng 48 khung/cảnh), **lời thoại của đúng khoảng thời gian đó**, và tóm tắt 2 cảnh trước để giữ mạch.
- **Model trả về thẻ cảnh dạng JSON:** mô tả, nhân vật (mô tả ngoại hình), hành động, bối cảnh, chữ trên màn hình, không khí, ý chính lời thoại. Kèm một keyframe mỗi cảnh.
- Ước tính **3–6 giờ** cho phim 3 giờ (khoảng 360 cảnh). Con số này sẽ đo thật ở mốc M2.

**[5] Tổng hợp và suy luận (llama.cpp `llama-server`, Qwen3-30B-A3B-Instruct-2507 Q4_K_M)**
- Cờ chạy: `-ngl 99 --n-cpu-moe <tự chỉnh theo VRAM> -c 65536 -ctk q8_0 -ctv q8_0 -fa on -t 8`.
- Gọi qua API tương thích OpenAI, có `response_format` JSON schema để đầu ra luôn đúng khuôn.
- **Đoạn (tầng 2):** gộp 3–6 cảnh liên tiếp (thẻ cảnh + lời thoại đầy đủ) → một file `seq_*.json` gồm tóm tắt, sự kiện (mỗi sự kiện có id, trỏ về cảnh gốc), nhân vật, địa điểm.
- **Chương (tầng 3):** gộp các đoạn trong khoảng 10–15 phút, cộng "câu chuyện tới giờ" (≤1.5K token) → một file `ch_*.json` gồm tiêu đề, tóm tắt, bước ngoặt, trỏ về các đoạn con. Phim 3 giờ ra khoảng 12–18 chương.
- **Sổ nhân vật:** cập nhật dần qua từng chương. LLM nối "người đàn ông áo xanh" (hình) với tên nhân vật được gọi trong lời thoại, để danh tính nhất quán suốt phim.
- **Cả phim:** tất cả chương (khoảng 12K token) + sổ nhân vật → tóm tắt, cấu trúc các hồi, tuyến nhân vật, chủ đề, và mục lục thời gian.
- **Hỏi đáp (không cần model embedding):**
  - Bước 1: LLM sinh từ khoá cho câu hỏi, cả tiếng Việt lẫn ngôn ngữ gốc của phim. Tra `index.sqlite` bằng FTS5, đồng thời LLM đọc tóm tắt các chương để chọn chương liên quan.
  - Bước 2: đi xuống cây (chương → đoạn → cảnh), nạp đầy đủ file cảnh + lời thoại của các nhánh trúng (≤40K token) → trả lời kèm mốc `[hh:mm:ss]` và id cảnh để bấm mở.
- Ước tính 20–40 phút cho phần tổng hợp.

**Tổng thời gian: khoảng 4–7 giờ cho phim 3 giờ, chạy qua đêm được.**

## Cấu hình 2: PC thấp — GTX 1060 6GB, Ryzen 7 2700X, 32GB DDR4-3200

Chạy được, nhưng **không dùng lại nguyên pipeline của máy chính**. Có 3 giới hạn cứng của card Pascal (GTX 10xx):

1. **AutoAWQ không chạy:** kernel AWQ đòi GPU từ đời Turing (sm 7.5) trở lên, GTX 1060 là sm 6.1. Qwen2.5-VL-7B-AWQ còn cần khoảng 7GB, lớn hơn 6GB. Reranker-4B fp16 (khoảng 8GB) cũng không vừa.
2. **Tính toán fp16 cực chậm trên GTX 10xx** (khoảng 1/64 tốc độ fp32), và không có FlashAttention 2. Vì vậy phải chạy **int8 / lượng tử GGUF**, không chạy fp16/bf16.
3. **PyTorch, CUDA và driver mới đang bỏ dần Pascal.** Cần ghim torch bản CUDA 12.1–12.6 (bản `main` đang dùng 2.5.1+cu121 là được) và dùng llama.cpp bản **CUDA 12.x**, không dùng bản CUDA 13.

**Giải pháp: thêm "hồ sơ phần cứng" để pipeline tự chọn backend** (file mới `hardware_profile.py`):
- Tự nhận diện qua `torch.cuda.get_device_capability()`, VRAM tổng và RAM.
- GPU sm < 7.5 hoặc VRAM < 8GB → hồ sơ `low`, còn lại → `high`.
- Có thể ép hồ sơ bằng biến môi trường `FILM_PROFILE`.

| Bước | Máy chính (`high`) — 4070 Ti S 16GB | PC thấp (`low`) — GTX 1060 6GB |
|---|---|---|
| Bóc băng | faster-whisper large-v3, CUDA **fp16** | faster-whisper large-v3, CUDA **int8** (Pascal chạy tốt int8 nhờ lệnh dp4a), khoảng 2GB VRAM |
| Mô tả cảnh | Qwen2.5-VL-7B-AWQ qua transformers, đưa vào dạng video | Qwen2.5-VL-7B **GGUF Q4_K_M + mmproj** qua **llama.cpp**, offload một phần layer sang RAM (`-ngl` tự chỉnh). Khung hình đưa vào dạng **nhiều ảnh** (llama.cpp chưa nhận video) |
| Thiết lập cảnh | 1 khung/giây, ≤48 khung/cảnh, cảnh 20–90 giây | **0.5 khung/giây, ≤12 khung/cảnh, ảnh cạnh dài ~448px, cảnh 30–120 giây** (ít cảnh hơn, khoảng 150–200 cảnh cho phim 3 giờ) |
| Tổng hợp + hỏi đáp | Qwen3-30B-A3B Q4_K_M, `-c 65536` | **Vẫn Qwen3-30B-A3B** — model MoE hợp nhất với máy ít VRAM nhiều RAM: phần chung + KV trên GPU (khoảng 4–5GB), expert trong RAM (khoảng 17GB). `-c 32768`, hỏi đáp nạp ≤20K token |
| Luồng CPU | `-t 8` / torch 8 luồng (trên 28) | `-t 6` / torch 4 luồng (trên 16), chừa lại cho Windows |
| Tốc độ LLM ước tính | 15–25 token/giây | 8–15 token/giây (DDR4 dual-channel khoảng 45GB/s là nút thắt) |
| **Tổng thời gian phim 3 giờ** | **4–7 giờ** | **khoảng 8–15 giờ → 1–2 đêm**, nhờ tạm dừng / chạy tiếp theo checkpoint |

Lời thoại (Whisper) gánh nhiều phần "hiểu nội dung" hơn trên máy thấp, vì mô tả hình thưa hơn. Thứ tự các bước giữ nguyên.

**Riêng cho PC thấp:**
- **Chống giật:** GTX 1060 vừa tính toán vừa xuất hình ra màn hình. Nên:
  - luôn chừa khoảng 1GB VRAM cho Windows
  - dùng batch nhỏ cho llama.cpp (`-ub 256`) để GPU không bị một kernel chiếm lâu, desktop và video vẫn mượt
  - tạm dừng giữa các cảnh khi `resource_guard` báo căng
- **Tab 1 và tab 2 hiện có** cũng không chạy trên máy này. Tab 1 sẽ dùng backend llama.cpp ở trên. Tab 2 dùng `Qwen3-Reranker-0.6B` chạy fp32 (khoảng 2.4GB), đổi qua `RERANKER_MODEL_ID` có sẵn.
- **Tải thêm:** Qwen2.5-VL-7B GGUF Q4_K_M + mmproj (khoảng 6GB), Qwen3-Reranker-0.6B (khoảng 1.2GB), cộng các file chung (llama.cpp, Qwen3-30B-A3B, whisper large-v3). **Cần khoảng 30GB trống** trên PC thấp.

## Kiểm soát tài nguyên (file mới `resource_guard.py`, dùng psutil + `torch.cuda.mem_get_info`)

Làm theo mẫu `mac_memory.py` của nhánh Mac, nhưng cho Windows:
- **Kiểm tra trước khi nạp:** `need_ram` / `need_vram` của từng bước; nhu cầu được hiệu chỉnh bằng đỉnh đo thật, lưu vào `.ram_profile.json`. Thiếu thì job chờ và báo rõ đang thiếu bao nhiêu.
- **Canh trong lúc chạy:** mỗi 2 giây đo RAM trống và `psutil.swap_memory().used` (pagefile).
  - Pagefile tăng quá 512MB, hoặc RAM trống dưới 2GB, thì **tạm dừng sau cảnh hiện tại**. Chỉ tạm dừng, không huỷ. Chạy tiếp được khi máy thoáng lại.
- **Nhường máy:** `psutil.Process().nice(BELOW_NORMAL_PRIORITY_CLASS)`, `torch.set_num_threads(8)`, llama.cpp `-t 8`.
- **Ghi log:** lấy mẫu RAM/VRAM/pagefile ra `<dự án>/10_nhat_ky/resource.log` (giữ lại để bạn đọc).
- **Thanh trạng thái:** thêm chip "RAM trống / pagefile" cạnh chip VRAM có sẵn (`app_status.vram`).

## Các file sẽ thêm hoặc sửa

- **Thêm `film/`** (package mới):
  - `pipeline.py`: điều phối các bước, trạng thái trong `project.json`, checkpoint
  - `stage_runner.py`: chạy tiến trình con, đọc tiến độ dạng JSON-lines từ stdout, dừng/tạm dừng
  - `stage_prepare.py` (có tìm và đọc phụ đề), `stage_asr.py`, `stage_scenes.py`, `stage_cards.py`, `stage_synthesis.py`
  - `project.py`: tạo dự án trong thư mục mặc định, mở, `move_to()` (Lưu / Lưu thành…, tạm dừng rồi chạy tiếp), `project.json`, `projects.json`, khoá dự án, kiểm tra dung lượng trống, phát hiện phim nguồn bị chuyển chỗ, hộp thoại chọn thư mục/file (tkinter chạy trong tiến trình con)
  - `store.py`: ghi và đọc file theo tầng **bên trong thư mục dự án** (ghi an toàn, đánh phiên bản `.rN`, không bao giờ xoá, chỉ dùng đường dẫn tương đối), sinh id cho cảnh / đoạn / chương / sự kiện
  - `index.py`: dựng và cập nhật `index.sqlite` (FTS5) từ các file JSON, hàm tìm kiếm, hàm lấy cây
  - `subtitles.py`: đọc `.srt` với tự nhận mã hoá, tách phụ đề nhúng bằng ffmpeg
  - `llm_client.py`: bật/tắt `llama-server`, chọn cổng trống, gọi API
  - `qa.py`: hỏi đáp
- **Thêm `resource_guard.py`:** kiểm tra và canh RAM/VRAM/pagefile.
- **Thêm `hardware_profile.py`:** nhận diện hồ sơ `high` / `low` và trả về model, backend, cờ llama.cpp, FPS, kích thước cảnh, số luồng.
- **Thêm `film/vision_backends.py`:** một giao diện `describe_scene(frames, transcript, context)`, hai cách chạy:
  - `transformers_awq`: bọc `LocalVisionAnalyzer`
  - `llamacpp_gguf`: gửi nhiều ảnh qua `llama-server`, dùng lại `llm_client.py`
- **Sửa [web_app.py](web_app.py):**
  - thêm loại job `film` chạy trong worker có sẵn
  - API dự án: `/api/project/pick-folder|pick-file|create|open|list`, `/api/project/<id>/save-as` (chuyển dữ liệu, có tiến độ), `/api/settings` (`projects_root`)
  - API chạy: `/api/project/<id>/start|pause|resume|stop|status|report|ask`, cộng `node/<node_id>|search|tree`
  - `_browser_watchdog` **không tắt app khi đang có job phim**; phim xong mà không còn tab nào thì mới tắt
  - chip RAM/pagefile trong `/api/status`
- **Sửa [vision_analyzer_app.py](vision_analyzer_app.py):** thêm hàm đọc khung hình kiểu tua tới (lấy từ nhánh Mac) và một hàm mô tả một cảnh từ danh sách khung hình + lời thoại. Tái dùng `_generate` / `timeline_instruction`.
- **Sửa [script_matcher.py](script_matcher.py):** `collect_candidates` nhận thêm danh sách thư mục (`11_xuat/` của các dự án).
- **Sửa [static/index.html](static/index.html):** thêm **tab 3 "Phim dài"**
  - khối Dự án: Tạo mới (chọn phim + đặt tên) / Mở dự án / **Lưu dự án** / **Lưu thành…** / danh sách gần đây
  - hiện đường dẫn thư mục dự án, dung lượng trống, và nhãn "Chưa lưu — đang ở thư mục mặc định"
  - chọn ngôn ngữ, FPS mô tả
  - 5 bước kèm tiến độ và thời gian còn lại dự kiến; nút Tạm dừng / Tiếp tục / Dừng
  - trục thời gian chương → cảnh, bấm vào thì tua `<video>` tới đúng giây
  - lời thoại theo cảnh, ô hỏi đáp, tải `.md` / `.json` / `.srt`
- **Sửa [requirements.txt](requirements.txt):** thêm `faster-whisper`, `scenedetect`.
  - Chạy `pip install --dry-run` trước để chắc không kéo lệch `numpy<2` / `tokenizers` / `transformers` đang ghim.
  - Nếu lệch thì bóc băng chạy ở venv riêng `.venv-asr`. Không sao, vì nó vốn chạy ở tiến trình con.
- **Thêm vào `.gitignore`:** `tools/llama.cpp/`, `.ram_profile.json`, `projects/`, `projects.json`, `settings.json`.

## Tải về (sẽ hỏi bạn trước từng mục khi triển khai)

| Thứ cần tải | Dung lượng | Nguồn |
|---|---|---|
| llama.cpp bản CUDA 12 cho Windows | ~0.5GB | GitHub releases của ggml-org/llama.cpp |
| Qwen3-30B-A3B-Instruct-2507 GGUF Q4_K_M | ~18.6GB | Hugging Face (tên repo và file chính xác kiểm lại lúc tải) |
| faster-whisper-large-v3 | ~3GB | Hugging Face `Systran/faster-whisper-large-v3` |

Tổng khoảng 22GB trên 82GB còn trống. Cache HF hiện đã có 45GB, trong đó có model của dự án khác (NLLB, chatterbox); tôi không đụng vào.

## Các mốc triển khai (mỗi mốc kiểm chứng xong mới sang mốc sau)

- **M0 – Môi trường:** chạy dry-run cài thư viện, tải model (có hỏi bạn), chạy thử `llama-server` với `--n-cpu-moe`, đo VRAM/RAM/tốc độ token thật.
- **M1 – Khung pipeline + bước 1–3:**
  - dựng khung: `project.py` (dự án trong thư mục bạn chọn), `store.py` + `index.py` (lưu trữ theo tầng), tiến trình con, `resource_guard`
  - phụ đề có sẵn thì bỏ qua Whisper
  - chạy trên **một đoạn phim thử 10 phút**
  - kiểm tra `.srt` khớp tiếng, ranh giới cảnh hợp lý
- **M2 – Bước 4:**
  - thẻ cảnh cho đoạn 10 phút
  - đo số giây/cảnh để suy ra thời gian cho phim 3 giờ
  - so chất lượng giữa có và không có lời thoại trong prompt
- **M3 – Bước 5:** chương, sổ nhân vật, báo cáo cả phim, hỏi đáp trên đoạn 10 phút.
- **M4 – Tab 3:** giao diện + chạy nền + tạm dừng / tiếp tục / dừng.
- **M5 – Chạy trọn một phim 2–3 giờ qua đêm:** tinh chỉnh ngưỡng tài nguyên, FPS, kích thước cảnh.
- **M6 – Hồ sơ `low` trên PC GTX 1060:**
  - ép `FILM_PROFILE=low` trên máy chính, chạy đoạn 10 phút để kiểm tra đường llama.cpp-vision
  - sau đó chạy trên PC thật và đo tốc độ từng bước
  - so chất lượng thẻ cảnh giữa hai hồ sơ trên cùng đoạn phim
  - nếu backend llama.cpp cho chất lượng ngang bản AWQ thì có thể cân nhắc dùng chung cho cả hai máy

## Kiểm chứng

- **Đúng nội dung:** trên đoạn 10 phút, đối chiếu `.srt` với lời thoại thật (bấm ngẫu nhiên 10 mốc), đối chiếu thẻ cảnh với hình, và mốc thời gian trong chương/hỏi đáp phải tua tới đúng cảnh.
- **Dự án:**
  - tạo dự án mới → dữ liệu nằm trong `projects/<tên>/`, đủ các thư mục con `00`–`11`, nhãn "Chưa lưu"
  - **Lưu dự án sang ổ khác trong lúc đang chạy bước 4** (ví dụ `D:\PhimDuAn`):
    - job tự tạm dừng, chuyển xong thì chạy tiếp ở chỗ mới, không làm lại cảnh nào
    - thư mục cũ trong `projects/` không còn
    - số file, kích thước, mã băm khớp
    - sau đó **không có file nào** của dự án nằm trong `vision_storage/`, `%TEMP%` hay thư mục app, ngoài một dòng đường dẫn trong `projects.json`
  - **Lưu cùng ổ:** xong tức thì
  - **Lưu vào thư mục đã có file:** tạo thư mục con `<tên dự án>/`
  - **Giả lập lỗi giữa chừng** (ổ đích đầy, hoặc rút USB): dữ liệu cũ còn nguyên, bản chép dở bị dọn
  - chép tay cả thư mục dự án sang chỗ khác → mở lại được, chạy tiếp được
  - đổi chỗ file phim → app báo và cho chọn lại
  - chạy 2 lần cùng lúc trên một dự án → lần thứ hai bị chặn
  - thư mục thiếu dung lượng → bị từ chối kèm con số cần có
- **Phụ đề có sẵn:**
  - đặt `phim.srt` cạnh phim → `project.json` ghi bước [2] "bỏ qua", không có file WAV, lời thoại trong các file cảnh lấy từ `.srt`
  - thử lại với `.srt` mã hoá cp1258 và UTF-16
- **Lưu trữ theo tầng:**
  - mỗi cảnh có đúng một `scene_XXXX.json` đủ các trường
  - phân tích lại tạo ra `.r2.json` và bản cũ vẫn còn
  - xoá `index.sqlite` rồi dựng lại → ra cùng kết quả
  - tìm "canh sat" (không dấu) ra các cảnh có "cảnh sát"
  - từ một sự kiện trong `ch_*.json` lần ngược về đúng cảnh và câu thoại
- **Chạy lại được:**
  - bấm Tạm dừng giữa bước 4 rồi Tiếp tục → không chạy lại cảnh đã xong
  - tắt hẳn tiến trình (kill) giữa chừng → mở lại app, chạy tiếp từ checkpoint
- **Tài nguyên:**
  - trong cả lần chạy, `resource.log` không cho thấy pagefile tăng quá ngưỡng
  - thử mở trình duyệt / làm việc khác trong lúc chạy để kiểm tra máy không giật
  - VRAM trả về gần 0 giữa các bước (xem chip VRAM)
- **Chạy nền:** đóng tab trong lúc chạy → job vẫn chạy (xem log console); mở lại tab thấy đúng tiến độ.
- **Tích hợp:** tab 2 thấy các cảnh của phim làm ứng viên khớp kịch bản.
- **Toàn bộ:** M5 chạy trọn một phim thật, xem `film_report.md` và thử 5–10 câu hỏi về cốt truyện.
- **PC thấp:**
  - `hardware_profile` nhận đúng `low` trên GTX 1060
  - VRAM không vượt khoảng 5GB (chừa cho màn hình)
  - xem một video YouTube trong lúc pipeline chạy mà không giật
  - chạy qua 2 đêm bằng tạm dừng / tiếp tục, kết quả không mất cảnh nào
