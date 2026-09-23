# Khớp kịch bản — kiến trúc "bộ não" chọn và sắp xếp cảnh

Tab 2 của ứng dụng nhận một kịch bản và trả về **kế hoạch dựng**: mỗi dòng kịch bản được
gán một cảnh quay có sẵn, đã sắp đúng thứ tự kịch bản, kèm điểm tin cậy và phương án thay thế.

## Đường ống

```
media_input/*.mp4, *.jpg
        │
        │  ① Qwen2.5-VL-7B-Instruct-AWQ  (tab 1, đã có sẵn)
        ▼
vision_storage/<tên file>/*.json     mô tả bằng chữ + keyframe
        │
        ② Qwen3-Reranker-4B
           chấm kỹ từng cặp (dòng kịch bản, mô tả cảnh) → xác suất 0..1
        ▼
        ③ Xếp cảnh (greedy toàn cục) → kế hoạch dựng .md / .json
```

Qwen3-Reranker là mô hình CHỈ ĐỌC CHỮ (cross-encoder), không nhìn được ảnh/video trực tiếp
— vì vậy Qwen2.5-VL phải chạy trước ở tab 1 để biến mỗi cảnh quay thành một mô tả bằng chữ,
rồi reranker mới chấm được. Reranker đọc cả hai vế (dòng kịch bản + mô tả cảnh) cùng lúc nên
chính xác hơn cách so vector, đổi lại phải chấm lần lượt N×M cặp thay vì so vector một lần.

## Một venv duy nhất, chạy trên Apple Silicon

Cả hai model đều chạy qua `transformers`, trên MPS, trong cùng một `.venv`:

- Qwen2.5-VL-7B-Instruct-AWQ nạp qua `transformers` + `gptqmodel` (backend torch-native
  `TorchAtenAwqLinear`, không cần CUDA). Có một lỗi khớp đường dẫn module đã vá thủ công trong
  `vision_analyzer_app.py:_load_model()` — xem comment ở đó — nếu không vá, toàn bộ vùng nhận
  diện thị giác bị nạp trọng số ngẫu nhiên thay vì trọng số thật, model chạy được nhưng "mù".
  Dùng `dtype=torch.bfloat16` (không dùng `float16` — tràn số trên MPS với checkpoint này).
- Qwen3-Reranker (bản GPTQ int4 định dạng compressed-tensors) nạp qua `transformers` —
  weights được `compressed-tensors` tự giải nén khi cần, không cần CUDA.

## Chia chỗ trong bộ nhớ hợp nhất 16GB

Hai model cộng lại vượt quá những gì nên giữ cùng lúc trong 16GB RAM hợp nhất (còn phải chia
cho OS, trình duyệt, chính ứng dụng), nên mỗi lúc chỉ một model được nạp:

- Bấm "Phân tích" (tab 1) → tự giải phóng reranker, nạp Qwen2.5-VL
- Bấm "Khớp kịch bản" (tab 2) → tự giải phóng Qwen2.5-VL, nạp reranker

Mỗi lần đổi mất vài chục giây nạp lại, đổi lấy việc không bao giờ tràn bộ nhớ giữa chừng.

## Theo dõi: mọi thứ hiện trên giao diện web

Thanh trạng thái nằm ngay dưới thanh tiêu đề, **luôn hiện ở cả hai tab**, và cập nhật
mỗi 1.5 giây. Nó trả lời đúng bốn câu hỏi hay gặp:

| Câu hỏi | Chỗ trả lời |
|---|---|
| Đang chạy tác vụ gì? | Dòng đầu: "Phân tích ảnh/video: tên_file — 45%" hoặc "Khớp kịch bản — 60%" |
| Đang làm gì trong tác vụ đó? | Dòng phụ màu xám: thông điệp mới nhất (đang nạp model, đang chấm cảnh 2/4…) |
| Model nào đang nằm trong bộ nhớ? | Hai chip: `không nạp` / `đang tải trọng số` / `đang nạp` / `sẵn sàng` |
| Còn bao nhiêu bộ nhớ hợp nhất? | Chip bộ nhớ, chuyển đỏ kèm chữ "gần đầy" khi vượt 85% |

Lần đầu dùng mỗi mô hình tải từ HuggingFace (Qwen2.5-VL) sẽ có giai đoạn tải hàng GB từ
mạng, đo bằng cách so dung lượng thư mục cache với tổng dung lượng repo, nên thanh tiến độ
chạy thật chứ không đứng im. Reranker nạp từ path cục bộ trong AI Hub nên không có bước này.

## Cách chạy

```
./start_app.sh
```

Tự mở trình duyệt khi server sẵn sàng, tự tắt khi đóng tab trình duyệt.

Đổi model bằng biến môi trường, không cần sửa code:

```
export VL_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct-AWQ
export RERANKER_MODEL_ID=Qwen/Qwen3-Reranker-0.6B
```

## Định dạng kịch bản

Nhận cả ba kiểu, tự nhận dạng:

1. Đánh số — `1.` `2.` `Cảnh 3:` `Shot 4 -` (ưu tiên cao nhất, một cảnh trải nhiều dòng cũng được)
2. Các khối cách nhau bằng dòng trống
3. Mỗi dòng là một cảnh

## Đọc kết quả

| Nhãn | Nghĩa |
|---|---|
| `tot` (≥ 0.60) | Reranker tự tin cảnh quay này khớp |
| `tam` (0.30–0.60) | Dùng được nhưng nên xem lại |
| `yeu` (< 0.30) | Khớp kém — cân nhắc quay bổ sung |
| `thieu_canh` | Hết cảnh quay chưa dùng (chỉ xảy ra ở chế độ không tái sử dụng) |

Kế hoạch được lưu ở `vision_storage/_plans/plan_<thời gian>.{json,md}`.
