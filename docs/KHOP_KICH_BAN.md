# Khớp kịch bản — kiến trúc "bộ não" chọn và sắp xếp cảnh

Tab 2 của ứng dụng nhận một kịch bản và trả về **kế hoạch dựng**: mỗi dòng kịch bản được
gán một cảnh quay có sẵn, đã sắp đúng thứ tự kịch bản, kèm điểm tin cậy và phương án thay thế.

## Đường ống

```
media_input/*.mp4, *.jpg
        │
        │  ① Qwen2.5-VL-7B-AWQ   (tab 1, đã có sẵn)
        ▼
vision_storage/<tên file>/*.json     mô tả bằng chữ + keyframe
        │
        ▼
        ② Qwen3-Reranker-4B
           chấm kỹ từng cặp (dòng kịch bản, mô tả cảnh) → xác suất 0..1
        ▼
        ③ Xếp cảnh (greedy toàn cục) → kế hoạch dựng .md / .json
```

## Chia chỗ trên GPU 16GB

Hai mô hình cộng lại (~7GB + ~8GB) sát ngưỡng 16GB, nên mỗi lúc chỉ một mô hình được nằm trên GPU:

- Bấm "Phân tích" (tab 1) → tự giải phóng reranker, nạp Qwen2.5-VL
- Bấm "Khớp kịch bản" (tab 2) → giải phóng Qwen2.5-VL, rồi mới nạp reranker

Mỗi lần đổi mất ~30 giây nạp lại, đổi lấy việc không bao giờ tràn VRAM giữa chừng.

## Theo dõi: mọi thứ hiện trên giao diện web

Thanh trạng thái nằm ngay dưới thanh tiêu đề, **luôn hiện ở cả hai tab**, và cập nhật
mỗi 1.5 giây. Nó trả lời đúng bốn câu hỏi hay gặp:

| Câu hỏi | Chỗ trả lời |
|---|---|
| Đang chạy tác vụ gì? | Dòng đầu: "Phân tích ảnh/video: tên_file — 45%" hoặc "Khớp kịch bản — 60%" |
| Đang làm gì trong tác vụ đó? | Dòng phụ màu xám: thông điệp mới nhất (đang nạp model, đang chấm cảnh 2/4…) |
| Model nào đang nằm trên GPU? | Hai chip: `không nạp` / `đang tải trọng số` / `đang nạp lên GPU` / `sẵn sàng` |
| Còn bao nhiêu VRAM? | Chip VRAM, chuyển đỏ kèm chữ "gần đầy" khi vượt 85% |

Lần đầu dùng mỗi mô hình sẽ có giai đoạn tải hàng GB từ mạng. Giai đoạn này được đo bằng
cách so dung lượng thư mục cache với tổng dung lượng repo, nên thanh tiến độ chạy thật
chứ không đứng im — đây là lúc dễ tưởng nhầm ứng dụng bị treo nhất.

## Cách chạy

```
start_app.bat                    # tự mở trình duyệt
```

Đổi mô hình reranker bằng biến môi trường, không cần sửa code:

```
set RERANKER_MODEL_ID=Qwen/Qwen3-Reranker-0.6B
```

Bản 0.6B chỉ ~1.2GB, nhẹ đến mức nằm chung GPU với Qwen2.5-VL được — hợp khi muốn
khỏi phải đổi model qua lại.

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
