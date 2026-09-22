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
        ├──② Qwen3-VL-Embedding-2B  (tuỳ chọn, sidecar)
        │     nhúng THẲNG ảnh + mô tả thành vector
        │     → lọc nhanh top-8 ứng viên cho mỗi dòng kịch bản
        ▼
        ③ Qwen3-Reranker-4B
           chấm kỹ từng cặp (dòng kịch bản, mô tả cảnh) → xác suất 0..1
        ▼
        ④ Xếp cảnh (greedy toàn cục) → kế hoạch dựng .md / .json
```

Vì sao phải hai mô hình chứ không một:

| | Reranker-4B | VL-Embedding-2B |
|---|---|---|
| Đọc được ảnh/video | Không, chỉ chữ | Có |
| Độ chính xác từng cặp | Cao (cross-encoder, đọc cả hai vế cùng lúc) | Thấp hơn (hai vế nhúng riêng rồi so vector) |
| Chi phí | N×M lần chạy GPU | N+M lần nhúng, **cache lại**, so khớp sau đó gần như miễn phí |

Nên embedding lo phần *rộng* (quét cả kho thật nhanh), reranker lo phần *sâu* (chốt đúng cảnh).
Đây là mô hình retrieve → rerank tiêu chuẩn.

## Hai môi trường ảo — cố ý, không phải trùng lặp

`Qwen3-VL-Embedding-2B` đòi `transformers>=4.57` + `torch 2.8`, trong khi ứng dụng chính
bị khoá ở `transformers 4.51.3` + `torch 2.5.1` vì AutoAWQ (nâng lên là vỡ Qwen2.5-VL-AWQ,
xem cảnh báo đầu `requirements.txt`). Hai bộ thư viện không sống chung được trong một
tiến trình, nên:

- `.venv` — app chính: Qwen2.5-VL + Qwen3-Reranker
- `.venv-embed` — sidecar: Qwen3-VL-Embedding, chạy riêng ở cổng 8011

Sidecar là **tuỳ chọn**. Không bật thì reranker chấm toàn bộ kho, kết quả vẫn đúng, chỉ chậm hơn khi kho lớn.

Đo trên máy này (30 cảnh quay, 3 dòng kịch bản): bật tầng lọc nhanh thì reranker chỉ
chấm 8 cảnh/dòng thay vì 30, và **cho ra đúng cùng một kết quả** — 0.972 và 0.980 ở hai
dòng có cảnh khớp. Vector của cảnh quay được cache trên đĩa nên lần chạy thứ hai trở đi
bỏ qua hẳn bước nhúng.

## Chia chỗ trên GPU 16GB

Ba mô hình cộng lại vượt xa 16GB, nên mỗi lúc chỉ một mô hình được nằm trên GPU:

- Bấm "Phân tích" (tab 1) → tự giải phóng reranker, nạp Qwen2.5-VL
- Bấm "Khớp kịch bản" (tab 2) → nhúng trước bằng sidecar, bảo sidecar nhả VRAM,
  giải phóng Qwen2.5-VL, rồi mới nạp reranker

Mỗi lần đổi mất ~30 giây nạp lại, đổi lấy việc không bao giờ tràn VRAM giữa chừng.

Thứ tự này quan trọng: nếu để Qwen2.5-VL nằm lại trong lúc sidecar nạp model nhúng thì
VRAM đỉnh chạm 15.4/16.4GB — sát ngưỡng tràn. Giải phóng trước khi gọi sidecar thì đỉnh
chỉ còn 12.5GB (số đo thật bằng `nvidia-smi`).

## Theo dõi: mọi thứ hiện trên giao diện web

Thanh trạng thái nằm ngay dưới thanh tiêu đề, **luôn hiện ở cả hai tab**, và cập nhật
mỗi 1.5 giây. Nó trả lời đúng bốn câu hỏi hay gặp:

| Câu hỏi | Chỗ trả lời |
|---|---|
| Đang chạy tác vụ gì? | Dòng đầu: "Phân tích ảnh/video: tên_file — 45%" hoặc "Khớp kịch bản — 60%" |
| Đang làm gì trong tác vụ đó? | Dòng phụ màu xám: thông điệp mới nhất (đang nạp model, đang chấm cảnh 2/4…) |
| Model nào đang nằm trên GPU? | Ba chip: `không nạp` / `đang tải trọng số` / `đang nạp lên GPU` / `sẵn sàng` |
| Còn bao nhiêu VRAM? | Chip VRAM, chuyển đỏ kèm chữ "gần đầy" khi vượt 85% |

**Không cần mở cửa sổ console của sidecar.** App chính hỏi `/health` của sidecar và hiển
thị luôn trạng thái của nó — kể cả tiến độ tải 4GB trọng số lần đầu và số mục đang nhúng.
Sidecar tắt thì chip ghi rõ "chưa bật sidecar".

Lần đầu dùng mỗi mô hình sẽ có giai đoạn tải hàng GB từ mạng. Giai đoạn này được đo bằng
cách so dung lượng thư mục cache với tổng dung lượng repo, nên thanh tiến độ chạy thật
chứ không đứng im — đây là lúc dễ tưởng nhầm ứng dụng bị treo nhất.

## Cách chạy

```
setup_embedding_sidecar.bat      # chạy MỘT LẦN, cài .venv-embed (~3GB + ~4.5GB model)

start_app.bat                    # app chính, tự mở trình duyệt
start_embedding_sidecar.bat      # sidecar + tự mở giao diện để theo dõi
```

Bấm file nào cũng có giao diện: `start_embedding_sidecar.bat` kiểm tra cổng 8000, thấy
app chính chưa chạy thì khởi động kèm luôn, còn đang chạy rồi thì chỉ mở trình duyệt.
Sidecar không có giao diện riêng nên nếu không làm vậy, cửa sổ của nó chỉ là một khung
đen không nói lên điều gì.

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
| `chua_cham` (điểm âm) | Tầng lọc nhanh đã gạt ra, reranker chưa hề chấm; chỉ được lấy để trám chỗ |
| `thieu_canh` | Hết cảnh quay chưa dùng (chỉ xảy ra ở chế độ không tái sử dụng) |

Kế hoạch được lưu ở `vision_storage/_plans/plan_<thời gian>.{json,md}`.
