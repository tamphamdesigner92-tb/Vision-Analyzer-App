#!/usr/bin/env bash
# ============================================================
#  Vision Analyzer - chay va tu mo trinh duyet (macOS)
# ============================================================
# Server tu chon mot cong con trong (khong ghim 8000 nua, de khoi dam vao ung dung khac)
# roi ghi so cong ra .runtime_port. Script nay doc file do de biet ma mo trinh duyet.
# Muon ghim mot cong co dinh: chay voi PORT=8000 ./start_app.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python3"
PORT_FILE=".runtime_port"

if [[ ! -x "$PY" ]]; then
    echo "[X] Không tìm thấy Python trong môi trường ảo: $PY"
    echo ""
    echo "    Hãy tạo lại môi trường ảo trước khi chạy:"
    echo "        python3 -m venv .venv"
    echo "        .venv/bin/python3 -m pip install --upgrade pip"
    echo "        .venv/bin/python3 -m pip install -r requirements.txt"
    echo ""
    exit 1
fi

if [[ ! -f "web_app.py" ]]; then
    echo "[X] Không tìm thấy web_app.py trong: $(pwd)"
    echo "    File start_app.sh này phải nằm cùng thư mục với web_app.py"
    exit 1
fi

# Hoi thang /api/status thay vi chi mo mot ket noi TCP: cong dong nghia la so cong ghi
# trong .runtime_port co the da bi mot ung dung khac cua ban nhan lai sau do. Ket noi TCP
# thanh cong chi noi "co ai do dang nghe", khong noi "do la Vision Analyzer".
app_alive() {
    local port="$1"
    [[ -n "$port" ]] || return 1
    curl -fsS -m 2 "http://127.0.0.1:${port}/api/status" 2>/dev/null | grep -q '"models"'
}

read_port() {
    [[ -f "$PORT_FILE" ]] || return 1
    tr -dc '0-9' < "$PORT_FILE"
}

# --- Neu server da chay san thi chi mo trinh duyet, khong khoi dong lan hai ---
# Kiem tra ca cong co that su mo hay khong: file .runtime_port con sot lai sau mot lan
# server chet bat thuong se tro toi mot cong khong con ai nghe.
if RUNNING_PORT="$(read_port)" && app_alive "$RUNNING_PORT"; then
    echo "[i] Server đang chạy sẵn tại http://127.0.0.1:${RUNNING_PORT}"
    echo "    Đang mở trình duyệt..."
    open "http://127.0.0.1:${RUNNING_PORT}"
    exit 0
fi

# File cu khong con gia tri -> xoa di de vong cho ben duoi khong bat nham so cong cu
rm -f "$PORT_FILE"

echo ""
echo "  Vision Analyzer"
echo "  ---------------------------------------------------------"
echo "  Thư mục ảnh/video : $(pwd)/media_input"
echo "  Thư mục kết quả   : $(pwd)/vision_storage"
echo "  Địa chỉ giao diện : server tự chọn cổng, sẽ in ra ngay dưới đây"
echo ""
echo "  Trình duyệt sẽ tự mở khi server sẵn sàng (~5 giây)."
echo "  Tiến trình phân tích sẽ hiện trực tiếp trên cửa sổ này."
echo ""
echo "  Ứng dụng sẽ TỰ TẮT khi bạn đóng tab trình duyệt,"
echo "  hoặc khi bạn đóng cửa sổ này / bấm Ctrl+C."
echo "  ---------------------------------------------------------"
echo ""

# --- Cho server ghi so cong va mo cong roi tu mo trinh duyet (chay nen) ---
(
    for _ in $(seq 1 150); do
        if PORT_READY="$(read_port)" && app_alive "$PORT_READY"; then
            open "http://127.0.0.1:${PORT_READY}"
            break
        fi
        sleep 0.8
    done
) &

# --- Chay server o cua so nay de xem duoc log tien trinh ---
# -u = khong dem output, de tien trinh hien ngay thay vi bi giu lai trong buffer
exec "$PY" -u web_app.py
