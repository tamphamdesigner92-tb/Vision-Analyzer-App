@echo off
rem ============================================================
rem  Bat sidecar nhung da phuong thuc (tang loc nhanh cua cascade).
rem  Chay truoc khi bam "Khop kich ban" neu kho canh quay lon.
rem  Dong cua so nay la tat sidecar; app chinh van chay binh thuong.
rem ============================================================

chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0"

set "PY=.venv-embed\Scripts\python.exe"

title Sidecar nhung da phuong thuc (dong cua so nay de tat)

if not exist "%PY%" (
    echo [X] Chua cai moi truong .venv-embed
    echo     Hay chay setup_embedding_sidecar.bat truoc.
    echo.
    pause
    exit /b 1
)

set "APP_PORT=8000"
set "APP_URL=http://127.0.0.1:%APP_PORT%"

echo.
echo   Sidecar nhung da phuong thuc - Qwen3-VL-Embedding-2B
echo   ---------------------------------------------------------
echo   Dia chi sidecar    : http://127.0.0.1:8011
echo   Giao dien theo doi : %APP_URL%
echo   Lan chay dau se tai ~4.5GB trong so model ve cache huggingface.
echo.
echo   Sidecar khong co giao dien rieng: moi trang thai cua no (dang tai,
echo   dang nap, dang nhung) hien tren thanh trang thai cua ung dung chinh.
echo   ---------------------------------------------------------
echo.

rem --- Dam bao co giao dien de theo doi: sidecar khong tu hien thi duoc gi ---
powershell -NoProfile -Command "try{$c=New-Object Net.Sockets.TcpClient;$c.Connect('127.0.0.1',%APP_PORT%);$c.Close();exit 0}catch{exit 1}" >nul 2>&1
if errorlevel 1 (
    echo [*] Ung dung chinh chua chay - dang khoi dong kem theo...
    rem start_app.bat tu mo trinh duyet khi server san sang, nen khong mo trung o day.
    start "" "%~dp0start_app.bat"
) else (
    echo [*] Ung dung chinh dang chay - dang mo trinh duyet...
    start "" "%APP_URL%"
)

echo.
"%PY%" -u embedding_sidecar\server.py

echo.
echo [i] Sidecar da dung.
pause
exit /b 0
