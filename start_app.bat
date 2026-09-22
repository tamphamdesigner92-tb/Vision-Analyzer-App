@echo off
rem ============================================================
rem  Vision Analyzer - double click de khoi dong va mo trinh duyet
rem ============================================================

rem Chuyen console sang UTF-8 de log tieng Viet cua server hien dung
chcp 65001 >nul 2>&1

setlocal
cd /d "%~dp0"

set "PORT=8000"
set "URL=http://127.0.0.1:%PORT%"
set "PY=.venv\Scripts\python.exe"

title Vision Analyzer (dong cua so nay de tat server)

if not exist "%PY%" (
    echo [X] Khong tim thay Python trong moi truong ao: %PY%
    echo.
    echo     Hay tao lai moi truong ao truoc khi chay:
    echo         python -m venv .venv
    echo         .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not exist "web_app.py" (
    echo [X] Khong tim thay web_app.py trong: %cd%
    echo     File .bat nay phai nam cung thu muc voi web_app.py
    echo.
    pause
    exit /b 1
)

rem --- Neu server da chay san thi chi mo trinh duyet, khong khoi dong lan hai ---
powershell -NoProfile -Command "try{$c=New-Object Net.Sockets.TcpClient;$c.Connect('127.0.0.1',%PORT%);$c.Close();exit 0}catch{exit 1}" >nul 2>&1
if not errorlevel 1 (
    echo [i] Server dang chay san tai %URL%
    echo     Dang mo trinh duyet...
    start "" "%URL%"
    rem Dung ping thay cho timeout: timeout se loi neu stdin bi chuyen huong
    ping -n 4 127.0.0.1 >nul 2>&1
    exit /b 0
)

echo.
echo   Vision Analyzer
echo   ---------------------------------------------------------
echo   Thu muc anh/video : %cd%\media_input
echo   Thu muc ket qua   : %cd%\vision_storage
echo   Dia chi giao dien : %URL%
echo.
echo   Trinh duyet se tu mo khi server san sang (~5 giay).
echo   Tien trinh phan tich se hien truc tiep tren cua so nay.
echo.
echo   Ung dung se TU TAT khi ban dong tab trinh duyet,
echo   hoac khi ban dong cua so nay / bam Ctrl+C.
echo   ---------------------------------------------------------
echo.

rem --- Cho server mo cong roi tu mo trinh duyet (chay nen, cua so thu nho) ---
start "Vision Analyzer - cho server" /min powershell -NoProfile -ExecutionPolicy Bypass -Command "for($i=0;$i -lt 150;$i++){try{$c=New-Object Net.Sockets.TcpClient;$c.Connect('127.0.0.1',%PORT%);$c.Close();Start-Process '%URL%';exit}catch{Start-Sleep -Milliseconds 800}}"

rem --- Chay server o cua so nay de xem duoc log tien trinh ---
rem -u = khong dem output, de tien trinh hien ngay thay vi bi giu lai trong buffer
"%PY%" -u web_app.py

echo.
echo [i] Server da dung.
pause
exit /b 0
