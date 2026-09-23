@echo off
rem ============================================================
rem  Vision Analyzer - double click de khoi dong va mo trinh duyet
rem ============================================================
rem Server tu chon mot cong con trong (khong ghim 8000 nua, de khoi dam vao ung dung
rem khac) roi ghi so cong ra .runtime_port. File .bat nay doc file do de mo trinh duyet.
rem Muon ghim mot cong co dinh: set PORT=8000 truoc khi chay.

rem Chuyen console sang UTF-8 de log tieng Viet cua server hien dung
chcp 65001 >nul 2>&1

setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "PORT_FILE=.runtime_port"

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
rem Kiem tra ca cong co that su mo hay khong: file .runtime_port con sot lai sau mot lan
rem server chet bat thuong se tro toi mot cong khong con ai nghe.
set "RUNNING_PORT="
if exist "%PORT_FILE%" set /p RUNNING_PORT=<"%PORT_FILE%"
if defined RUNNING_PORT (
    rem Hoi thang /api/status thay vi chi mo ket noi TCP: cong dong nghia la so cong ghi
    rem trong .runtime_port co the da bi mot ung dung khac nhan lai sau do.
    powershell -NoProfile -Command "try{$r=Invoke-WebRequest -Uri ('http://127.0.0.1:%RUNNING_PORT%/api/status') -TimeoutSec 2 -UseBasicParsing;if($r.Content -match '\"models\"'){exit 0};exit 1}catch{exit 1}" >nul 2>&1
    if not errorlevel 1 (
        echo [i] Server dang chay san tai http://127.0.0.1:%RUNNING_PORT%
        echo     Dang mo trinh duyet...
        start "" "http://127.0.0.1:%RUNNING_PORT%"
        rem Dung ping thay cho timeout: timeout se loi neu stdin bi chuyen huong
        ping -n 4 127.0.0.1 >nul 2>&1
        exit /b 0
    )
)

rem File cu khong con gia tri -> xoa di de vong cho ben duoi khong bat nham so cong cu
del /q "%PORT_FILE%" >nul 2>&1

echo.
echo   Vision Analyzer
echo   ---------------------------------------------------------
echo   Thu muc anh/video : %cd%\media_input
echo   Thu muc ket qua   : %cd%\vision_storage
echo   Dia chi giao dien : server tu chon cong, se in ra ngay duoi day
echo.
echo   Trinh duyet se tu mo khi server san sang (~5 giay).
echo   Tien trinh phan tich se hien truc tiep tren cua so nay.
echo.
echo   Ung dung se TU TAT khi ban dong tab trinh duyet,
echo   hoac khi ban dong cua so nay / bam Ctrl+C.
echo   ---------------------------------------------------------
echo.

rem --- Cho server ghi so cong va mo cong roi tu mo trinh duyet (chay nen, cua so thu nho) ---
start "Vision Analyzer - cho server" /min powershell -NoProfile -ExecutionPolicy Bypass -Command "for($i=0;$i -lt 150;$i++){$p=$null;if(Test-Path '%PORT_FILE%'){$p=(Get-Content '%PORT_FILE%' -Raw).Trim()};if($p){try{$r=Invoke-WebRequest -Uri (\"http://127.0.0.1:$p/api/status\") -TimeoutSec 2 -UseBasicParsing;if($r.Content -match '\"models\"'){Start-Process \"http://127.0.0.1:$p\";exit}}catch{}};Start-Sleep -Milliseconds 800}"

rem --- Chay server o cua so nay de xem duoc log tien trinh ---
rem -u = khong dem output, de tien trinh hien ngay thay vi bi giu lai trong buffer
"%PY%" -u web_app.py

echo.
echo [i] Server da dung.
pause
exit /b 0
