@echo off
rem ============================================================
rem  Cai moi truong ao THU HAI cho sidecar nhung da phuong thuc
rem  (Qwen3-VL-Embedding-2B). Chi can chay MOT LAN.
rem
rem  VI SAO PHAI CO VENV RIENG:
rem  Model nay doi transformers>=4.57 va torch 2.8. Ung dung chinh bi khoa o
rem  transformers 4.51.3 / torch 2.5.1 vi AutoAWQ - nang len la vo Qwen2.5-VL-AWQ.
rem  Nen hai bo thu vien phai nam o hai venv khac nhau, chay o hai tien trinh khac nhau.
rem
rem  Dung luong tai ve: ~3GB thu vien + ~4.5GB trong so model.
rem ============================================================

chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0"

set "PY=.venv-embed\Scripts\python.exe"

echo.
echo   Cai sidecar nhung da phuong thuc (Qwen3-VL-Embedding-2B)
echo   ---------------------------------------------------------
echo.

if not exist "%PY%" (
    echo [*] Dang tao moi truong ao .venv-embed ...
    python -m venv .venv-embed
    if errorlevel 1 (
        echo [X] Khong tao duoc moi truong ao. Kiem tra lai Python tren may.
        pause
        exit /b 1
    )
)

echo [*] Nang cap pip ...
"%PY%" -m pip install --upgrade pip wheel
if errorlevel 1 goto :failed

echo.
echo [*] Cai torch 2.8 ban CUDA 12.8 (~2.5GB, cho mot lat) ...
"%PY%" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :failed

echo.
echo [*] Cai transformers va cac thu vien con lai ...
"%PY%" -m pip install -r embedding_sidecar\requirements.txt
if errorlevel 1 goto :failed

echo.
echo [*] Kiem tra nhanh ...
"%PY%" -c "import torch,transformers;print('torch',torch.__version__,'| CUDA',torch.cuda.is_available(),'| transformers',transformers.__version__)"
if errorlevel 1 goto :failed

echo.
echo   ---------------------------------------------------------
echo   [OK] Da cai xong.
echo.
echo   Tu gio, bat sidecar bang: start_embedding_sidecar.bat
echo   (bat TRUOC khi bam "Khop kich ban" thi app chinh se tu dung
echo    tang loc nhanh; khong bat thi app van chay binh thuong)
echo   ---------------------------------------------------------
echo.
pause
exit /b 0

:failed
echo.
echo [X] Cai dat that bai o buoc tren. Doc thong bao loi de biet ly do.
pause
exit /b 1
