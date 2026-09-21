@echo off
setlocal
cd /d "%~dp0"
if not exist venv-cuda\Scripts\python.exe (
  echo RTX 2060 SUPER 用の CUDA 環境がありません。先に install_cuda.bat を実行してください。
  echo 現在の venv は ROCm（Radeon 780M）用なので、NVIDIA GPU は認識されません。
  exit /b 1
)
set HF_HOME=%CD%\models
set HUGGINGFACE_HUB_CACHE=%CD%\models\hub
set SAM_AUDIO_EXPECT_BACKEND=cuda
REM echo Browser test: http://127.0.0.1:8765/test/
venv-cuda\Scripts\python.exe -m modules.server --host 127.0.0.1 --port 8765 --device cuda %*
endlocal
