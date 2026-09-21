@echo off
setlocal
cd /d "%~dp0"
if not exist venv-amd\Scripts\python.exe (
  echo RX 7900 XTX 用の ROCm 環境がありません。先に install_amd.bat を実行してください。
  exit /b 1
)
set HF_HOME=%CD%\models
set HUGGINGFACE_HUB_CACHE=%CD%\models\hub
set XFORMERS_FORCE_DISABLE_TRITON=1
set SAM_AUDIO_EXPECT_BACKEND=rocm
call "%~dp0_load_hf_token.bat"
REM echo Browser test: http://127.0.0.1:8765/test/
venv-amd\Scripts\python.exe -m modules.server --host 127.0.0.1 --port 8765 --device cuda %*
endlocal
