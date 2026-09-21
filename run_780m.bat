@echo off
setlocal
cd /d "%~dp0"
set HF_HOME=%CD%\models
set HUGGINGFACE_HUB_CACHE=%CD%\models\hub
set XFORMERS_FORCE_DISABLE_TRITON=1
set SAM_AUDIO_EXPECT_BACKEND=rocm
REM echo Browser test: http://127.0.0.1:8765/test/
venv\Scripts\python.exe -m modules.server --host 127.0.0.1 --port 8765 --device cuda %*
endlocal
