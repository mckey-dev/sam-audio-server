@echo off
setlocal
cd /d "%~dp0"
if not exist venv-cpu\Scripts\python.exe (
  echo CPU 用の環境がありません。先に install_cpu.bat を実行してください。
  exit /b 1
)
set HF_HOME=%CD%\models
set HUGGINGFACE_HUB_CACHE=%CD%\models\hub
set SAM_AUDIO_EXPECT_BACKEND=cpu
REM echo Browser test: http://127.0.0.1:8765/test/
venv-cpu\Scripts\python.exe -m modules.server --host 127.0.0.1 --port 8765 --device cpu %*
endlocal
