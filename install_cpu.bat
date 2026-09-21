@echo off
setlocal
cd /d "%~dp0"

if not exist venv-cpu\Scripts\python.exe (
  py -3.12 -m venv venv-cpu
  if errorlevel 1 py -3 -m venv venv-cpu
)
venv-cpu\Scripts\python.exe -m pip install -U pip
venv-cpu\Scripts\python.exe -m pip install torch torchvision torchaudio
venv-cpu\Scripts\python.exe -m pip install -r requirements-base.txt
venv-cpu\Scripts\python.exe -m pip install git+https://github.com/facebookresearch/sam-audio.git
venv-cpu\Scripts\python.exe -m pip uninstall -y torchcodec
echo Done. Run: run_cpu.bat
endlocal
