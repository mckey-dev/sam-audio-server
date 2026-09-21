@echo off
setlocal
cd /d "%~dp0"
REM RX 7900 XTX — gfx1100. Pin versions per AMD docs if install fails.

if not exist venv-amd\Scripts\python.exe (
  py -3.12 -m venv venv-amd
  if errorlevel 1 py -3 -m venv venv-amd
)
venv-amd\Scripts\python.exe -m pip install -U pip
venv-amd\Scripts\python.exe -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ ^
  "torch[device-gfx1100]==2.12.0+rocm7.14.1" ^
  "torchvision[device-gfx1100]==0.27.0+rocm7.14.1" ^
  "torchaudio==2.11.0+rocm7.14.1"
venv-amd\Scripts\python.exe -m pip install -r requirements-base.txt
venv-amd\Scripts\python.exe -m pip install git+https://github.com/facebookresearch/sam-audio.git
venv-amd\Scripts\python.exe -m pip uninstall -y torchcodec
echo Done. Run: run_amd.bat
endlocal
