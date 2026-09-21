@echo off
setlocal
cd /d "%~dp0"
REM Radeon 780M — gfx1103. Pin versions per AMD docs if install fails.

if not exist venv\Scripts\python.exe (
  py -3.12 -m venv venv
  if errorlevel 1 py -3 -m venv venv
)
venv\Scripts\python.exe -m pip install -U pip
venv\Scripts\python.exe -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ ^
  "torch[device-gfx1103]==2.12.0+rocm7.14.1" ^
  "torchvision[device-gfx1103]==0.27.0+rocm7.14.1" ^
  "torchaudio==2.11.0+rocm7.14.1"
venv\Scripts\python.exe -m pip install -r requirements-base.txt
venv\Scripts\python.exe -m pip install git+https://github.com/facebookresearch/sam-audio.git
venv\Scripts\python.exe -m pip uninstall -y torchcodec
echo Done. Run: run_780m.bat
endlocal
