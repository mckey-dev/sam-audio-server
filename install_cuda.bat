@echo off
setlocal
cd /d "%~dp0"
REM NVIDIA CUDA (RTX 2060 SUPER). Separate venv so ROCm 780M install is kept.

if not exist venv-cuda\Scripts\python.exe (
  py -3.12 -m venv venv-cuda
  if errorlevel 1 py -3 -m venv venv-cuda
)
venv-cuda\Scripts\python.exe -m pip install -U pip
venv-cuda\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
venv-cuda\Scripts\python.exe -m pip install -r requirements-base.txt
venv-cuda\Scripts\python.exe -m pip install git+https://github.com/facebookresearch/sam-audio.git
venv-cuda\Scripts\python.exe -m pip uninstall -y torchcodec
echo Done. Run: run_cuda.bat
endlocal
