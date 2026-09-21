@echo off
setlocal
cd /d "%~dp0"
echo SAM Audio weights are downloaded from aoiandroid/sam-audio-models.
echo Login is usually optional. Use it only if a Hugging Face download asks for auth.
echo.
if exist venv\Scripts\hf.exe (
  venv\Scripts\hf.exe auth login
) else (
  venv\Scripts\huggingface-cli.exe login
)
endlocal
