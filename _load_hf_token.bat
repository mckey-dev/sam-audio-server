@echo off
REM Load HF_TOKEN from hf_token.txt if the environment variable is empty.
if defined HF_TOKEN goto :eof
if exist "%~dp0hf_token.txt" (
  set /p HF_TOKEN=<"%~dp0hf_token.txt"
)
