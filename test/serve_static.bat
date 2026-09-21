@echo off
REM Optional: serve test/ on port 8080 (API must allow CORS — server defaults include localhost:8080).
cd /d "%~dp0"
cd ..
if exist venv\Scripts\python.exe (
  venv\Scripts\python.exe -m http.server 8080 --directory test
) else (
  py -3 -m http.server 8080 --directory test
)
