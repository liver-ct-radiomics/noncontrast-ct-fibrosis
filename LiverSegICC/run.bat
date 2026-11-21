@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo The program is not set up yet. Please run setup.bat first.
  pause
  exit /b 1
)
set PYTHONDONTWRITEBYTECODE=1
start "" ".venv\Scripts\pythonw.exe" "src\app.py"
