@echo off
setlocal EnableExtensions
chcp 65001 >nul
title LiverSegICC setup
cd /d "%~dp0"

rem A zip viewer runs files from a temporary copy, which would vanish later
echo %~dp0 | findstr /r /i /c:"\\Temp[0-9]*_.*\.zip" >nul
if not errorlevel 1 (
  echo This folder is still inside the zip file.
  echo Extract the zip to a normal folder first, then run setup.bat from there.
  pause
  exit /b 1
)

echo.
echo === LiverSegICC setup ===
echo.
echo Looking for Python 3.11 or 3.12...
set "PY="
call :try_python py -3.11
if not defined PY call :try_python py -3.12
if not defined PY call :try_python "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY call :try_python "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY call :try_python python
if not defined PY (
  echo.
  echo Python 3.11 or 3.12 was not found.
  echo The download page will open now. Install Python 3.11, keeping the default options
  echo including "tcl/tk", then run setup.bat again.
  echo     https://www.python.org/downloads/release/python-3119/
  start "" "https://www.python.org/downloads/release/python-3119/"
  echo.
  pause
  exit /b 1
)
echo Using: %PY%

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo Creating the program's Python environment in .venv ...
  %PY% -m venv .venv
  if errorlevel 1 goto :failed
)

echo.
echo Installing the libraries for the editor and the analysis ^(about 150 MB^)...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --no-input -r requirements.txt
if errorlevel 1 goto :failed

echo.
echo Checking the calculations...
set PYTHONDONTWRITEBYTECODE=1
".venv\Scripts\python.exe" "src\selftest.py"
if errorlevel 1 goto :failed

echo.
echo Setup finished. Start the program with run.bat.
echo TotalSegmentator is looked for when the program opens; if it is missing, the program
echo shows how to install it.
echo.
pause
exit /b 0

:failed
echo.
echo Setup did not finish. Please read the messages above. Internet access is needed.
echo.
pause
exit /b 1

:try_python
%* -c "import sys, tkinter; sys.exit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) else 1)" >nul 2>&1
if not errorlevel 1 set PY=%*
exit /b 0
