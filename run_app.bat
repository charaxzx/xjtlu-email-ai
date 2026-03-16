@echo off
chcp 65001 >nul
title AI Email Assistant Launcher
cls

echo ========================================================
echo          AI Email Assistant Launcher
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python environment...

if exist ".venv\Scripts\activate.bat" (
    echo    - Found .venv, activating...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo    - Found venv, activating...
    call venv\Scripts\activate.bat
) else (
    echo    - No virtual environment found, using system Python...
)

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python not found!
    echo Please make sure Python is installed and added to PATH.
    echo.
    pause
    exit /b
)

echo.
echo [2/3] Starting server...
echo    - URL: http://localhost:8001
echo    - Opening browser...
echo.

start "" "http://localhost:8001"

echo [3/3] Server started. Do not close this window.
echo.
echo Press Ctrl+C to stop.
echo ========================================================
echo.

cd src
python app.py

pause
