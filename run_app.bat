@echo off
rem Keep this file ASCII-only: Chinese Windows CMD often mis-parses UTF-8 .bat and tries to run garbage as commands.
chcp 65001 >nul
title Practice2 Web Launcher
cls

echo ========================================================
echo   Practice2 - Web launcher
echo ========================================================
echo.

cd /d "%~dp0"

set "PY_EXE="
if exist ".venv\Scripts\python.exe" (
    set "PY_EXE=%CD%\.venv\Scripts\python.exe"
    echo Found .venv Python.
) else if exist "venv\Scripts\python.exe" (
    set "PY_EXE=%CD%\venv\Scripts\python.exe"
    echo Found venv Python.
) else (
    set "PY_EXE=python"
    echo No project venv found, using python from PATH...
)

"%PY_EXE%" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python not found.
    echo Install Python 3.10+ or create .venv first, then retry.
    echo.
    pause
    exit /b 1
)

echo.
echo Opening browser: http://localhost:8001
start "" "http://localhost:8001"

echo.
echo Starting server... Do not close this window.
echo Press Ctrl+C to stop.
echo.

cd src
"%PY_EXE%" app.py

echo.
echo Server exited.
pause
