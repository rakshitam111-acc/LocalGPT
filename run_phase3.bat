@echo off
chcp 65001 >nul
title LocalGPT Platform - Multi-Device Server
cls
echo ====================================================================
echo  🚀 LocalGPT AI Platform Server
echo ====================================================================
echo.
echo  💻 On THIS computer, open:
echo     👉 http://localhost:8000
echo.
echo  📱 On your PHONE / OTHER DEVICES (Same Wi-Fi), open:
echo     👉 http://10.99.79.131:8000
echo.
echo ====================================================================
echo  Starting server... (Keep this window open)
echo ====================================================================
echo.

:: Automatically free port 8000 if another instance is already running
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

cd /d "%~dp0"
set PYTHONPATH=%~dp0backend;%~dp0
"%~dp0.venv\Scripts\python.exe" -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000

pause
