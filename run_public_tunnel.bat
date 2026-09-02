@echo off
chcp 65001 >nul
title LocalGPT — 1-Click Server & Web Launcher
color 0A

echo =====================================================================
echo                 🚀 LOCALGPT 1-CLICK LAUNCHER
echo =====================================================================
echo.

:: 1. Clear any stuck process on port 8080
echo [1/4] Checking Port 8080...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8080 ^| findstr LISTENING') do (
    echo Terminating old process on port 8080 (PID: %%a)...
    taskkill /F /PID %%a >nul 2>&1
)

:: 2. Find current Wi-Fi IP address dynamically
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr /i "IPv4" ^| findstr "192. 172. 10."') do (
    set "LOCAL_IP=%%i"
    goto :found_ip
)
:found_ip
set "LOCAL_IP=%LOCAL_IP: =%"
if "%LOCAL_IP%"=="" set "LOCAL_IP=172.21.89.165"

:: 3. Start Backend Server on dedicated Port 8080
echo [2/4] Starting LocalGPT AI Backend Engine on Port 8080...
start /B "" "%~dp0.venv\Scripts\python.exe" -m uvicorn app.main:app --app-dir "%~dp0backend" --host 0.0.0.0 --port 8080

:: 4. Automatically open the browser
echo [3/4] Opening Web Interface in your default browser...
timeout /t 2 /nobreak >nul
start http://localhost:8080

echo.
echo =====================================================================
echo 💻 LOCAL ACCESS:
echo    👉 http://localhost:8080
echo.
echo 📱 MOBILE & OTHER LAPTOPS (Same Wi-Fi / Hotspot - NEVER EXPIRES):
echo    👉 http://%LOCAL_IP%:8080
echo.
echo 🌐 PUBLIC TUNNEL (For Mobile Data / Remote Friends):
echo    Starting Cloudflare Tunnel below...
echo =====================================================================
echo.

:: 5. Launch Cloudflare Tunnel pointing to Port 8080
"%~dp0cloudflared.exe" tunnel --no-tls-verify --url http://localhost:8080

pause
