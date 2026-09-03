@echo off
chcp 65001 >nul
title LocalGPT — Complete 24/7 Platform Launcher
color 0A

echo =====================================================================
echo                🚀 LOCALGPT MULTI-DEVICE LAUNCHER
echo =====================================================================
echo.

cd /d "c:\Users\ASUS\OneDrive\Attachments\Desktop\LLM XRAY"

echo [1/3] Detecting Active Local Wi-Fi IP...
for /f "tokens=*" %%i in ('powershell -Command "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -match 'Wi-Fi' -or $_.InterfaceAlias -match 'Wireless' } | Select-Object -First 1).IPAddress"') do set LOCAL_IP=%%i

if "%LOCAL_IP%"=="" (
    for /f "tokens=*" %%i in ('powershell -Command "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notmatch 'Loopback' -and $_.IPAddress -match '^(192\.168|10\.|172\.)' } | Select-Object -First 1).IPAddress"') do set LOCAL_IP=%%i
)

echo       Wi-Fi Direct Link: http://%LOCAL_IP%:8080
echo.

echo [2/3] Starting LocalGPT Python Backend on Port 8080...
start /b "LocalGPT Backend" "c:\Users\ASUS\OneDrive\Attachments\Desktop\LLM XRAY\.venv\Scripts\python.exe" -m uvicorn app.main:app --app-dir "c:\Users\ASUS\OneDrive\Attachments\Desktop\LLM XRAY\backend" --host 0.0.0.0 --port 8080 > nul 2>&1

timeout /t 3 /nobreak > nul

echo [3/3] Starting Cloudflare Public Secure Tunnel...
echo.
echo =====================================================================
echo 🌐 CONNECT FROM ANY DEVICE ANYWHERE:
echo.
echo ⚡ On Same Wi-Fi / Hotspot:
echo    👉 http://%LOCAL_IP%:8080
echo.
echo 🌍 On Mobile Data / Outside Home:
echo    Watch below for your https://*.trycloudflare.com link...
echo =====================================================================
echo.

start http://localhost:8080

.\cloudflared.exe tunnel --no-tls-verify --url http://localhost:8080
