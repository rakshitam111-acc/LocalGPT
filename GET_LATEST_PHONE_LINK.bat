@echo off
chcp 65001 >nul
title LocalGPT — Current Phone Link
color 0B

for /f "tokens=*" %%i in ('powershell -Command "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -match 'Wi-Fi' -or $_.InterfaceAlias -match 'Wireless' } | Select-Object -First 1).IPAddress"') do set LOCAL_IP=%%i

echo =====================================================================
echo           📱 YOUR ACTIVE LOCALGPT PHONE LINK RIGHT NOW:
echo =====================================================================
echo.
echo ⚡ On Wi-Fi / Hotspot (Instant ^& 0 Lag):
echo    👉 http://%LOCAL_IP%:8080
echo.
echo =====================================================================
echo.
pause
