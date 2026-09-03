@echo off
chcp 65001 >nul
title LocalGPT — Setup Auto-Start On Boot
color 0A

echo =====================================================================
echo           🚀 LOCALGPT AUTO-START ON RESTART / BOOT SETUP
echo =====================================================================
echo.

set SCRIPT_DIR=%~dp0
set TARGET_VBS=%SCRIPT_DIR%start_localgpt_background.vbs
set STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
set SHORTCUT_PATH=%STARTUP_DIR%\LocalGPT_AutoStart.lnk

powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT_PATH%'); $s.TargetPath = '%TARGET_VBS%'; $s.WorkingDirectory = '%SCRIPT_DIR%'; $s.IconLocation = '%SCRIPT_DIR%backend\app\static\favicon.ico'; $s.Save()"

if exist "%SHORTCUT_PATH%" (
    echo [SUCCESS] Auto-Start shortcut installed in Windows Startup folder!
    echo.
    echo Path: %SHORTCUT_PATH%
    echo.
    echo =====================================================================
    echo 🎉 YOU ARE ALL SET!
    echo.
    echo Whenever your laptop restarts or boots up:
    echo 1. LocalGPT will start automatically in the background.
    echo 2. Your Wi-Fi link and Cloudflare tunnel will be live immediately!
    echo =====================================================================
) else (
    echo [ERROR] Could not create shortcut. Please try running as Administrator.
)

echo.
pause
