@echo off
chcp 65001 >nul
title LocalGPT — 24/7 Hugging Face Cloud Deployer
color 0B

echo =====================================================================
echo          🚀 LOCALGPT 24/7 HUGGING FACE CLOUD DEPLOYER
echo =====================================================================
echo.
echo This tool deploys LocalGPT to Hugging Face Spaces so it runs 24/7 in
echo the cloud even when your laptop is sleeping or turned off!
echo.
echo STEP 1: Create a free Space on Hugging Face:
echo   1. Open: https://huggingface.co/new-space
echo   2. Name your Space (e.g. "localgpt-ai")
echo   3. Select License: "MIT" or "Apache 2.0"
echo   4. Select SDK: "Docker" -> "Blank"
echo   5. Click "Create Space"
echo.
echo =====================================================================
echo.

set /p HF_REPO="Enter your Hugging Face Space Git URL (e.g. https://huggingface.co/spaces/username/localgpt-ai): "

if "%HF_REPO%"=="" (
    echo.
    echo [ERROR] No URL entered. Exiting...
    pause
    exit /b
)

echo.
echo [1/3] Preparing Git repository...
git init >nul 2>&1
git remote remove space >nul 2>&1
git remote add space %HF_REPO%

echo [2/3] Adding application files...
git add Dockerfile README.md backend/ frontend/ data/ requirements.txt

echo [3/3] Committing and deploying to 24/7 Hugging Face Cloud...
git commit -m "Deploy 24/7 LocalGPT with RAG, Vision & Web Search" >nul 2>&1
git push -f space main

echo.
echo =====================================================================
echo 🎉 DEPLOYMENT COMPLETE!
echo.
echo Your LocalGPT AI platform is now building on Hugging Face Spaces!
echo In 1-2 minutes, open your Space URL on your phone or any device:
echo 👉 %HF_REPO%
echo =====================================================================
echo.
pause
