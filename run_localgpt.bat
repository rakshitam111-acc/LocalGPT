@echo off
title LocalGPT - Local AI Assistant with RAG & X-Ray
echo ========================================================
echo Starting LocalGPT...
echo ========================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m streamlit run app.py
pause
