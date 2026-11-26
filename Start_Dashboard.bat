@echo off
cd /d "%~dp0"
echo ============================================================================
echo   VeriMedia - Multimodal Deepfake ^& Misinformation Forensic Studio
echo ============================================================================
echo.
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv not found!
    pause
    exit /b 1
)
echo [*] Launching Interactive Streamlit Dashboard on http://localhost:8501 ...
echo [*] Press Ctrl+C in this window to stop the server.
echo.
".venv\Scripts\python.exe" -m streamlit run frontend/app.py --server.port 8501
pause
