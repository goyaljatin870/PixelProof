@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================================
echo   VeriMedia - End-to-End CLI Pipeline ^& System Verification Demo
echo ============================================================================
echo.
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv not found!
    pause
    exit /b 1
)
".venv\Scripts\python.exe" run_all.py --demo
echo.
echo [*] Running Automated PyTest Verification Suite...
".venv\Scripts\python.exe" -m pytest tests/ -v
echo.
pause
