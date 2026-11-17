@echo off
cd /d "%~dp0"
set "PYTHONPATH=%cd%"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run dashboard\app.py
) else (
    python -m streamlit run dashboard\app.py
)
pause
