@echo off
cd /d "%~dp0"
set "PYTHONPATH=%cd%"
echo Starting FastAPI Inference & Telemetry Server at http://127.0.0.1:8000/docs
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
) else (
    python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
)
pause
