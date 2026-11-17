@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================================
echo  AI-Based Data Center Energy Optimization System - Enterprise Setup & Run
echo ============================================================================

set "SYS_PY="
where python >nul 2>nul
if %errorlevel%==0 (
    python -c "import sys; assert sys.version_info >= (3,10)" >nul 2>nul
    if !errorlevel!==0 set "SYS_PY=python"
)
if not defined SYS_PY (
    where py >nul 2>nul
    if %errorlevel%==0 (
        py -3 -c "import sys; assert sys.version_info >= (3,10)" >nul 2>nul
        if !errorlevel!==0 set "SYS_PY=py -3"
    )
)

if not defined SYS_PY (
    echo [ERROR] Python 3.10+ was not found on your PATH.
    echo Please install Python 3.10 or 3.11 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/7] Creating isolated virtual environment (.venv)...
    %SYS_PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

set "VENV_PY=.venv\Scripts\python.exe"
set "PYTHONPATH=%cd%"

echo [2/7] Installing dependencies from requirements.txt...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
)

echo [3/7] Running pre-flight environment diagnostics...
"%VENV_PY%" verify_setup.py --skip-model-check
if errorlevel 1 (
    pause
    exit /b 1
)

if not exist "data\raw\dc_telemetry_30d.csv" (
    echo [4/7] Generating 30-day synthetic telemetry dataset...
    "%VENV_PY%" scripts\generate_dataset.py
) else (
    echo [4/7] Pre-generated 30-day telemetry dataset found. Skipping regeneration.
)

if not exist "artifacts\lstm_forecaster.pt" (
    echo [5/7] Training Multi-Horizon LSTM Forecaster...
    "%VENV_PY%" scripts\train_forecaster.py
) else (
    echo [5/7] Pre-trained LSTM Forecaster found. Skipping retraining.
)

echo [6/7] Training PPO Deep Reinforcement Learning Agent (100,000 steps)...
"%VENV_PY%" scripts\train_agent.py --timesteps 100000
if errorlevel 1 (
    echo [ERROR] PPO training failed.
    pause
    exit /b 1
)

echo [7/7] Running RL vs. PID Benchmark Suite...
"%VENV_PY%" src\evaluation\benchmark.py

echo ============================================================================
echo  All Done! Launching Interactive Command Center Dashboard...
echo ============================================================================
"%VENV_PY%" -m streamlit run dashboard\app.py
pause
