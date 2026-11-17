# AI-Based Data Center Energy Optimization System (Enterprise Edition)

An industrial-grade, end-to-end capstone system combining a **Physics-Informed Data Center Digital Twin**, a **Multi-Horizon LSTM Workload & Weather Forecaster**, a **Safe Deep Reinforcement Learning (PPO) Cooling & Server Consolidation Controller**, a **FastAPI REST Inference Microservice**, and a **Multi-Page Streamlit Command Center Dashboard**.

---

## Pre-Bundled Ready-to-Run Assets
To save setup time and eliminate dependency ordering issues on your laptop, this repository ships with:
- **`data/raw/dc_telemetry_30d.csv`** — Pre-generated 30-day (8,640 5-minute intervals) synthetic data center telemetry dataset.
- **`artifacts/lstm_forecaster.pt`** — Pre-trained PyTorch `MultiHorizonLSTM` checkpoint (+15 min to +60 min rack CPU & ambient weather forecaster).
- **`artifacts/model_metadata.json`** — Recorded validation metrics and training lineage.

---

## Quick Start on Windows (1-Click Automated Setup)

1. Extract this ZIP folder anywhere on your PC (e.g., `Desktop\dc-energy-optimization`).
2. Double-click **`run_all.bat`**. It automatically:
   - Detects Python 3.10+ and creates an isolated `.venv` virtual environment
   - Installs all required dependencies (`requirements.txt`)
   - Runs `verify_setup.py` to validate Gymnasium and PyTorch
   - Trains the **PPO Reinforcement Learning Agent** (`artifacts/ppo_dc_optimizer.zip`)
   - Runs the **RL vs. PID Benchmark Suite** (`artifacts/benchmark_summary.csv`)
   - Launches the **Streamlit Command Center Dashboard** in your browser at `http://localhost:8501`

### Everyday Launchers (After First Setup)
- **`launch_dashboard.bat`** — Opens the interactive Streamlit Command Center immediately.
- **`launch_api.bat`** — Starts the FastAPI REST server at `http://127.0.0.1:8000/docs` (Swagger UI).

---

## Manual CLI Workflow (Windows / Linux / macOS)

```bash
# 1. Create & activate virtual environment
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS:        source .venv/bin/activate

# 2. Install dependencies & verify environment
pip install -r requirements.txt
python verify_setup.py

# 3. (Optional) Regenerate dataset or retrain LSTM
python scripts/generate_dataset.py --days 30
python scripts/train_forecaster.py --epochs 35

# 4. Train PPO Reinforcement Learning Controller
python scripts/train_agent.py --timesteps 100000

# 5. Run RL vs. PID Benchmark Evaluation
python src/evaluation/benchmark.py

# 6. Launch Dashboard or FastAPI Server
streamlit run dashboard/app.py
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

## Running Automated Unit & Integration Tests

```bash
pytest
```
