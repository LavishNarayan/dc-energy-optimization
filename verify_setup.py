from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main(skip_model_check: bool = False) -> int:
    print("=== AI Data Center Energy Optimization: Pre-Flight Verification ===")
    print(f"Python Version: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("[FAIL] Python 3.10+ is required.")
        return 1

    required_modules = [
        "numpy",
        "pandas",
        "scipy",
        "yaml",
        "torch",
        "gymnasium",
        "stable_baselines3",
        "fastapi",
        "uvicorn",
        "streamlit",
        "plotly",
    ]
    missing = []
    for mod in required_modules:
        try:
            __import__(mod)
            print(f"  [OK] Package '{mod}' installed")
        except ImportError:
            print(f"  [MISSING] Package '{mod}' NOT installed")
            missing.append(mod)

    if missing:
        print(f"\n[FAIL] Missing packages: {', '.join(missing)}. Run: pip install -r requirements.txt")
        return 1

    from gymnasium.utils.env_checker import check_env
    from src.simulator.dc_env import DataCenterEnv

    env = DataCenterEnv()
    check_env(env)
    print("  [OK] Gymnasium DataCenterEnv passed check_env()")

    if not skip_model_check:
        dataset_path = ROOT_DIR / "data" / "raw" / "dc_telemetry_30d.csv"
        lstm_path = ROOT_DIR / "artifacts" / "lstm_forecaster.pt"
        ppo_path = ROOT_DIR / "artifacts" / "ppo_dc_optimizer.zip"
        print(f"  [{'OK' if dataset_path.exists() else 'WARN'}] 30-day telemetry dataset: {dataset_path.name}")
        print(f"  [{'OK' if lstm_path.exists() else 'WARN'}] Trained LSTM forecaster: {lstm_path.name}")
        print(f"  [{'OK' if ppo_path.exists() else 'WARN'}] Trained PPO RL policy: {ppo_path.name}")

    print("\n[SUCCESS] System verification complete!")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-model-check", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(skip_model_check=args.skip_model_check))
