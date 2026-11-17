from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.controllers.pid_baseline import PIDBaselineController
from src.simulator.dc_env import DataCenterEnv
from src.simulator.thermodynamics import DataCenterConfig
from src.utils.logger import get_logger

logger = get_logger("dc_optimizer.dataset")


def generate_30day_dataset(
    output_path: str | Path = ROOT_DIR / "data" / "raw" / "dc_telemetry_30d.csv",
    days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    cfg = DataCenterConfig.from_yaml()
    env = DataCenterEnv(config=cfg, episode_days=days, seed=seed)
    pid = PIDBaselineController(num_racks=cfg.num_racks)

    obs, _ = env.reset(seed=seed)
    pid.reset()
    rng = np.random.default_rng(seed)

    records: list[dict[str, float | int | str]] = []
    start_ts = pd.Timestamp("2026-01-01 00:00:00")

    for step in range(env.episode_steps):
        action = pid.act(obs, env)

        dither = rng.normal(0.0, 0.05, size=3).astype(np.float32)
        dither[2] = 0.0
        exploratory_action = np.clip(action + dither, -1.0, 1.0)

        next_obs, reward, _, truncated, info = env.step(exploratory_action)

        cpu_now = obs[2 * cfg.num_racks : 3 * cfg.num_racks]
        t_in = info["t_in"]
        t_out = info["t_out"]

        row: dict[str, float | int | str] = {
            "timestamp": (start_ts + pd.Timedelta(minutes=5 * step)).isoformat(),
            "step": step,
            "t_ambient_c": round(float(info["t_ambient_c"]), 3),
            "carbon_intensity_gco2_kwh": round(float(info["carbon_intensity"]), 2),
            "t_supply_c": round(float(info["t_supply_c"]), 3),
            "airflow_frac": round(float(info["airflow_frac"]), 4),
            "consolidation_ratio": round(float(info["consolidation_ratio"]), 4),
            "mean_cpu_util": round(float(np.mean(cpu_now)), 4),
            "max_cpu_util": round(float(np.max(cpu_now)), 4),
            "mean_inlet_temp_c": round(float(np.mean(t_in)), 3),
            "max_inlet_temp_c": round(float(info["max_inlet_temp"]), 3),
            "mean_outlet_temp_c": round(float(np.mean(t_out)), 3),
            "it_power_kw": round(float(info["it_power_kw"]), 3),
            "cooling_power_kw": round(float(info["cooling_power_kw"]), 3),
            "chiller_power_kw": round(float(info["chiller_power_kw"]), 3),
            "fan_power_kw": round(float(info["fan_power_kw"]), 3),
            "total_power_kw": round(float(info["total_power_kw"]), 3),
            "pue": round(float(info["pue"]), 4),
            "cop": round(float(info["cop"]), 3),
            "active_servers": int(info["active_servers"]),
            "thermal_violations": int(info["thermal_violations"]),
            "step_kwh": round(float(info["step_kwh"]), 4),
            "step_carbon_kg": round(float(info["step_carbon_kg"]), 4),
        }

        for r in range(cfg.num_racks):
            row[f"rack_{r}_cpu"] = round(float(cpu_now[r]), 4)
            row[f"rack_{r}_t_in"] = round(float(t_in[r]), 3)
            row[f"rack_{r}_t_out"] = round(float(t_out[r]), 3)

        records.append(row)
        obs = next_obs
        if truncated:
            break

    df = pd.DataFrame(records)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    logger.info(f"Saved {len(df):,} rows ({days} days @ 5-min intervals) -> {out_file}")
    logger.info(
        f"Baseline Mean PUE: {df['pue'].mean():.3f} | "
        f"Energy: {df['step_kwh'].sum():,.1f} kWh | "
        f"CO2: {df['step_carbon_kg'].sum():,.1f} kg"
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data center telemetry dataset")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_30day_dataset(days=args.days, seed=args.seed)
