from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from stable_baselines3 import PPO

from src.controllers.pid_baseline import PIDBaselineController
from src.models.forecaster import ForecasterPredictor
from src.models.safety_shield import PhysicsSafetyShield
from src.simulator.dc_env import DataCenterEnv
from src.simulator.thermodynamics import DataCenterConfig
from src.utils.logger import get_logger

logger = get_logger("dc_optimizer.benchmark")


def rollout_controller(
    controller_name: str,
    seed: int = 999,
    episode_days: int = 7,
    ppo_model: PPO | None = None,
    forecaster_fn=None,
    use_shield: bool = False,
    inject_stress_fault: bool = False,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    cfg = DataCenterConfig.from_yaml()
    env = DataCenterEnv(
        config=cfg,
        episode_days=episode_days,
        forecast_horizon=12,
        forecaster_fn=forecaster_fn,
        seed=seed,
    )
    obs, _ = env.reset(seed=seed)

    pid = PIDBaselineController(num_racks=cfg.num_racks)
    pid.reset()
    shield = PhysicsSafetyShield(trigger_temp_c=26.4)

    rows: list[dict[str, float | int | str]] = []
    fault_step = env.episode_steps // 2

    for step in range(env.episode_steps):
        if inject_stress_fault and step == fault_step:
            env.inject_fault(chiller_efficiency_mult=0.65, load_spike=0.28)

        if controller_name == "PID Baseline":
            action = pid.act(obs, env)
            overridden = False
        else:
            assert ppo_model is not None
            action, _ = ppo_model.predict(obs, deterministic=True)
            if use_shield:
                action, overridden = shield.filter_action(action, env)
            else:
                overridden = False

        next_obs, reward, _, truncated, info = env.step(action)

        dt_hours = cfg.dt_seconds / 3600.0
        rows.append(
            {
                "controller": controller_name,
                "step": step,
                "hour": round(step * dt_hours, 2),
                "it_power_kw": float(info["it_power_kw"]),
                "cooling_power_kw": float(info["cooling_power_kw"]),
                "total_power_kw": float(info["total_power_kw"]),
                "pue": float(info["pue"]),
                "t_supply_c": float(info["t_supply_c"]),
                "airflow_frac": float(info["airflow_frac"]),
                "consolidation_ratio": float(info["consolidation_ratio"]),
                "max_inlet_temp_c": float(info["max_inlet_temp"]),
                "thermal_violations": int(info["thermal_violations"]),
                "shield_override": int(overridden),
                "step_kwh": float(info["step_kwh"]),
                "step_carbon_kg": float(info["step_carbon_kg"]),
            }
        )
        obs = next_obs
        if truncated:
            break

    ts_df = pd.DataFrame(rows)
    dt_hours = cfg.dt_seconds / 3600.0
    total_kwh = float(ts_df["step_kwh"].sum())
    cooling_kwh = float((ts_df["cooling_power_kw"] * dt_hours).sum())
    total_carbon_kg = float(ts_df["step_carbon_kg"].sum())
    cost_usd = total_kwh * 0.12

    summary: dict[str, float | int | str] = {
        "Controller": controller_name,
        "Mean PUE": round(float(ts_df["pue"].mean()), 3),
        "Total Energy (kWh)": round(total_kwh, 1),
        "Cooling Energy (kWh)": round(cooling_kwh, 1),
        "CO2 Emissions (kg)": round(total_carbon_kg, 1),
        "Est. Cost ($)": round(cost_usd, 2),
        "Peak Inlet Temp (C)": round(float(ts_df["max_inlet_temp_c"].max()), 2),
        "SLA Violations (>27C)": int((ts_df["thermal_violations"] > 0).sum()),
        "Shield Overrides": int(ts_df["shield_override"].sum()),
    }
    return summary, ts_df


def run_benchmark(
    forecaster_path: str | Path = ROOT_DIR / "artifacts" / "lstm_forecaster.pt",
    ppo_path: str | Path = ROOT_DIR / "artifacts" / "ppo_dc_optimizer.zip",
    output_dir: str | Path = ROOT_DIR / "artifacts",
) -> pd.DataFrame:
    forecaster_fn = None
    if Path(forecaster_path).exists():
        predictor = ForecasterPredictor.load(forecaster_path, device="cpu")
        forecaster_fn = predictor.make_env_forecaster_fn()

    ppo_model = PPO.load(ppo_path, device="cpu")

    scenarios = [
        ("7-Day Normal Operation", 7, False),
        ("3-Day Stress Test (Load Spike + Chiller Fault)", 3, True),
    ]

    all_summaries: list[dict[str, float | int | str]] = []
    all_timeseries: list[pd.DataFrame] = []

    for scenario_name, days, stress in scenarios:
        logger.info(f"Running benchmark scenario: {scenario_name}")
        pid_sum, pid_ts = rollout_controller(
            "PID Baseline", seed=999, episode_days=days, inject_stress_fault=stress
        )
        raw_sum, raw_ts = rollout_controller(
            "PPO + LSTM (Raw)",
            seed=999,
            episode_days=days,
            ppo_model=ppo_model,
            forecaster_fn=forecaster_fn,
            use_shield=False,
            inject_stress_fault=stress,
        )
        safe_sum, safe_ts = rollout_controller(
            "PPO + LSTM + Shield",
            seed=999,
            episode_days=days,
            ppo_model=ppo_model,
            forecaster_fn=forecaster_fn,
            use_shield=True,
            inject_stress_fault=stress,
        )

        base_kwh = float(pid_sum["Total Energy (kWh)"])
        base_cool = float(pid_sum["Cooling Energy (kWh)"])

        for s, ts in [(pid_sum, pid_ts), (raw_sum, raw_ts), (safe_sum, safe_ts)]:
            s["Scenario"] = scenario_name
            s["Total Energy Saved (%)"] = round(
                100.0 * (base_kwh - float(s["Total Energy (kWh)"])) / base_kwh, 2
            )
            s["Cooling Saved (%)"] = round(
                100.0 * (base_cool - float(s["Cooling Energy (kWh)"])) / base_cool, 2
            )
            ts["scenario"] = scenario_name
            all_summaries.append(s)
            all_timeseries.append(ts)

    summary_df = pd.DataFrame(all_summaries)
    ts_all_df = pd.concat(all_timeseries, ignore_index=True)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path / "benchmark_summary.csv", index=False)
    ts_all_df.to_csv(out_path / "benchmark_timeseries.csv", index=False)

    print("\n=== BENCHMARK COMPARISON: PID BASELINE vs. AI OPTIMIZER ===")
    print(summary_df.to_markdown(index=False))
    return summary_df


if __name__ == "__main__":
    run_benchmark()
