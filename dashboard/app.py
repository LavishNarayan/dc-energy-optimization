from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.components.kpi_charts import (
    build_actuator_telemetry_chart,
    build_benchmark_comparison_chart,
    build_power_split_chart,
    build_pue_gauge,
)
from dashboard.components.thermal_heatmap import (
    build_rack_profile_bar,
    build_rack_thermal_heatmap,
)
from src.controllers.pid_baseline import PIDBaselineController
from src.controllers.rl_agent import RLCoolingAgent
from src.evaluation.benchmark import run_benchmark
from src.models.forecaster import ForecasterPredictor
from src.simulator.dc_env import DataCenterEnv
from src.simulator.thermodynamics import DataCenterConfig

FORECASTER_PATH = ROOT_DIR / "artifacts" / "lstm_forecaster.pt"
PPO_PATH = ROOT_DIR / "artifacts" / "ppo_dc_optimizer.zip"
BENCH_SUMMARY_PATH = ROOT_DIR / "artifacts" / "benchmark_summary.csv"
BENCH_TS_PATH = ROOT_DIR / "artifacts" / "benchmark_timeseries.csv"
META_PATH = ROOT_DIR / "artifacts" / "model_metadata.json"


@st.cache_resource
def load_ai_artifacts():
    predictor = None
    forecaster_fn = None

    if FORECASTER_PATH.exists():
        predictor = ForecasterPredictor.load(FORECASTER_PATH, device="cpu")
        forecaster_fn = predictor.make_env_forecaster_fn()

    agent = RLCoolingAgent(model_path=PPO_PATH, use_shield=True)
    return predictor, forecaster_fn, agent


def init_simulation_state(seed: int = 101) -> None:
    _, forecaster_fn, agent = load_ai_artifacts()
    cfg = DataCenterConfig.from_yaml()

    ai_env = DataCenterEnv(
        config=cfg,
        episode_days=7,
        forecast_horizon=12,
        forecaster_fn=forecaster_fn,
        seed=seed,
    )
    pid_env = DataCenterEnv(
        config=cfg,
        episode_days=7,
        forecast_horizon=12,
        forecaster_fn=None,
        seed=seed,
    )

    ai_obs, _ = ai_env.reset(seed=seed)
    pid_obs, _ = pid_env.reset(seed=seed)

    pid_ctrl = PIDBaselineController(num_racks=cfg.num_racks)
    pid_ctrl.reset()
    agent.shield.reset_stats()

    st.session_state.ai_env = ai_env
    st.session_state.pid_env = pid_env
    st.session_state.ai_obs = ai_obs
    st.session_state.pid_obs = pid_obs
    st.session_state.pid_ctrl = pid_ctrl
    st.session_state.sim_step = 0
    st.session_state.history = []
    st.session_state.last_ai_info = ai_env.last_metrics
    st.session_state.last_pid_info = pid_env.last_metrics
    st.session_state.auto_play = False


def advance_simulation(num_steps: int = 1, use_shield: bool = True) -> None:
    _, _, agent = load_ai_artifacts()
    agent.use_shield = use_shield

    ai_env: DataCenterEnv = st.session_state.ai_env
    pid_env: DataCenterEnv = st.session_state.pid_env
    pid_ctrl: PIDBaselineController = st.session_state.pid_ctrl

    start_ts = pd.Timestamp("2026-11-15 00:00:00")

    for _ in range(num_steps):
        if st.session_state.sim_step >= ai_env.episode_steps:
            st.session_state.auto_play = False
            break

        ai_action, meta = agent.predict_action(st.session_state.ai_obs, ai_env)
        pid_action = pid_ctrl.act(st.session_state.pid_obs, pid_env)

        ai_obs, _, _, truncated, ai_info = ai_env.step(ai_action)
        pid_obs, _, _, _, pid_info = pid_env.step(pid_action)

        st.session_state.ai_obs = ai_obs
        st.session_state.pid_obs = pid_obs
        st.session_state.last_ai_info = ai_info
        st.session_state.last_pid_info = pid_info

        step_idx = st.session_state.sim_step
        sim_time = (start_ts + pd.Timedelta(minutes=5 * step_idx)).strftime("%d %b %H:%M")

        st.session_state.history.append(
            {
                "step": step_idx,
                "sim_time": sim_time,
                "it_power_kw": float(ai_info["it_power_kw"]),
                "chiller_power_kw": float(ai_info["chiller_power_kw"]),
                "fan_power_kw": float(ai_info["fan_power_kw"]),
                "cooling_power_kw": float(ai_info["cooling_power_kw"]),
                "total_power_kw": float(ai_info["total_power_kw"]),
                "pid_total_power_kw": float(pid_info["total_power_kw"]),
                "pue": float(ai_info["pue"]),
                "pid_pue": float(pid_info["pue"]),
                "t_supply_c": float(ai_info["t_supply_c"]),
                "airflow_frac": float(ai_info["airflow_frac"]),
                "consolidation_ratio": float(ai_info["consolidation_ratio"]),
                "max_inlet_temp_c": float(ai_info["max_inlet_temp"]),
                "pid_max_inlet_temp_c": float(pid_info["max_inlet_temp"]),
                "thermal_violations": int(ai_info["thermal_violations"]),
                "shield_override": int(bool(meta["shield_overridden"])),
                "latency_ms": float(meta["inference_latency_ms"]),
                "ai_kwh": float(ai_info["step_kwh"]),
                "pid_kwh": float(pid_info["step_kwh"]),
                "ai_carbon_kg": float(ai_info["step_carbon_kg"]),
                "pid_carbon_kg": float(pid_info["step_carbon_kg"]),
            }
        )
        st.session_state.sim_step += 1
        if truncated:
            st.session_state.auto_play = False
            break


def page_live_command_center() -> None:
    st.title("⚡ AI Data Center Energy Optimization Command Center")

    _, _, agent = load_ai_artifacts()
    mode_label = (
        "🟢 Active Policy: Trained PPO + LSTM Forecaster"
        if agent.policy is not None
        else "🟡 Active Policy: Model-Predictive Heuristic (Run train_agent.py to load PPO weights)"
    )
    st.caption(f"{mode_label} | Control Interval: 5 min | ASHRAE Max Inlet Ceiling: 27.0 °C")

    with st.sidebar:
        st.subheader("🎛️ Simulation Controls")
        use_shield = st.toggle("Physics Safety Shield (< 27°C)", value=True)

        c1, c2 = st.columns(2)
        if c1.button("Step (+5m)", use_container_width=True):
            advance_simulation(num_steps=1, use_shield=use_shield)
        if c2.button("Run (+1 hr)", use_container_width=True):
            advance_simulation(num_steps=12, use_shield=use_shield)

        c3, c4 = st.columns(2)
        if c3.button("Run (+6 hrs)", use_container_width=True):
            advance_simulation(num_steps=72, use_shield=use_shield)
        if c4.button("Reset Twin", use_container_width=True):
            init_simulation_state(seed=int(np.random.randint(1, 9999)))
            st.rerun()

        st.session_state.auto_play = st.toggle(
            "Auto-Play Live Stream", value=st.session_state.auto_play
        )

        st.divider()
        st.subheader("🔥 Chaos & Fault Injection")
        if st.button("⚡ Inject Compute Spike (+30%)", use_container_width=True):
            st.session_state.ai_env.inject_fault(
                chiller_efficiency_mult=st.session_state.ai_env.physics.chiller_efficiency_mult,
                load_spike=0.30,
            )
            st.session_state.pid_env.inject_fault(
                chiller_efficiency_mult=st.session_state.pid_env.physics.chiller_efficiency_mult,
                load_spike=0.30,
            )
            st.toast("Injected +30% IT burst load across racks for next 60 minutes!")

        if st.button("❄️ Simulate Chiller Fault (COP -35%)", use_container_width=True):
            st.session_state.ai_env.inject_fault(chiller_efficiency_mult=0.65, load_spike=0.0)
            st.session_state.pid_env.inject_fault(chiller_efficiency_mult=0.65, load_spike=0.0)
            st.toast("Chiller efficiency degraded to 65%!", icon="⚠️")

        if st.button("✅ Restore Healthy Chiller", use_container_width=True):
            st.session_state.ai_env.inject_fault(chiller_efficiency_mult=1.0, load_spike=0.0)
            st.session_state.pid_env.inject_fault(chiller_efficiency_mult=1.0, load_spike=0.0)
            st.toast("Chiller plant restored to 100% nominal COP.")

    if not st.session_state.history:
        advance_simulation(num_steps=12, use_shield=use_shield)

    hist_df = pd.DataFrame(st.session_state.history)
    latest = hist_df.iloc[-1]
    ai_info = st.session_state.last_ai_info

    ai_kwh_total = float(hist_df["ai_kwh"].sum())
    pid_kwh_total = float(hist_df["pid_kwh"].sum())
    saved_kwh = pid_kwh_total - ai_kwh_total
    saved_pct = (100.0 * saved_kwh / pid_kwh_total) if pid_kwh_total > 0 else 0.0
    saved_usd = saved_kwh * 0.12
    saved_co2_kg = float(hist_df["pid_carbon_kg"].sum() - hist_df["ai_carbon_kg"].sum())

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(
        "Current AI PUE",
        f"{latest['pue']:.3f}",
        delta=f"{latest['pue'] - latest['pid_pue']:.3f} vs PID",
        delta_color="inverse",
    )
    k2.metric(
        "Total Facility Power",
        f"{latest['total_power_kw']:.1f} kW",
        delta=f"{latest['total_power_kw'] - latest['pid_total_power_kw']:.1f} kW",
        delta_color="inverse",
    )
    k3.metric(
        "Peak Rack Inlet",
        f"{latest['max_inlet_temp_c']:.2f} °C",
        delta=f"{latest['max_inlet_temp_c'] - 27.0:.2f} °C margin",
        delta_color="inverse",
    )
    k4.metric(
        "Energy Saved",
        f"{saved_kwh:,.1f} kWh",
        delta=f"{saved_pct:.1f}% reduction",
    )
    k5.metric(
        "Est. Cost & CO₂ Saved",
        f"${saved_usd:,.2f}",
        delta=f"{saved_co2_kg:,.1f} kg CO₂",
    )
    k6.metric(
        "Safety Shield Triggers",
        f"{int(hist_df['shield_override'].sum())}",
        delta=f"SLA Breaches: {int((hist_df['thermal_violations'] > 0).sum())}",
        delta_color="inverse",
    )

    st.divider()

    col_gauge, col_heat = st.columns([1, 2.3])
    with col_gauge:
        st.plotly_chart(
            build_pue_gauge(float(latest["pue"]), float(latest["pid_pue"])),
            use_container_width=True,
        )
    with col_heat:
        n = st.session_state.ai_env.cfg.num_racks
        cpu_now = st.session_state.ai_obs[2 * n : 3 * n]
        st.plotly_chart(
            build_rack_thermal_heatmap(
                t_in=ai_info["t_in"],
                t_out=ai_info["t_out"],
                cpu_util=cpu_now,
                rack_it_kw=ai_info["rack_it_kw"],
            ),
            use_container_width=True,
        )

    c_left, c_right = st.columns(2)
    with c_left:
        st.plotly_chart(build_power_split_chart(hist_df), use_container_width=True)
    with c_right:
        st.plotly_chart(build_actuator_telemetry_chart(hist_df), use_container_width=True)

    st.plotly_chart(
        build_rack_profile_bar(t_in=ai_info["t_in"], t_out=ai_info["t_out"]),
        use_container_width=True,
    )

    if st.session_state.auto_play:
        time.sleep(0.6)
        advance_simulation(num_steps=1, use_shield=use_shield)
        st.rerun()


def page_benchmark_analytics() -> None:
    st.title("📊 RL Policy vs. PID Baseline Benchmark Analytics")
    st.write(
        "Evaluates **PID Baseline**, **PPO + LSTM (Raw)**, and **PPO + LSTM + Physics Safety Shield** "
        "across normal multi-day operation and severe chiller/workload fault scenarios."
    )

    if not BENCH_SUMMARY_PATH.exists() or not BENCH_TS_PATH.exists():
        st.warning("Benchmark CSV artifacts not found in `artifacts/`.")
        if PPO_PATH.exists():
            if st.button("▶️ Run Full Benchmark Suite Now"):
                with st.spinner("Running 7-day normal and 3-day stress-test rollouts..."):
                    run_benchmark()
                st.rerun()
        else:
            st.info(
                "Train the PPO agent first via `python scripts/train_agent.py` and "
                "`python src/evaluation/benchmark.py` (or run `run_all.bat`)."
            )
        return

    summary_df = pd.read_csv(BENCH_SUMMARY_PATH)
    ts_df = pd.read_csv(BENCH_TS_PATH)

    scenarios = summary_df["Scenario"].unique().tolist()
    selected_scenario = st.selectbox("Select Evaluation Scenario", scenarios)

    sub_summary = summary_df[summary_df["Scenario"] == selected_scenario]
    st.subheader("Executive KPI Comparison Table")
    st.dataframe(sub_summary, use_container_width=True, hide_index=True)

    st.plotly_chart(
        build_benchmark_comparison_chart(summary_df, selected_scenario),
        use_container_width=True,
    )

    sub_ts = ts_df[ts_df["scenario"] == selected_scenario]
    st.subheader("Episode Trajectory Comparison (PUE & Peak Inlet Temperature)")
    c1, c2 = st.columns(2)
    with c1:
        pue_piv = sub_ts.pivot(index="hour", columns="controller", values="pue")
        st.line_chart(pue_piv, height=300)
    with c2:
        temp_piv = sub_ts.pivot(index="hour", columns="controller", values="max_inlet_temp_c")
        st.line_chart(temp_piv, height=300)


def page_system_inspector() -> None:
    st.title("🧠 System Architecture & Model Inspector")

    predictor, _, agent = load_ai_artifacts()
    cfg = DataCenterConfig.from_yaml()

    col1, col2, col3 = st.columns(3)
    col1.metric("LSTM Forecaster", "Loaded" if predictor else "Not Trained")
    col2.metric("RL Policy Status", agent.policy_mode)
    col3.metric("Safety Shield Ceiling", f"{cfg.t_inlet_safe_max_c:.1f} °C")

    if META_PATH.exists():
        st.subheader("Model Registry & Validation Telemetry (`artifacts/model_metadata.json`)")
        st.json(json.loads(META_PATH.read_text(encoding="utf-8")))

    st.subheader("Digital Twin Physical Plant Specifications")
    specs_df = pd.DataFrame(
        [
            {"Parameter": "Server Racks × Servers/Rack", "Value": f"{cfg.num_racks} × {cfg.servers_per_rack} ({cfg.num_racks * cfg.servers_per_rack} servers)"},
            {"Parameter": "Server Power Envelope (Sleep / Idle / Peak)", "Value": f"{cfg.p_server_sleep_w} W / {cfg.p_server_idle_w} W / {cfg.p_server_peak_w} W"},
            {"Parameter": "CRAC Supply Air Setpoint Range", "Value": f"{cfg.t_supply_min_c} °C – {cfg.t_supply_max_c} °C"},
            {"Parameter": "CRAC Airflow Modulation Range", "Value": f"{cfg.airflow_min_frac * 100:.0f}% – {cfg.airflow_max_frac * 100:.0f}% (Cubic Fan Law)"},
            {"Parameter": "Nominal Chiller COP", "Value": f"{cfg.cop_nominal:.2f} (Carnot supply & ambient adjusted)"},
            {"Parameter": "ASHRAE TC 9.9 Safe / Critical Limits", "Value": f"{cfg.t_inlet_safe_max_c:.1f} °C / {cfg.t_inlet_critical_c:.1f} °C"},
        ]
    )
    st.dataframe(specs_df, use_container_width=True, hide_index=True)

    if predictor is not None and "ai_env" in st.session_state:
        st.subheader("Live 60-Minute LSTM Forecast Horizon (+5 min to +60 min)")
        env: DataCenterEnv = st.session_state.ai_env
        if len(env.obs_history) >= 24:
            hist_win = np.stack(env.obs_history[-24:], axis=0)
            fut_cpu, fut_amb = predictor.predict_from_obs_window(hist_win)
            fc_df = pd.DataFrame(
                {
                    "Horizon (min)": [(i + 1) * 5 for i in range(len(fut_amb))],
                    "Predicted Mean CPU (%)": np.round(fut_cpu.mean(axis=1) * 100.0, 2),
                    "Predicted Outdoor Temp (°C)": np.round(fut_amb, 2),
                }
            )
            st.dataframe(fc_df, use_container_width=True, hide_index=True)


st.set_page_config(
    page_title="AI Data Center Energy Optimizer",
    page_icon="⚡",
    layout="wide",
)

if "ai_env" not in st.session_state:
    init_simulation_state(seed=101)

pg = st.navigation(
    {
        "Operations": [
            st.Page(page_live_command_center, title="Live Command Center", icon="⚡", default=True),
            st.Page(page_benchmark_analytics, title="RL vs. PID Benchmarks", icon="📊"),
        ],
        "Engineering": [
            st.Page(page_system_inspector, title="System & Model Inspector", icon="🧠"),
        ],
    }
)
pg.run()
