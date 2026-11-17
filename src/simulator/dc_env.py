from __future__ import annotations

from typing import Any, Callable
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.simulator.thermodynamics import DataCenterConfig, DataCenterPhysics
from src.simulator.workload_gen import TelemetryProfileGenerator
from src.utils.config_loader import load_yaml_config


class DataCenterEnv(gym.Env):
    """
    Custom Gymnasium environment for AI-Based Data Center Energy Optimization.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: DataCenterConfig | None = None,
        episode_days: int = 1,
        forecast_horizon: int = 12,
        reward_weights: tuple[float, float, float, float] | None = None,
        forecaster_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.cfg = config or DataCenterConfig.from_yaml()
        self.physics = DataCenterPhysics(self.cfg)
        self.n = self.cfg.num_racks
        self.horizon = forecast_horizon

        if reward_weights is None:
            raw_cfg = load_yaml_config("configs/env_config.yaml").get("reward_weights", {})
            reward_weights = (
                float(raw_cfg.get("w_energy", 1.0)),
                float(raw_cfg.get("w_carbon", 0.4)),
                float(raw_cfg.get("w_thermal", 2.5)),
                float(raw_cfg.get("w_smoothness", 0.15)),
            )
        self.w1, self.w2, self.w3, self.w4 = reward_weights
        self.forecaster_fn = forecaster_fn
        self.base_seed = seed

        steps_per_day = int((24 * 3600) // self.cfg.dt_seconds)
        self.episode_steps = episode_days * steps_per_day
        self.profile_gen = TelemetryProfileGenerator(
            num_racks=self.n,
            dt_minutes=int(self.cfg.dt_seconds // 60),
            seed=seed,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        obs_dim = (3 * self.n) + 4 + (self.horizon * self.n) + self.horizon + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.trajectories: dict[str, np.ndarray] = {}
        self.step_idx: int = 0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.last_metrics: dict[str, Any] = {}
        self.obs_history: list[np.ndarray] = []

    def rescale_action(self, action: np.ndarray) -> tuple[float, float, float]:
        """Maps normalized [-1, 1]^3 RL action to physical actuator setpoints."""
        a = np.clip(action, -1.0, 1.0)
        t_sup = self.cfg.t_supply_min_c + 0.5 * (a[0] + 1.0) * (
            self.cfg.t_supply_max_c - self.cfg.t_supply_min_c
        )
        airflow = self.cfg.airflow_min_frac + 0.5 * (a[1] + 1.0) * (
            self.cfg.airflow_max_frac - self.cfg.airflow_min_frac
        )
        consol = 0.5 * (a[2] + 1.0)
        return float(t_sup), float(airflow), float(consol)

    def normalize_action(self, t_sup: float, airflow: float, consol: float) -> np.ndarray:
        """Inverse mapping from physical setpoints to [-1, 1]^3."""
        a0 = 2.0 * (t_sup - self.cfg.t_supply_min_c) / (
            self.cfg.t_supply_max_c - self.cfg.t_supply_min_c
        ) - 1.0
        a1 = 2.0 * (airflow - self.cfg.airflow_min_frac) / (
            self.cfg.airflow_max_frac - self.cfg.airflow_min_frac
        ) - 1.0
        a2 = 2.0 * consol - 1.0
        return np.clip(np.array([a0, a1, a2], dtype=np.float32), -1.0, 1.0)

    def _get_forecasts(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns 60-minute ahead CPU load and ambient temperature trajectories."""
        if self.forecaster_fn is not None and len(self.obs_history) >= 24:
            hist_window = np.stack(self.obs_history[-24:], axis=0)
            return self.forecaster_fn(hist_window)

        end_idx = self.step_idx + 1 + self.horizon
        future_cpu = self.trajectories["cpu_util"][self.step_idx + 1 : end_idx].copy()
        future_amb = self.trajectories["t_ambient"][self.step_idx + 1 : end_idx].copy()

        if len(future_cpu) < self.horizon:
            pad_len = self.horizon - len(future_cpu)
            future_cpu = np.vstack([future_cpu, np.repeat(future_cpu[-1:], pad_len, axis=0)])
            future_amb = np.concatenate([future_amb, np.repeat(future_amb[-1:], pad_len)])

        horizon_scale = np.linspace(0.01, 0.05, self.horizon)[:, None]
        future_cpu = np.clip(
            future_cpu + self.np_random.normal(0.0, horizon_scale, size=future_cpu.shape),
            0.05,
            1.0,
        )
        future_amb = future_amb + self.np_random.normal(0.0, 0.35, size=future_amb.shape)
        return future_cpu.astype(np.float32), future_amb.astype(np.float32)

    def _build_obs(self) -> np.ndarray:
        cpu_now = self.trajectories["cpu_util"][self.step_idx]
        t_amb_now = float(self.trajectories["t_ambient"][self.step_idx])
        c_grid_now = float(self.trajectories["carbon_intensity"][self.step_idx])
        future_cpu, future_amb = self._get_forecasts()

        obs = np.concatenate(
            [
                self.physics.t_in.astype(np.float32),
                self.physics.t_out.astype(np.float32),
                cpu_now.astype(np.float32),
                np.array(
                    [
                        self.last_metrics["it_power_kw"],
                        self.last_metrics["cooling_power_kw"],
                        t_amb_now,
                        c_grid_now,
                    ],
                    dtype=np.float32,
                ),
                future_cpu.flatten(),
                future_amb.flatten(),
                self.prev_action.astype(np.float32),
            ]
        )
        return obs

    def _compute_reward(self, metrics: dict[str, Any], action: np.ndarray, c_grid: float) -> float:
        p_total = float(metrics["total_power_kw"])
        t_in = np.asarray(metrics["t_in"], dtype=np.float64)

        energy_term = p_total / self.cfg.p_max_total_kw
        carbon_term = (p_total * c_grid) / (self.cfg.p_max_total_kw * 450.0)

        over_safe = np.maximum(0.0, t_in - self.cfg.t_inlet_safe_max_c)
        critical_hits = (t_in >= self.cfg.t_inlet_critical_c).astype(np.float64)
        thermal_loss = float(np.sum(over_safe ** 2 + 50.0 * critical_hits))

        smooth_loss = float(np.sum((action - self.prev_action) ** 2))

        reward = -(
            self.w1 * energy_term
            + self.w2 * carbon_term
            + self.w3 * thermal_loss
            + self.w4 * smooth_loss
        )
        return float(reward)

    def inject_fault(self, chiller_efficiency_mult: float = 0.65, load_spike: float = 0.0) -> None:
        """Hook for chaos/stress testing via API or Streamlit dashboard."""
        self.physics.chiller_efficiency_mult = float(np.clip(chiller_efficiency_mult, 0.2, 1.0))
        if load_spike > 0.0:
            end = min(len(self.trajectories["cpu_util"]), self.step_idx + 12)
            self.trajectories["cpu_util"][self.step_idx : end] = np.clip(
                self.trajectories["cpu_util"][self.step_idx : end] + load_spike, 0.0, 0.99
            )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        ep_seed = int(self.np_random.integers(0, 1_000_000)) if seed is None else seed
        self.profile_gen = TelemetryProfileGenerator(
            num_racks=self.n,
            dt_minutes=int(self.cfg.dt_seconds // 60),
            seed=ep_seed,
        )

        self.trajectories = self.profile_gen.generate(self.episode_steps + self.horizon + 2)
        self.step_idx = 0
        self.physics.reset(t_supply_init=19.0, rng=self.np_random)
        self.prev_action = self.normalize_action(19.0, 0.65, 0.0)
        self.obs_history = []

        self.last_metrics = self.physics.step(
            rack_cpu_util=self.trajectories["cpu_util"][0],
            t_ambient_c=float(self.trajectories["t_ambient"][0]),
            t_supply_c=19.0,
            airflow_frac=0.65,
            consolidation_ratio=0.0,
        )
        obs = self._build_obs()
        self.obs_history.append(obs)
        return obs, {"pue": self.last_metrics["pue"]}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        t_sup, airflow, consol = self.rescale_action(action)

        cpu_now = self.trajectories["cpu_util"][self.step_idx]
        t_amb_now = float(self.trajectories["t_ambient"][self.step_idx])
        c_grid_now = float(self.trajectories["carbon_intensity"][self.step_idx])

        metrics = self.physics.step(
            rack_cpu_util=cpu_now,
            t_ambient_c=t_amb_now,
            t_supply_c=t_sup,
            airflow_frac=airflow,
            consolidation_ratio=consol,
        )

        reward = self._compute_reward(metrics, action, c_grid_now)
        self.prev_action = action.copy()
        self.last_metrics = metrics
        self.step_idx += 1

        terminated = False
        truncated = self.step_idx >= self.episode_steps

        dt_hours = self.cfg.dt_seconds / 3600.0
        step_kwh = float(metrics["total_power_kw"]) * dt_hours
        step_carbon_kg = (step_kwh * c_grid_now) / 1000.0

        info = {
            **metrics,
            "t_supply_c": t_sup,
            "airflow_frac": airflow,
            "consolidation_ratio": consol,
            "t_ambient_c": t_amb_now,
            "carbon_intensity": c_grid_now,
            "step_kwh": step_kwh,
            "step_carbon_kg": step_carbon_kg,
        }

        obs = self._build_obs()
        self.obs_history.append(obs)
        return obs, reward, terminated, truncated, info
