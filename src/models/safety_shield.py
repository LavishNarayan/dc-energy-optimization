from __future__ import annotations

import copy
from typing import Any
import gymnasium as gym
import numpy as np

from src.simulator.dc_env import DataCenterEnv
from src.simulator.thermodynamics import DataCenterPhysics


class PhysicsSafetyShield:
    """
    Model-based safety filter that intercepts RL actions prior to plant actuation.
    Uses a one-step lookahead thermodynamic copy to verify whether the proposed
    setpoints keep all rack inlet temperatures below ASHRAE threshold (27.0 °C).
    """

    def __init__(
        self,
        trigger_temp_c: float = 26.4,
        max_correction_iters: int = 8,
    ) -> None:
        self.trigger_temp_c = trigger_temp_c
        self.max_correction_iters = max_correction_iters
        self.override_count: int = 0
        self.total_steps: int = 0

    def reset_stats(self) -> None:
        self.override_count = 0
        self.total_steps = 0

    def _simulate_next_max_inlet(
        self,
        physics: DataCenterPhysics,
        rack_cpu_util: np.ndarray,
        t_ambient_c: float,
        t_supply_c: float,
        airflow_frac: float,
        consolidation_ratio: float,
    ) -> float:
        sandbox = copy.deepcopy(physics)
        metrics = sandbox.step(
            rack_cpu_util=rack_cpu_util,
            t_ambient_c=t_ambient_c,
            t_supply_c=t_supply_c,
            airflow_frac=airflow_frac,
            consolidation_ratio=consolidation_ratio,
        )
        return float(metrics["max_inlet_temp"])

    def filter_action(
        self,
        action: np.ndarray,
        env: DataCenterEnv,
    ) -> tuple[np.ndarray, bool]:
        """
        Verifies and, if necessary, minimally modifies `action` so predicted
        t_in(t+1) stays <= trigger_temp_c. Returns (safe_action, was_overridden).
        """
        self.total_steps += 1
        t_sup, airflow, consol = env.rescale_action(action)
        cpu_now = env.trajectories["cpu_util"][env.step_idx]
        t_amb_now = float(env.trajectories["t_ambient"][env.step_idx])

        pred_max_inlet = self._simulate_next_max_inlet(
            env.physics, cpu_now, t_amb_now, t_sup, airflow, consol
        )

        if pred_max_inlet <= self.trigger_temp_c:
            return action, False

        safe_sup, safe_air, safe_consol = t_sup, airflow, consol
        for _ in range(self.max_correction_iters):
            safe_sup = max(env.cfg.t_supply_min_c, safe_sup - 1.2)
            safe_air = min(env.cfg.airflow_max_frac, safe_air + 0.12)
            safe_consol = max(0.0, safe_consol - 0.15)

            pred_max_inlet = self._simulate_next_max_inlet(
                env.physics, cpu_now, t_amb_now, safe_sup, safe_air, safe_consol
            )
            if pred_max_inlet <= self.trigger_temp_c:
                break

        self.override_count += 1
        safe_action = env.normalize_action(safe_sup, safe_air, safe_consol)
        return safe_action, True


class SafeDataCenterWrapper(gym.ActionWrapper):
    """Gymnasium ActionWrapper applying PhysicsSafetyShield transparently on every step."""

    def __init__(self, env: DataCenterEnv, trigger_temp_c: float = 26.4) -> None:
        super().__init__(env)
        self.shield = PhysicsSafetyShield(trigger_temp_c=trigger_temp_c)
        self.last_overridden: bool = False

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        self.shield.reset_stats()
        self.last_overridden = False
        return self.env.reset(**kwargs)

    def action(self, action: np.ndarray) -> np.ndarray:
        base_env: DataCenterEnv = self.env.unwrapped  # type: ignore[assignment]
        safe_action, overridden = self.shield.filter_action(action, base_env)
        self.last_overridden = overridden
        return safe_action
