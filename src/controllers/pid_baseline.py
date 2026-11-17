from __future__ import annotations

import numpy as np
from src.simulator.dc_env import DataCenterEnv


class PIDBaselineController:
    """
    Industry-standard baseline data center controller:
      - Fixed conservative CRAC supply air setpoint (18.5 °C)
      - Closed-loop PID regulating CRAC fan speed to hold peak cold-aisle inlet <= 24.0 °C
      - No dynamic server consolidation (consolidation_ratio = 0.0)
    """

    def __init__(
        self,
        num_racks: int = 10,
        target_inlet_c: float = 24.0,
        fixed_supply_c: float = 18.5,
        kp: float = 0.14,
        ki: float = 0.02,
        kd: float = 0.05,
    ) -> None:
        self.num_racks = num_racks
        self.target_inlet_c = target_inlet_c
        self.fixed_supply_c = fixed_supply_c
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral_err: float = 0.0
        self.prev_err: float = 0.0
        self.base_airflow: float = 0.65

    def reset(self) -> None:
        self.integral_err = 0.0
        self.prev_err = 0.0

    def compute_setpoints(self, obs: np.ndarray) -> tuple[float, float, float]:
        t_in = obs[: self.num_racks]
        max_inlet = float(np.max(t_in))

        err = max_inlet - self.target_inlet_c
        self.integral_err = float(np.clip(self.integral_err + err, -10.0, 10.0))
        deriv_err = err - self.prev_err
        self.prev_err = err

        delta_airflow = (
            self.kp * err + self.ki * self.integral_err + self.kd * deriv_err
        )
        airflow_frac = float(np.clip(self.base_airflow + delta_airflow, 0.35, 1.00))
        return self.fixed_supply_c, airflow_frac, 0.0

    def act(self, obs: np.ndarray, env: DataCenterEnv) -> np.ndarray:
        t_sup, airflow, consol = self.compute_setpoints(obs)
        return env.normalize_action(t_sup, airflow, consol)
