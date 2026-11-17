from __future__ import annotations

from pathlib import Path
import time
from typing import Any
import numpy as np

from src.models.safety_shield import PhysicsSafetyShield
from src.simulator.dc_env import DataCenterEnv

try:
    from stable_baselines3 import PPO, SAC
except ImportError:
    PPO = None  # type: ignore[assignment,misc]
    SAC = None  # type: ignore[assignment,misc]


class RLCoolingAgent:
    """
    Production inference wrapper combining a trained Stable-Baselines3 policy (PPO/SAC)
    with a deterministic Model-Predictive Heuristic fallback and PhysicsSafetyShield.
    """

    def __init__(
        self,
        model_path: str | Path = "artifacts/ppo_dc_optimizer.zip",
        use_shield: bool = True,
        trigger_temp_c: float = 26.4,
    ) -> None:
        self.model_path = Path(model_path)
        self.use_shield = use_shield
        self.shield = PhysicsSafetyShield(trigger_temp_c=trigger_temp_c)
        self.policy = None
        self.policy_mode = "heuristic_mpc_fallback"
        self.last_latency_ms: float = 0.0

        if self.model_path.exists() and PPO is not None:
            try:
                self.policy = PPO.load(self.model_path, device="cpu")
                self.policy_mode = "ppo_trained"
            except Exception:
                if SAC is not None:
                    self.policy = SAC.load(self.model_path, device="cpu")
                    self.policy_mode = "sac_trained"

    @staticmethod
    def _heuristic_mpc_action(obs: np.ndarray, env: DataCenterEnv) -> np.ndarray:
        n = env.cfg.num_racks
        t_in = obs[:n]
        cpu_now = obs[2 * n : 3 * n]
        max_in = float(np.max(t_in))
        mean_cpu = float(np.mean(cpu_now))

        headroom = max(0.0, 25.5 - max_in)
        t_sup = float(np.clip(18.0 + 0.75 * headroom, 16.5, 22.8))
        airflow = float(np.clip(0.38 + 0.55 * mean_cpu + max(0.0, max_in - 25.0) * 0.15, 0.35, 0.98))
        consol = float(np.clip(0.85 - 0.65 * mean_cpu, 0.15, 0.85))
        return env.normalize_action(t_sup, airflow, consol)

    def predict_action(
        self, obs: np.ndarray, env: DataCenterEnv
    ) -> tuple[np.ndarray, dict[str, Any]]:
        t0 = time.perf_counter()
        if self.policy is not None:
            raw_action, _ = self.policy.predict(obs, deterministic=True)
        else:
            raw_action = self._heuristic_mpc_action(obs, env)

        if self.use_shield:
            safe_action, overridden = self.shield.filter_action(raw_action, env)
        else:
            safe_action, overridden = raw_action, False

        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        t_sup, airflow, consol = env.rescale_action(safe_action)

        meta = {
            "policy_mode": self.policy_mode,
            "shield_overridden": overridden,
            "inference_latency_ms": round(self.last_latency_ms, 3),
            "t_supply_c": round(t_sup, 3),
            "airflow_frac": round(airflow, 4),
            "consolidation_ratio": round(consol, 4),
        }
        return safe_action, meta
