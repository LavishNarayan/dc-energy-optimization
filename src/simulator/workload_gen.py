from __future__ import annotations

import numpy as np


class TelemetryProfileGenerator:
    """
    Generates realistic 5-minute time-series trajectories for:
      1. Multi-rack IT CPU utilization (diurnal enterprise curves + burst spikes)
      2. Outdoor dry-bulb ambient temperature (°C)
      3. Marginal grid carbon intensity (gCO2/kWh)
    """

    def __init__(self, num_racks: int = 10, dt_minutes: int = 5, seed: int = 42) -> None:
        self.num_racks = num_racks
        self.dt_minutes = dt_minutes
        self.steps_per_day = (24 * 60) // dt_minutes
        self.rng = np.random.default_rng(seed)

        self.rack_weights = np.linspace(0.85, 1.18, num_racks)
        self.rng.shuffle(self.rack_weights)

    def generate(self, num_steps: int) -> dict[str, np.ndarray]:
        t = np.arange(num_steps, dtype=np.float64)
        hour_of_day = (t * self.dt_minutes / 60.0) % 24.0
        day_of_week = ((t // self.steps_per_day) % 7).astype(int)

        diurnal = 0.52 + 0.24 * np.sin(2.0 * np.pi * (hour_of_day - 8.5) / 24.0)
        weekend_mask = np.where((day_of_week == 5) | (day_of_week == 6), 0.78, 1.0)
        base_util = diurnal * weekend_mask

        cpu_util = base_util[:, None] * self.rack_weights[None, :]
        cpu_util += self.rng.normal(0.0, 0.03, size=(num_steps, self.num_racks))

        burst_starts = self.rng.choice(
            np.arange(max(1, num_steps - 12)),
            size=max(1, num_steps // 140),
            replace=False,
        )
        for start in burst_starts:
            duration = int(self.rng.integers(4, 12))
            target_racks = self.rng.choice(
                self.num_racks, size=max(2, self.num_racks // 3), replace=False
            )
            cpu_util[start : start + duration, target_racks] += self.rng.uniform(0.18, 0.32)

        cpu_util = np.clip(cpu_util, 0.08, 0.98)

        weather_wave = 3.5 * np.sin(2.0 * np.pi * t / (self.steps_per_day * 5.0))
        t_ambient = (
            26.0
            + 6.5 * np.sin(2.0 * np.pi * (hour_of_day - 9.0) / 24.0)
            + weather_wave
            + self.rng.normal(0.0, 0.25, size=num_steps)
        )

        carbon_intensity = (
            420.0
            - 110.0 * np.exp(-0.5 * ((hour_of_day - 13.0) / 3.0) ** 2)
            + 75.0 * np.exp(-0.5 * ((hour_of_day - 19.5) / 2.5) ** 2)
            + self.rng.normal(0.0, 8.0, size=num_steps)
        )
        carbon_intensity = np.clip(carbon_intensity, 180.0, 650.0)

        return {
            "cpu_util": cpu_util.astype(np.float32),
            "t_ambient": t_ambient.astype(np.float32),
            "carbon_intensity": carbon_intensity.astype(np.float32),
        }
