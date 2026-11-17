from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from src.utils.config_loader import load_yaml_config


@dataclass
class DataCenterConfig:
    """Physical and thermodynamic parameters for a multi-rack data center."""

    num_racks: int = 10
    servers_per_rack: int = 20
    dt_seconds: float = 300.0          # 5-minute control interval
    substeps: int = 15                 # 20-second ODE integration sub-steps

    # Server power parameters (Watts per server)
    p_server_idle_w: float = 120.0
    p_server_peak_w: float = 450.0
    p_server_sleep_w: float = 15.0
    cpu_power_exponent: float = 1.3    # Non-linear dynamic power curve
    max_consolidated_util: float = 0.85

    # Thermal capacitance & airflow properties
    rack_thermal_cap_kj_k: float = 45.0   # Combined metal + air thermal mass per rack
    air_specific_heat_kj_kgk: float = 1.005
    max_airflow_per_rack_kg_s: float = 1.60

    # CRAC & Chiller plant parameters
    t_supply_min_c: float = 16.0
    t_supply_max_c: float = 24.0
    airflow_min_frac: float = 0.30
    airflow_max_frac: float = 1.00
    cop_nominal: float = 4.0
    cop_supply_slope: float = 0.08     # COP gain per °C increase in supply setpoint
    cop_ambient_slope: float = 0.06    # COP loss per °C increase in outdoor ambient temp
    p_fan_max_kw: float = 18.0         # Rated blower power at 100% airflow
    p_pump_base_kw: float = 2.0        # Constant chilled-water circulation pump draw

    # ASHRAE TC 9.9 thermal thresholds (°C)
    t_inlet_safe_max_c: float = 27.0
    t_inlet_critical_c: float = 30.0

    # Derived normalization constants
    p_max_total_kw: float = field(init=False)

    def __post_init__(self) -> None:
        max_it_kw = (self.num_racks * self.servers_per_rack * self.p_server_peak_w) / 1000.0
        max_cool_kw = (max_it_kw / 2.2) + self.p_fan_max_kw + self.p_pump_base_kw
        self.p_max_total_kw = max_it_kw + max_cool_kw

    @classmethod
    def from_yaml(cls, rel_path: str = "configs/env_config.yaml") -> DataCenterConfig:
        raw = load_yaml_config(rel_path)
        env_s = raw.get("environment", {})
        srv_s = raw.get("servers", {})
        hvac_s = raw.get("crac_and_chiller", {})
        lim_s = raw.get("ashrae_limits", {})
        return cls(
            num_racks=int(env_s.get("num_racks", 10)),
            servers_per_rack=int(env_s.get("servers_per_rack", 20)),
            dt_seconds=float(env_s.get("dt_seconds", 300.0)),
            substeps=int(env_s.get("substeps", 15)),
            p_server_idle_w=float(srv_s.get("p_server_idle_w", 120.0)),
            p_server_peak_w=float(srv_s.get("p_server_peak_w", 450.0)),
            p_server_sleep_w=float(srv_s.get("p_server_sleep_w", 15.0)),
            cpu_power_exponent=float(srv_s.get("cpu_power_exponent", 1.3)),
            max_consolidated_util=float(srv_s.get("max_consolidated_util", 0.85)),
            t_supply_min_c=float(hvac_s.get("t_supply_min_c", 16.0)),
            t_supply_max_c=float(hvac_s.get("t_supply_max_c", 24.0)),
            airflow_min_frac=float(hvac_s.get("airflow_min_frac", 0.30)),
            airflow_max_frac=float(hvac_s.get("airflow_max_frac", 1.00)),
            cop_nominal=float(hvac_s.get("cop_nominal", 4.0)),
            cop_supply_slope=float(hvac_s.get("cop_supply_slope", 0.08)),
            cop_ambient_slope=float(hvac_s.get("cop_ambient_slope", 0.06)),
            p_fan_max_kw=float(hvac_s.get("p_fan_max_kw", 18.0)),
            p_pump_base_kw=float(hvac_s.get("p_pump_base_kw", 2.0)),
            t_inlet_safe_max_c=float(lim_s.get("t_inlet_safe_max_c", 27.0)),
            t_inlet_critical_c=float(lim_s.get("t_inlet_critical_c", 30.0)),
        )


class DataCenterPhysics:
    """Coupled IT workload, airflow recirculation, and CRAC thermodynamic plant."""

    def __init__(self, config: DataCenterConfig | None = None) -> None:
        self.cfg = config or DataCenterConfig.from_yaml()
        self.n = self.cfg.num_racks

        # Hot-air recirculation fraction per rack (edge racks experience higher leakage)
        self.recirc_factor = np.full(self.n, 0.13, dtype=np.float64)
        self.recirc_factor[0] = 0.22
        self.recirc_factor[-1] = 0.22

        # Spatial mixing weights for recirculated hot-aisle exhaust
        idx = np.arange(self.n)
        dist = np.abs(idx[:, None] - idx[None, :])
        weights = np.exp(-0.8 * dist)
        self.mixing_matrix = weights / weights.sum(axis=1, keepdims=True)

        # Dynamic state variables
        self.t_in = np.full(self.n, 20.0, dtype=np.float64)
        self.t_out = np.full(self.n, 26.0, dtype=np.float64)
        self.chiller_efficiency_mult: float = 1.0

    def reset(self, t_supply_init: float = 19.0, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng()
        noise = rng.uniform(-0.4, 0.4, size=self.n)
        self.t_in = np.full(self.n, t_supply_init + 1.0, dtype=np.float64) + noise
        self.t_out = self.t_in + rng.uniform(5.0, 7.0, size=self.n)
        self.chiller_efficiency_mult = 1.0

    def compute_it_power(
        self,
        rack_cpu_util: np.ndarray,
        consolidation_ratio: float,
    ) -> tuple[np.ndarray, int]:
        """
        Calculates per-rack IT power (kW) after applying VM consolidation.
        Consolidation packs workloads onto fewer active servers and puts idle ones to sleep.
        """
        util = np.clip(rack_cpu_util, 0.02, 1.0)
        total_servers = self.cfg.servers_per_rack

        min_needed = np.ceil((util * total_servers) / self.cfg.max_consolidated_util).astype(int)
        min_needed = np.clip(min_needed, 1, total_servers)

        target_active = np.round(
            total_servers - consolidation_ratio * (total_servers - min_needed)
        ).astype(int)
        target_active = np.clip(target_active, min_needed, total_servers)
        sleep_servers = total_servers - target_active

        eff_util = np.clip((util * total_servers) / target_active, 0.0, 1.0)

        active_w = target_active * (
            self.cfg.p_server_idle_w
            + (self.cfg.p_server_peak_w - self.cfg.p_server_idle_w)
            * (eff_util ** self.cfg.cpu_power_exponent)
        )
        sleep_w = sleep_servers * self.cfg.p_server_sleep_w
        rack_it_kw = (active_w + sleep_w) / 1000.0
        return rack_it_kw, int(target_active.sum())

    def compute_cop(self, t_supply_c: float, t_ambient_c: float) -> float:
        """Carnot-inspired chiller Coefficient of Performance model."""
        raw_cop = self.cfg.cop_nominal * (
            1.0
            + self.cfg.cop_supply_slope * (t_supply_c - 18.0)
            - self.cfg.cop_ambient_slope * (t_ambient_c - 25.0)
        )
        degraded_cop = raw_cop * self.chiller_efficiency_mult
        return float(np.clip(degraded_cop, 1.3, 7.0))

    def step(
        self,
        rack_cpu_util: np.ndarray,
        t_ambient_c: float,
        t_supply_c: float,
        airflow_frac: float,
        consolidation_ratio: float,
    ) -> dict[str, float | np.ndarray | int]:
        """Advances thermal and electrical state by one control interval (dt_seconds)."""
        t_supply_c = float(np.clip(t_supply_c, self.cfg.t_supply_min_c, self.cfg.t_supply_max_c))
        airflow_frac = float(np.clip(airflow_frac, self.cfg.airflow_min_frac, self.cfg.airflow_max_frac))
        consolidation_ratio = float(np.clip(consolidation_ratio, 0.0, 1.0))

        rack_it_kw, active_servers = self.compute_it_power(rack_cpu_util, consolidation_ratio)
        total_it_kw = float(rack_it_kw.sum())

        m_dot_rack = airflow_frac * self.cfg.max_airflow_per_rack_kg_s
        cp = self.cfg.air_specific_heat_kj_kgk
        sub_dt = self.cfg.dt_seconds / self.cfg.substeps

        for _ in range(self.cfg.substeps):
            recirc_temp = self.mixing_matrix @ self.t_out
            self.t_in = (1.0 - self.recirc_factor) * t_supply_c + self.recirc_factor * recirc_temp

            q_conv_kw = m_dot_rack * cp * (self.t_out - self.t_in)
            d_tout = ((rack_it_kw - q_conv_kw) / self.cfg.rack_thermal_cap_kj_k) * sub_dt
            self.t_out = np.clip(self.t_out + d_tout, self.t_in + 0.5, 65.0)

        recirc_temp = self.mixing_matrix @ self.t_out
        self.t_in = (1.0 - self.recirc_factor) * t_supply_c + self.recirc_factor * recirc_temp

        q_removed_kw = float(np.sum(np.maximum(0.0, m_dot_rack * cp * (self.t_out - t_supply_c))))
        cop = self.compute_cop(t_supply_c, t_ambient_c)

        p_chiller_kw = q_removed_kw / cop
        p_fan_kw = self.cfg.p_fan_max_kw * (airflow_frac ** 3) + self.cfg.p_pump_base_kw
        total_cool_kw = p_chiller_kw + p_fan_kw
        total_power_kw = total_it_kw + total_cool_kw
        pue = total_power_kw / max(total_it_kw, 1e-6)

        violations = int(np.sum(self.t_in > self.cfg.t_inlet_safe_max_c))

        return {
            "t_in": self.t_in.copy(),
            "t_out": self.t_out.copy(),
            "rack_it_kw": rack_it_kw,
            "it_power_kw": total_it_kw,
            "cooling_power_kw": total_cool_kw,
            "chiller_power_kw": p_chiller_kw,
            "fan_power_kw": p_fan_kw,
            "total_power_kw": total_power_kw,
            "pue": pue,
            "cop": cop,
            "active_servers": active_servers,
            "max_inlet_temp": float(self.t_in.max()),
            "thermal_violations": violations,
        }
