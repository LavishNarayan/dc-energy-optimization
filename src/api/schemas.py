from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    policy_mode: str
    forecaster_loaded: bool
    num_racks: int
    ashrae_safe_limit_c: float


class StepRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=288, description="Number of 5-minute control steps to advance")
    use_shield: bool = Field(default=True, description="Enable ASHRAE 27°C Physics Safety Shield")


class FaultRequest(BaseModel):
    chiller_efficiency_mult: float = Field(default=0.65, ge=0.2, le=1.0)
    load_spike: float = Field(default=0.25, ge=0.0, le=0.60)


class TelemetryResponse(BaseModel):
    step: int
    policy_mode: str
    shield_overridden: bool
    inference_latency_ms: float
    pue: float
    pid_baseline_pue: float
    it_power_kw: float
    cooling_power_kw: float
    total_power_kw: float
    pid_total_power_kw: float
    t_supply_c: float
    airflow_frac: float
    consolidation_ratio: float
    max_inlet_temp_c: float
    thermal_violations: int
    cumulative_energy_saved_kwh: float
    cumulative_co2_saved_kg: float
    rack_inlet_temps_c: list[float]
    rack_outlet_temps_c: list[float]
