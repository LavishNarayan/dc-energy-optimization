from __future__ import annotations

from pathlib import Path
import sys
from fastapi import FastAPI

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api.schemas import FaultRequest, HealthResponse, StepRequest, TelemetryResponse
from src.controllers.pid_baseline import PIDBaselineController
from src.controllers.rl_agent import RLCoolingAgent
from src.models.forecaster import ForecasterPredictor
from src.simulator.dc_env import DataCenterEnv
from src.simulator.thermodynamics import DataCenterConfig
from src.utils.logger import get_logger

logger = get_logger("dc_optimizer.api")

FORECASTER_PATH = ROOT_DIR / "artifacts" / "lstm_forecaster.pt"
PPO_PATH = ROOT_DIR / "artifacts" / "ppo_dc_optimizer.zip"

app = FastAPI(
    title="AI Data Center Energy Optimization API",
    description="REST inference, telemetry streaming, and digital twin control service",
    version="1.0.0",
)


class PlantService:
    def __init__(self) -> None:
        self.cfg = DataCenterConfig.from_yaml()
        self.forecaster_loaded = False
        forecaster_fn = None

        if FORECASTER_PATH.exists():
            predictor = ForecasterPredictor.load(FORECASTER_PATH, device="cpu")
            forecaster_fn = predictor.make_env_forecaster_fn()
            self.forecaster_loaded = True

        self.agent = RLCoolingAgent(model_path=PPO_PATH, use_shield=True)
        self.ai_env = DataCenterEnv(config=self.cfg, episode_days=7, forecaster_fn=forecaster_fn, seed=101)
        self.pid_env = DataCenterEnv(config=self.cfg, episode_days=7, forecaster_fn=None, seed=101)
        self.pid_ctrl = PIDBaselineController(num_racks=self.cfg.num_racks)

        self.ai_obs, _ = self.ai_env.reset(seed=101)
        self.pid_obs, _ = self.pid_env.reset(seed=101)
        self.pid_ctrl.reset()

        self.step_count = 0
        self.cum_saved_kwh = 0.0
        self.cum_saved_co2 = 0.0
        self.latest_telemetry: TelemetryResponse | None = None

    def reset(self, seed: int = 101) -> None:
        self.ai_obs, _ = self.ai_env.reset(seed=seed)
        self.pid_obs, _ = self.pid_env.reset(seed=seed)
        self.pid_ctrl.reset()
        self.agent.shield.reset_stats()
        self.step_count = 0
        self.cum_saved_kwh = 0.0
        self.cum_saved_co2 = 0.0
        self.advance(steps=1, use_shield=True)

    def advance(self, steps: int = 1, use_shield: bool = True) -> TelemetryResponse:
        self.agent.use_shield = use_shield
        for _ in range(steps):
            if self.step_count >= self.ai_env.episode_steps:
                self.reset()
                break

            ai_action, meta = self.agent.predict_action(self.ai_obs, self.ai_env)
            pid_action = self.pid_ctrl.act(self.pid_obs, self.pid_env)

            self.ai_obs, _, _, _, ai_info = self.ai_env.step(ai_action)
            self.pid_obs, _, _, _, pid_info = self.pid_env.step(pid_action)

            self.cum_saved_kwh += float(pid_info["step_kwh"] - ai_info["step_kwh"])
            self.cum_saved_co2 += float(pid_info["step_carbon_kg"] - ai_info["step_carbon_kg"])
            self.step_count += 1

            self.latest_telemetry = TelemetryResponse(
                step=self.step_count,
                policy_mode=str(meta["policy_mode"]),
                shield_overridden=bool(meta["shield_overridden"]),
                inference_latency_ms=float(meta["inference_latency_ms"]),
                pue=round(float(ai_info["pue"]), 4),
                pid_baseline_pue=round(float(pid_info["pue"]), 4),
                it_power_kw=round(float(ai_info["it_power_kw"]), 3),
                cooling_power_kw=round(float(ai_info["cooling_power_kw"]), 3),
                total_power_kw=round(float(ai_info["total_power_kw"]), 3),
                pid_total_power_kw=round(float(pid_info["total_power_kw"]), 3),
                t_supply_c=round(float(ai_info["t_supply_c"]), 3),
                airflow_frac=round(float(ai_info["airflow_frac"]), 4),
                consolidation_ratio=round(float(ai_info["consolidation_ratio"]), 4),
                max_inlet_temp_c=round(float(ai_info["max_inlet_temp"]), 3),
                thermal_violations=int(ai_info["thermal_violations"]),
                cumulative_energy_saved_kwh=round(self.cum_saved_kwh, 3),
                cumulative_co2_saved_kg=round(self.cum_saved_co2, 3),
                rack_inlet_temps_c=[round(float(x), 2) for x in ai_info["t_in"]],
                rack_outlet_temps_c=[round(float(x), 2) for x in ai_info["t_out"]],
            )
        assert self.latest_telemetry is not None
        return self.latest_telemetry


service = PlantService()
service.advance(steps=1, use_shield=True)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        policy_mode=service.agent.policy_mode,
        forecaster_loaded=service.forecaster_loaded,
        num_racks=service.cfg.num_racks,
        ashrae_safe_limit_c=service.cfg.t_inlet_safe_max_c,
    )


@app.get("/metrics", response_model=TelemetryResponse)
def get_latest_metrics() -> TelemetryResponse:
    assert service.latest_telemetry is not None
    return service.latest_telemetry


@app.post("/step", response_model=TelemetryResponse)
def step_plant(req: StepRequest) -> TelemetryResponse:
    return service.advance(steps=req.steps, use_shield=req.use_shield)


@app.post("/inject-fault")
def inject_fault(req: FaultRequest) -> dict[str, str | float]:
    service.ai_env.inject_fault(
        chiller_efficiency_mult=req.chiller_efficiency_mult,
        load_spike=req.load_spike,
    )
    service.pid_env.inject_fault(
        chiller_efficiency_mult=req.chiller_efficiency_mult,
        load_spike=req.load_spike,
    )
    logger.warning(
        f"Injected fault: chiller_efficiency={req.chiller_efficiency_mult}, load_spike={req.load_spike}"
    )
    return {
        "status": "fault_injected",
        "chiller_efficiency_mult": req.chiller_efficiency_mult,
        "load_spike": req.load_spike,
    }


@app.post("/reset", response_model=TelemetryResponse)
def reset_plant(seed: int = 101) -> TelemetryResponse:
    service.reset(seed=seed)
    assert service.latest_telemetry is not None
    return service.latest_telemetry
