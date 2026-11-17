from fastapi.testclient import TestClient
from gymnasium.utils.env_checker import check_env

from src.api.main import app
from src.controllers.pid_baseline import PIDBaselineController
from src.controllers.rl_agent import RLCoolingAgent
from src.simulator.dc_env import DataCenterEnv


def test_gymnasium_compliance_and_controllers():
    env = DataCenterEnv(episode_days=1, seed=42)
    check_env(env)

    obs, _ = env.reset(seed=42)
    pid = PIDBaselineController(num_racks=10)
    pid.reset()
    agent = RLCoolingAgent(model_path="nonexistent.zip", use_shield=True)

    for _ in range(10):
        a_pid = pid.act(obs, env)
        a_ai, meta = agent.predict_action(obs, env)
        assert a_pid.shape == (3,)
        assert a_ai.shape == (3,)
        assert "t_supply_c" in meta
        obs, reward, terminated, truncated, info = env.step(a_ai)
        assert not terminated
        assert info["pue"] >= 1.0


def test_fastapi_endpoints():
    client = TestClient(app)

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    res_step = client.post("/step", json={"steps": 3, "use_shield": True})
    assert res_step.status_code == 200
    body = res_step.json()
    assert body["pue"] >= 1.0
    assert len(body["rack_inlet_temps_c"]) == 10

    res_fault = client.post("/inject-fault", json={"chiller_efficiency_mult": 0.7, "load_spike": 0.2})
    assert res_fault.status_code == 200
