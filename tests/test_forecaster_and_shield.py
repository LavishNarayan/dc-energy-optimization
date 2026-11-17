import numpy as np
import torch

from src.models.forecaster import ForecasterPredictor, MultiHorizonLSTM
from src.models.safety_shield import PhysicsSafetyShield
from src.simulator.dc_env import DataCenterEnv


def test_lstm_forward_and_serialization(tmp_path):
    model = MultiHorizonLSTM(input_dim=14, hidden_dim=32, num_layers=1, horizon=12, num_racks=10)
    x = torch.randn(4, 24, 14)
    out = model(x)
    assert out.shape == (4, 12, 11)

    predictor = ForecasterPredictor(
        model=model,
        x_mean=np.zeros(14, dtype=np.float32),
        x_std=np.ones(14, dtype=np.float32),
        y_mean=np.zeros(11, dtype=np.float32),
        y_std=np.ones(11, dtype=np.float32),
        seq_len=24,
        device="cpu",
    )
    ckpt = tmp_path / "test_lstm.pt"
    predictor.save(ckpt)
    loaded = ForecasterPredictor.load(ckpt, device="cpu")

    obs_window = np.random.rand(24, 167).astype(np.float32)
    fut_cpu, fut_amb = loaded.predict_from_obs_window(obs_window)
    assert fut_cpu.shape == (12, 10)
    assert fut_amb.shape == (12,)


def test_safety_shield_overrides_unsafe_heating():
    env = DataCenterEnv(episode_days=1, seed=12)
    env.reset(seed=12)
    # Force high compute burst + hot room near threshold
    env.trajectories["cpu_util"][env.step_idx] = 0.95
    env.physics.t_in[:] = 26.2
    env.physics.t_out[:] = 42.0

    shield = PhysicsSafetyShield(trigger_temp_c=26.4)
    # Reckless action: maximum supply temp (24°C) and minimum airflow (30%)
    reckless_action = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    safe_action, overridden = shield.filter_action(reckless_action, env)

    assert overridden is True
    assert safe_action[0] < reckless_action[0]  # Supply temp lowered
    assert safe_action[1] > reckless_action[1]  # Airflow increased
