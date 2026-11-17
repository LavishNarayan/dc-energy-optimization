from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.models.forecaster import ForecasterPredictor
from src.simulator.dc_env import DataCenterEnv
from src.simulator.thermodynamics import DataCenterConfig
from src.utils.config_loader import load_yaml_config
from src.utils.logger import get_logger

logger = get_logger("dc_optimizer.train_rl")


def make_env(
    forecaster_path: str | Path = ROOT_DIR / "artifacts" / "lstm_forecaster.pt",
    episode_days: int = 3,
    seed: int = 42,
):
    def _init():
        forecaster_fn = None
        if Path(forecaster_path).exists():
            predictor = ForecasterPredictor.load(forecaster_path, device="cpu")
            forecaster_fn = predictor.make_env_forecaster_fn()

        env = DataCenterEnv(
            config=DataCenterConfig.from_yaml(),
            episode_days=episode_days,
            forecast_horizon=12,
            forecaster_fn=forecaster_fn,
            seed=seed,
        )
        return Monitor(env)

    return _init


def train_ppo_agent(
    forecaster_path: str | Path = ROOT_DIR / "artifacts" / "lstm_forecaster.pt",
    model_save_path: str | Path = ROOT_DIR / "artifacts" / "ppo_dc_optimizer",
    total_timesteps: int | None = None,
    seed: int = 42,
) -> PPO:
    cfg = load_yaml_config("configs/train_rl.yaml")
    ppo_cfg = cfg.get("ppo", {})

    steps = int(total_timesteps if total_timesteps is not None else cfg.get("total_timesteps", 100_000))
    ep_days = int(cfg.get("episode_days", 3))
    eval_freq = int(cfg.get("eval_freq", 10_000))

    log_dir = ROOT_DIR / "artifacts" / "logs"
    best_dir = ROOT_DIR / "artifacts" / "best_ppo"
    log_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([make_env(forecaster_path, episode_days=ep_days, seed=seed)])
    eval_env = DummyVecEnv([make_env(forecaster_path, episode_days=ep_days, seed=seed + 500)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 2048)),
        batch_size=int(ppo_cfg.get("batch_size", 128)),
        n_epochs=int(ppo_cfg.get("n_epochs", 10)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.005)),
        policy_kwargs=dict(
            net_arch=dict(
                pi=ppo_cfg.get("net_arch_pi", [256, 256]),
                vf=ppo_cfg.get("net_arch_vf", [256, 256]),
            )
        ),
        verbose=1,
        seed=seed,
    )

    logger.info(f"Starting PPO training for {steps:,} timesteps...")
    model.learn(total_timesteps=steps, callback=eval_callback, progress_bar=False)
    model.save(str(model_save_path))
    logger.info(f"Saved final PPO policy -> {model_save_path}.zip")

    best_zip = best_dir / "best_model.zip"
    if best_zip.exists():
        best_model = PPO.load(best_zip, env=eval_env)
        best_model.save(str(model_save_path))
        logger.info(f"Promoted highest-reward evaluation checkpoint -> {model_save_path}.zip")
        final_model = best_model
    else:
        final_model = model

    meta_path = ROOT_DIR / "artifacts" / "model_metadata.json"
    existing = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    existing["ppo_agent"] = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_timesteps": steps,
        "episode_days": ep_days,
        "model_path": "artifacts/ppo_dc_optimizer.zip",
    }
    meta_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    return final_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Reinforcement Learning Cooling Agent")
    parser.add_argument("--timesteps", type=int, default=None, help="Total RL environment steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_ppo_agent(total_timesteps=args.timesteps, seed=args.seed)
