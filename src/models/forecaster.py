from __future__ import annotations

from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class MultiHorizonLSTM(nn.Module):
    """
    Sequence-to-horizon LSTM predicting future per-rack CPU utilization
    and outdoor ambient temperature over H=12 steps (60 minutes).
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 128,
        num_layers: int = 2,
        horizon: int = 12,
        num_racks: int = 10,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.num_racks = num_racks
        self.target_dim = num_racks + 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon * self.target_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        pred = self.head(last_hidden)
        return pred.view(-1, self.horizon, self.target_dim)


class ForecasterPredictor:
    """
    Handles feature alignment, standardization, checkpoint serialization,
    and Gymnasium environment callback integration for MultiHorizonLSTM.
    """

    def __init__(
        self,
        model: MultiHorizonLSTM,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        seq_len: int = 24,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.model.eval()

        self.x_mean = x_mean.astype(np.float32)
        self.x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
        self.y_mean = y_mean.astype(np.float32)
        self.y_std = np.where(y_std < 1e-6, 1.0, y_std).astype(np.float32)
        self.seq_len = seq_len
        self.num_racks = model.num_racks
        self.horizon = model.horizon

    @staticmethod
    def extract_features_from_df(df: pd.DataFrame, num_racks: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Extracts aligned input features X_raw (T, 14) and targets Y_raw (T, 11) from CSV."""
        rack_cols = [f"rack_{r}_cpu" for r in range(num_racks)]
        cpu_mat = df[rack_cols].to_numpy(dtype=np.float32)
        t_amb = df["t_ambient_c"].to_numpy(dtype=np.float32)[:, None]
        t_in_mean = df["mean_inlet_temp_c"].to_numpy(dtype=np.float32)[:, None]
        it_kw = df["it_power_kw"].to_numpy(dtype=np.float32)[:, None]
        carbon = df["carbon_intensity_gco2_kwh"].to_numpy(dtype=np.float32)[:, None]

        x_raw = np.concatenate([cpu_mat, t_amb, t_in_mean, it_kw, carbon], axis=1)
        y_raw = np.concatenate([cpu_mat, t_amb], axis=1)
        return x_raw, y_raw

    def extract_features_from_obs_window(self, obs_window: np.ndarray) -> np.ndarray:
        """Extracts the exact same 14 features from DataCenterEnv's rolling observation buffer."""
        n = self.num_racks
        t_in_mean = obs_window[:, 0:n].mean(axis=1, keepdims=True)
        cpu_mat = obs_window[:, 2 * n : 3 * n]
        it_kw = obs_window[:, 3 * n : 3 * n + 1]
        t_amb = obs_window[:, 3 * n + 2 : 3 * n + 3]
        carbon = obs_window[:, 3 * n + 3 : 3 * n + 4]

        return np.concatenate([cpu_mat, t_amb, t_in_mean, it_kw, carbon], axis=1).astype(np.float32)

    @torch.no_grad()
    def predict_from_obs_window(self, obs_window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicts (future_cpu [H, N], future_amb [H]) from a (24, obs_dim) window."""
        x_raw = self.extract_features_from_obs_window(obs_window[-self.seq_len :])
        x_norm = (x_raw - self.x_mean) / self.x_std
        x_tensor = torch.from_numpy(x_norm[None, :, :]).to(self.device)

        pred_norm = self.model(x_tensor).cpu().numpy()[0]
        pred = pred_norm * self.y_std + self.y_mean

        future_cpu = np.clip(pred[:, : self.num_racks], 0.05, 0.99).astype(np.float32)
        future_amb = pred[:, self.num_racks].astype(np.float32)
        return future_cpu, future_amb

    def make_env_forecaster_fn(self) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
        return self.predict_from_obs_window

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": self.x_mean.shape[0],
                "hidden_dim": self.model.lstm.hidden_size,
                "num_layers": self.model.lstm.num_layers,
                "horizon": self.horizon,
                "num_racks": self.num_racks,
                "seq_len": self.seq_len,
                "x_mean": self.x_mean,
                "x_std": self.x_std,
                "y_mean": self.y_mean,
                "y_std": self.y_std,
            },
            out,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> ForecasterPredictor:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = MultiHorizonLSTM(
            input_dim=int(ckpt["input_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            num_layers=int(ckpt["num_layers"]),
            horizon=int(ckpt["horizon"]),
            num_racks=int(ckpt["num_racks"]),
        )
        model.load_state_dict(ckpt["state_dict"])
        return cls(
            model=model,
            x_mean=ckpt["x_mean"],
            x_std=ckpt["x_std"],
            y_mean=ckpt["y_mean"],
            y_std=ckpt["y_std"],
            seq_len=int(ckpt["seq_len"]),
            device=device,
        )
