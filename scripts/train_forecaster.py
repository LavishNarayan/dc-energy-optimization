from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.generate_dataset import generate_30day_dataset
from src.models.forecaster import ForecasterPredictor, MultiHorizonLSTM
from src.utils.config_loader import load_yaml_config
from src.utils.logger import get_logger

logger = get_logger("dc_optimizer.train_lstm")


def build_sliding_windows(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    seq_len: int = 24,
    horizon: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    num_samples = len(x_raw) - seq_len - horizon + 1
    x_list = np.stack([x_raw[i : i + seq_len] for i in range(num_samples)], axis=0)
    y_list = np.stack(
        [y_raw[i + seq_len : i + seq_len + horizon] for i in range(num_samples)],
        axis=0,
    )
    return x_list, y_list


def update_metadata(updates: dict) -> None:
    meta_path = ROOT_DIR / "artifacts" / "model_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
    existing.update(updates)
    meta_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def train_forecaster(
    csv_path: str | Path = ROOT_DIR / "data" / "raw" / "dc_telemetry_30d.csv",
    save_path: str | Path = ROOT_DIR / "artifacts" / "lstm_forecaster.pt",
    epochs: int | None = None,
    seed: int = 42,
) -> ForecasterPredictor:
    cfg = load_yaml_config("configs/train_lstm.yaml")
    arch_cfg = cfg.get("architecture", {})
    tr_cfg = cfg.get("training", {})

    seq_len = int(arch_cfg.get("seq_len", 24))
    horizon = int(arch_cfg.get("horizon", 12))
    hidden_dim = int(arch_cfg.get("hidden_dim", 128))
    num_layers = int(arch_cfg.get("num_layers", 2))
    dropout = float(arch_cfg.get("dropout", 0.15))
    num_racks = 10

    n_epochs = int(epochs if epochs is not None else tr_cfg.get("epochs", 35))
    batch_size = int(tr_cfg.get("batch_size", 64))
    lr = float(tr_cfg.get("learning_rate", 1e-3))

    torch.manual_seed(seed)
    np.random.seed(seed)

    data_file = Path(csv_path)
    if not data_file.exists():
        logger.info(f"Dataset not found at {data_file}. Generating 30-day synthetic dataset...")
        df = generate_30day_dataset(output_path=csv_path, seed=seed)
    else:
        df = pd.read_csv(data_file)

    x_raw, y_raw = ForecasterPredictor.extract_features_from_df(df, num_racks=num_racks)

    split_idx = int(0.8 * len(x_raw))
    x_train_raw, y_train_raw = x_raw[:split_idx], y_raw[:split_idx]
    x_val_raw, y_val_raw = x_raw[split_idx:], y_raw[split_idx:]

    x_mean, x_std = x_train_raw.mean(axis=0), x_train_raw.std(axis=0)
    y_mean, y_std = y_train_raw.mean(axis=0), y_train_raw.std(axis=0)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)

    x_train_norm = (x_train_raw - x_mean) / x_std
    y_train_norm = (y_train_raw - y_mean) / y_std
    x_val_norm = (x_val_raw - x_mean) / x_std
    y_val_norm = (y_val_raw - y_mean) / y_std

    x_tr_win, y_tr_win = build_sliding_windows(x_train_norm, y_train_norm, seq_len, horizon)
    x_va_win, y_va_win = build_sliding_windows(x_val_norm, y_val_norm, seq_len, horizon)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr_win), torch.from_numpy(y_tr_win)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_va_win), torch.from_numpy(y_va_win)),
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHorizonLSTM(
        input_dim=x_raw.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        horizon=horizon,
        num_racks=num_racks,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    best_state = None

    logger.info(
        f"Training MultiHorizonLSTM on {device} | Train windows: {len(x_tr_win)} | Val windows: {len(x_va_win)}"
    )
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(x_tr_win)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_loss += criterion(pred, by).item() * len(bx)
        val_loss /= len(x_va_win)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:02d}/{n_epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

    assert best_state is not None
    model.load_state_dict(best_state)
    predictor = ForecasterPredictor(
        model=model,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        seq_len=seq_len,
        device=str(device),
    )
    predictor.save(save_path)

    model.eval()
    with torch.no_grad():
        preds_norm = model(torch.from_numpy(x_va_win).to(device)).cpu().numpy()
    preds_phys = preds_norm * y_std + y_mean
    truth_phys = y_va_win * y_std + y_mean

    cpu_mae_15m = float(np.mean(np.abs(preds_phys[:, 2, :num_racks] - truth_phys[:, 2, :num_racks])) * 100.0)
    cpu_mae_60m = float(np.mean(np.abs(preds_phys[:, 11, :num_racks] - truth_phys[:, 11, :num_racks])) * 100.0)
    amb_mae_60m = float(np.mean(np.abs(preds_phys[:, 11, num_racks] - truth_phys[:, 11, num_racks])))

    update_metadata(
        {
            "lstm_forecaster": {
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "epochs": n_epochs,
                "best_val_loss": round(best_val_loss, 6),
                "cpu_mae_15min_pct": round(cpu_mae_15m, 3),
                "cpu_mae_60min_pct": round(cpu_mae_60m, 3),
                "ambient_mae_60min_c": round(amb_mae_60m, 3),
            }
        }
    )

    logger.info(f"Saved best LSTM model -> {save_path}")
    logger.info(f"Validation CPU Load MAE (+15 min): {cpu_mae_15m:.2f}% | (+60 min): {cpu_mae_60m:.2f}%")
    logger.info(f"Validation Ambient Temp MAE (+60 min): {amb_mae_60m:.2f} °C")
    return predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Horizon LSTM Forecaster")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_forecaster(epochs=args.epochs, seed=args.seed)
