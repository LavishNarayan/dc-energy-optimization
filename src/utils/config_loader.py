from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_yaml_config(rel_path: str) -> dict[str, Any]:
    """Loads a YAML configuration file resolved from the repository root."""
    cfg_path = ROOT_DIR / rel_path
    if not cfg_path.exists() or yaml is None:
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}
