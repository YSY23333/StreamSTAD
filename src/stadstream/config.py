from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_dir"] = str(path.parent.resolve())
    return cfg


def resolve_path(path: str | Path, base_dir: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()

