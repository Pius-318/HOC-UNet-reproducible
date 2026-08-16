"""Configuration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import yaml


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        if path.suffix.lower() == ".json":
            return json.load(f)
    raise ValueError(f"Unsupported config format: {path}")


def merge_overrides(config: dict[str, Any], overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    merged = dict(config)
    for key, value in (overrides or {}).items():
        parts = key.split(".")
        cursor = merged
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return merged
