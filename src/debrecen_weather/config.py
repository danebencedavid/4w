"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(config_path)
    cfg["_repo_root"] = str(config_path.resolve().parent.parent)
    return cfg


def repo_root(cfg: dict[str, Any]) -> Path:
    """Return the project root inferred from the config location."""
    return Path(cfg.get("_repo_root", ".")).resolve()


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a path from the config's paths section relative to the repo root."""
    value = cfg["paths"][key]
    path = Path(value)
    if not path.is_absolute():
        path = repo_root(cfg) / path
    return path


def resolve_optional_path(cfg: dict[str, Any], key: str) -> Path | None:
    """Resolve a configured path if it exists."""
    if key not in cfg.get("paths", {}):
        return None
    return resolve_path(cfg, key)


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    """Create a directory if needed."""
    path.mkdir(parents=True, exist_ok=True)
