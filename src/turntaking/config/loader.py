"""
Configuration loader.

This module is the single place that knows how to:

- read YAML from disk,
- migrate legacy visualization keys,
- validate and materialize a typed ``TurntakingConfig``.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from turntaking.config.analysis_schema import TurntakingConfig

LEGACY_VIZ_KEY_MAP: dict[tuple[str, ...], tuple[str, ...]] = {
    ("viz", "topomaps", "erp", "static"): ("viz", "erp_topo"),
    ("viz", "topomaps", "erp", "svg"): ("viz", "erp_topomaps"),
    ("viz", "topomaps", "tfr", "static"): ("viz", "tfr_topos"),
    ("viz", "topomaps", "tfr", "svg"): ("viz", "tfr_topomaps"),
}


def load_config(path: Path) -> TurntakingConfig:
    """
    Load a YAML config file and return a validated ``TurntakingConfig``.

    Parameters
    ----------
    path
        Path to a YAML file (typically ``workflow/config.yaml``).
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")
    migrated = _migrate_legacy_keys(raw)
    return TurntakingConfig.from_dict(migrated)


def _migrate_legacy_keys(raw: dict[str, Any]) -> dict[str, Any]:
    config_dict = deepcopy(raw)
    for new_path, legacy_path in LEGACY_VIZ_KEY_MAP.items():
        value = _deep_get(config_dict, new_path)
        if value is None:
            continue
        _deep_setdefault(config_dict, legacy_path, value)
    return config_dict


def _deep_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _deep_setdefault(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current: dict[str, Any] = payload
    for key in path[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    current.setdefault(path[-1], value)
