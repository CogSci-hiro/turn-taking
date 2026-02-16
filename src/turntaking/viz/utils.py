"""Shared helpers for visualization entrypoints."""


from collections.abc import Mapping
from pathlib import Path
from typing import Any


def cfg_get(cfg: Any, *keys: str) -> Any:
    current: Any = cfg
    for key in keys:
        if isinstance(current, Mapping):
            current = current[key]
        else:
            current = getattr(current, key)
    return current


def cfg_get_optional(cfg: Any, *keys: str, default: Any = None) -> Any:
    current: Any = cfg
    for key in keys:
        if isinstance(current, Mapping):
            if key not in current:
                return default
            current = current[key]
            continue
        if not hasattr(current, key):
            return default
        current = getattr(current, key)
    return current


def out_dir(cfg: Any) -> Path:
    value = cfg_get_optional(cfg, "io", "out_dir")
    if value is None:
        value = cfg_get(cfg, "paths", "out_dir")
    return Path(value)


def resolve_from_out_dir(cfg: Any, relative_or_absolute: str | Path) -> Path:
    root = out_dir(cfg).resolve()
    path = Path(relative_or_absolute)
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Viz artifact must live under out_dir. path={resolved}, out_dir={root}")
    return resolved


def viz_mode(cfg: Any, family: str, default: str) -> str:
    value = cfg_get_optional(cfg, "viz", family, "mode", default=default)
    return str(value)

