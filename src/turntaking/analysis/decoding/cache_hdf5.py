
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Tuple, Literal

import numpy as np

from turntaking.analysis.decoding.io import (
    Hdf5CacheParams,
    load_subject_feature_cache_hdf5,
    save_subject_feature_cache_hdf5,
)

Contrast = Literal["latency", "duration"]


@dataclass(frozen=True)
class DecodingCacheConfig:
    compression: str | None
    compression_level: int
    dtype: str

    @staticmethod
    def from_config_dict(cfg: dict[str, Any]) -> "DecodingCacheConfig":
        # Only used if --cache-features is passed.
        compression = _get_optional(cfg, "analysis.decoding.cache.compression", "gzip")
        level = int(_get_optional(cfg, "analysis.decoding.cache.compression_level", 4))
        dtype = str(_get_optional(cfg, "analysis.decoding.cache.dtype", "float32"))
        return DecodingCacheConfig(compression=compression, compression_level=level, dtype=dtype)


def _get_optional(cfg: dict[str, Any], dotted_key: str, default: Any) -> Any:
    cur: Any = cfg
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    if not isinstance(cur, dict):
        return default
    return cur.get(parts[-1], default)


def make_hdf5_cache_io(
    *,
    out_dir: Path,
    contrast: Contrast,
    cache_cfg: DecodingCacheConfig,
) -> Tuple[
    Callable[[str], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    Callable[[str, np.ndarray, np.ndarray, np.ndarray], None],
]:
    params = Hdf5CacheParams(
        compression=cache_cfg.compression,
        compression_level=cache_cfg.compression_level,
        x_dtype=cache_cfg.dtype,
    )

    def _load(subject: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return load_subject_feature_cache_hdf5(
            out_dir=out_dir,
            contrast=contrast,
            subject=subject,
        )

    def _save(subject: str, X: np.ndarray, y: np.ndarray, times_s: np.ndarray) -> None:
        save_subject_feature_cache_hdf5(
            out_dir=out_dir,
            contrast=contrast,
            subject=subject,
            X=X,
            y=y,
            times_s=times_s,
            cache_params=params,
        )

    return _load, _save
