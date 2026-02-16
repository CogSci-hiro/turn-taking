
"""Decoding-domain I/O helpers for scores, feature caches, and cluster outputs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Tuple

import h5py
import numpy as np
import pandas as pd

from turntaking.analysis.utils.io import ensure_dir_exists, save_array_nd, save_dataframe_csv

ContrastName = Literal["latency", "duration"]

_SCORES_FNAME: Final[str] = "scores.npy"
_TIMES_FNAME: Final[str] = "times.npy"

__all__ = [
    "ContrastName",
    "Hdf5CacheParams",
    "DecodingScorePaths",
    "DecodingClusterResults",
    "get_decoding_out_dir",
    "get_decoding_cluster_out_dir",
    "save_decoding_scores",
    "load_decoding_scores",
    "get_feature_cache_path",
    "save_subject_feature_cache_hdf5",
    "load_subject_feature_cache_hdf5",
    "save_decoding_cluster_results_hdf5",
    "write_decoding_cluster_outputs",
    "load_decoding_cluster_results_hdf5",
]


@dataclass(frozen=True)
class Hdf5CacheParams:
    compression: str | None = "gzip"
    compression_level: int = 4
    x_dtype: str = "float32"


@dataclass(frozen=True)
class DecodingScorePaths:
    scores_npy: Path
    times_npy: Path

    @staticmethod
    def from_dir(decoding_dir: Path) -> "DecodingScorePaths":
        return DecodingScorePaths(
            scores_npy=decoding_dir / _SCORES_FNAME,
            times_npy=decoding_dir / _TIMES_FNAME,
        )


@dataclass(frozen=True)
class DecodingClusterResults:
    t_values: np.ndarray
    clusters: list[tuple[np.ndarray, np.ndarray]]
    p_values: np.ndarray
    h0: np.ndarray


def get_decoding_out_dir(out_dir: Path, contrast: ContrastName) -> Path:
    return Path(out_dir) / "decoding" / "erp" / contrast


def get_decoding_cluster_out_dir(out_dir: Path, contrast: ContrastName) -> Path:
    return Path(out_dir) / "stats" / "decoding" / "erp" / contrast


def save_decoding_scores(
    *,
    out_dir: Path,
    contrast: ContrastName,
    scores: np.ndarray,
    times_s: np.ndarray,
) -> Tuple[Path, Path]:
    out_path = ensure_dir_exists(get_decoding_out_dir(out_dir, contrast))
    scores_path = save_array_nd(scores, out_path / _SCORES_FNAME)
    times_path = save_array_nd(np.asarray(times_s, dtype=np.float64), out_path / _TIMES_FNAME)
    return scores_path, times_path


def load_decoding_scores(paths: DecodingScorePaths) -> tuple[np.ndarray, np.ndarray]:
    if not paths.scores_npy.exists():
        raise FileNotFoundError(f"Missing scores file: {paths.scores_npy}")
    if not paths.times_npy.exists():
        raise FileNotFoundError(f"Missing times file: {paths.times_npy}")
    scores = np.load(paths.scores_npy)
    times_s = np.load(paths.times_npy)
    _validate_scores_shapes(scores, times_s)
    return scores, times_s


def _validate_scores_shapes(scores: np.ndarray, times_s: np.ndarray) -> None:
    if scores.ndim != 4:
        raise ValueError(f"Expected scores as 4D array, got shape={scores.shape}.")
    if times_s.ndim != 1:
        raise ValueError(f"Expected times as 1D array, got shape={times_s.shape}.")
    if scores.shape[2] != times_s.shape[0] or scores.shape[3] != times_s.shape[0]:
        raise ValueError(
            f"Time axis mismatch: scores has n_times={scores.shape[2]}x{scores.shape[3]}, "
            f"times has n_times={times_s.shape[0]}."
        )


def get_feature_cache_path(out_dir: Path, contrast: ContrastName, subject: str) -> Path:
    return get_decoding_out_dir(out_dir, contrast) / "features" / f"{subject}.h5"


def save_subject_feature_cache_hdf5(
    *,
    out_dir: Path,
    contrast: ContrastName,
    subject: str,
    X: np.ndarray,
    y: np.ndarray,
    times_s: np.ndarray,
    cache_params: Hdf5CacheParams,
) -> Path:
    cache_path = get_feature_cache_path(out_dir, contrast, subject)
    ensure_dir_exists(cache_path.parent)
    with h5py.File(cache_path, "w") as h5:
        _write_feature_cache(h5, X, y, times_s, cache_params)
    return cache_path


def _write_feature_cache(
    handle: h5py.File,
    X: np.ndarray,
    y: np.ndarray,
    times_s: np.ndarray,
    cache_params: Hdf5CacheParams,
) -> None:
    X_to_write = np.asarray(X, dtype=np.dtype(cache_params.x_dtype))
    y_to_write = np.asarray(y, dtype=np.int8)
    times_to_write = np.asarray(times_s, dtype=np.float64)
    compression = cache_params.compression
    compression_opts = cache_params.compression_level if compression == "gzip" else None
    trial_chunk = min(32, max(1, X_to_write.shape[0]))
    chunks = (trial_chunk, X_to_write.shape[1], X_to_write.shape[2])
    handle.create_dataset(
        "X",
        data=X_to_write,
        compression=compression,
        compression_opts=compression_opts,
        chunks=chunks,
    )
    handle.create_dataset("y", data=y_to_write)
    handle.create_dataset("times", data=times_to_write)


def load_subject_feature_cache_hdf5(
    *,
    out_dir: Path,
    contrast: ContrastName,
    subject: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = get_feature_cache_path(out_dir, contrast, subject)
    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")
    with h5py.File(cache_path, "r") as h5:
        X = np.asarray(h5["X"])
        y = np.asarray(h5["y"])
        times_s = np.asarray(h5["times"])
    return X, y, times_s


def save_decoding_cluster_results_hdf5(
    *,
    out_hdf5: Path,
    t_values: np.ndarray,
    clusters: list[tuple[np.ndarray, np.ndarray]],
    p_values: np.ndarray,
    h0: np.ndarray,
) -> Path:
    ensure_dir_exists(out_hdf5.parent)
    with h5py.File(out_hdf5, "w") as handle:
        handle.create_dataset("t-values", data=t_values, dtype=float)
        handle.create_dataset("p-values", data=p_values, dtype=float)
        handle.create_dataset("h0", data=h0, dtype=float)
        for idx, (train_idx, test_idx) in enumerate(clusters):
            handle.create_dataset(f"clusters/train-{idx}", data=np.asarray(train_idx, dtype=int), dtype=int)
            handle.create_dataset(f"clusters/test-{idx}", data=np.asarray(test_idx, dtype=int), dtype=int)
    return out_hdf5


def write_decoding_cluster_outputs(
    *,
    out_dir: Path,
    contrast: ContrastName,
    t_values: np.ndarray,
    clusters: list[tuple[np.ndarray, np.ndarray]],
    p_values: np.ndarray,
    h0: np.ndarray,
    summary: pd.DataFrame,
) -> tuple[Path, Path]:
    stats_dir = ensure_dir_exists(get_decoding_cluster_out_dir(out_dir, contrast))
    hdf5_path = save_decoding_cluster_results_hdf5(
        out_hdf5=stats_dir / "cluster_results.hdf5",
        t_values=t_values,
        clusters=clusters,
        p_values=p_values,
        h0=h0,
    )
    csv_path = save_dataframe_csv(summary, stats_dir / "cluster_summary.csv")
    return hdf5_path, csv_path


def load_decoding_cluster_results_hdf5(path: Path) -> DecodingClusterResults:
    if not path.exists():
        raise FileNotFoundError(f"Missing cluster results: {path}")
    with h5py.File(path, "r") as handle:
        t_values = np.asarray(handle["t-values"])
        p_values = np.asarray(handle["p-values"])
        h0 = np.asarray(handle["h0"])
        clusters = _read_clusters(handle)
    return DecodingClusterResults(t_values=t_values, clusters=clusters, p_values=p_values, h0=h0)


def _read_clusters(handle: h5py.File) -> list[tuple[np.ndarray, np.ndarray]]:
    clusters: list[tuple[np.ndarray, np.ndarray]] = []
    index = 0
    while True:
        train_key = f"clusters/train-{index}"
        test_key = f"clusters/test-{index}"
        if train_key not in handle or test_key not in handle:
            break
        train_idx = np.asarray(handle[train_key], dtype=int)
        test_idx = np.asarray(handle[test_key], dtype=int)
        clusters.append((train_idx, test_idx))
        index += 1
    return clusters
