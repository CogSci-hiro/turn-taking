
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import h5py
import numpy as np

ContrastName = Literal["latency", "duration"]


@dataclass(frozen=True)
class Hdf5CacheParams:
    """
    Parameters for HDF5 feature caching.

    Attributes
    ----------
    compression
        HDF5 compression algorithm ("gzip" or None).
    compression_level
        Gzip level (1-9) if compression="gzip".
    x_dtype
        dtype for cached X, typically float32 for size/perf.
    """

    compression: str | None = "gzip"
    compression_level: int = 4
    x_dtype: str = "float32"


# =============================================================================
#                     ########################################
#                     #         DECODING I/O (ERP)            #
#                     ########################################
# =============================================================================


def get_decoding_out_dir(out_dir: Path, contrast: ContrastName) -> Path:
    return out_dir / "decoding" / "erp" / contrast


def save_decoding_scores(
    *,
    out_dir: Path,
    contrast: ContrastName,
    scores: np.ndarray,
    times_s: np.ndarray,
) -> Tuple[Path, Path]:
    """
    Save decoding outputs.

    Writes:
    - scores.npy : (n_subjects, n_splits, n_times, n_times)
    - times.npy  : (n_times,)

    Returns
    -------
    scores_path, times_path
        Paths to saved files.

    Usage example
    -------------
        save_decoding_scores(out_dir=Path("out"), contrast="duration", scores=scores, times_s=times_s)
    """
    out_path = get_decoding_out_dir(out_dir, contrast)
    out_path.mkdir(parents=True, exist_ok=True)

    scores_path = out_path / "scores.npy"
    times_path = out_path / "times.npy"

    np.save(scores_path, scores)
    np.save(times_path, np.asarray(times_s, dtype=np.float64))

    return scores_path, times_path


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
    """
    Save per-subject decoding features to HDF5.

    Structure:
    - /X      (n_trials, n_channels, n_times) float32 (by default), chunked + compressed
    - /y      (n_trials,) int8
    - /times  (n_times,) float64

    Usage example
    -------------
        save_subject_feature_cache_hdf5(
            out_dir=Path("out"), contrast="duration", subject="sub-001",
            X=X, y=y, times_s=times_s, cache_params=Hdf5CacheParams()
        )
    """
    cache_path = get_feature_cache_path(out_dir, contrast, subject)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    X_to_write = np.asarray(X, dtype=np.dtype(cache_params.x_dtype))
    y_to_write = np.asarray(y, dtype=np.int8)
    times_to_write = np.asarray(times_s, dtype=np.float64)

    compression = cache_params.compression
    compression_opts = cache_params.compression_level if compression == "gzip" else None

    # Chunk along trials for reasonable read/write performance.
    trial_chunk = min(32, max(1, X_to_write.shape[0]))
    chunks = (trial_chunk, X_to_write.shape[1], X_to_write.shape[2])

    with h5py.File(cache_path, "w") as h5:
        h5.create_dataset(
            "X",
            data=X_to_write,
            compression=compression,
            compression_opts=compression_opts,
            chunks=chunks,
        )
        h5.create_dataset("y", data=y_to_write)
        h5.create_dataset("times", data=times_to_write)

    return cache_path


def load_subject_feature_cache_hdf5(
    *,
    out_dir: Path,
    contrast: ContrastName,
    subject: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load per-subject decoding features from HDF5.

    Returns
    -------
    X, y, times_s
        X shape (n_trials, n_channels, n_times), y shape (n_trials,), times shape (n_times,)

    Usage example
    -------------
        X, y, times_s = load_subject_feature_cache_hdf5(
            out_dir=Path("out"), contrast="duration", subject="sub-001"
        )
    """
    cache_path = get_feature_cache_path(out_dir, contrast, subject)
    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")

    with h5py.File(cache_path, "r") as h5:
        X = np.asarray(h5["X"])
        y = np.asarray(h5["y"])
        times_s = np.asarray(h5["times"])

    return X, y, times_s


def save_decoding_outputs(
    *,
    out_dir: Path,
    contrast: ContrastName,
    scores: np.ndarray,
    times_s: np.ndarray,
) -> Tuple[Path, Path]:
    out_path = out_dir / "decoding" / "erp" / contrast
    out_path.mkdir(parents=True, exist_ok=True)

    scores_path = out_path / "scores.npy"
    times_path = out_path / "times.npy"

    np.save(scores_path, scores)
    np.save(times_path, np.asarray(times_s, dtype=np.float64))

    return scores_path, times_path
