from __future__ import annotations

"""Reusable helpers for numerical similarity checks against reference artifacts."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimilarityThresholds:
    """Similarity tolerances for numerical comparisons against a reference artifact."""

    max_abs_error: float = 1e-6
    mean_abs_error: float = 1e-8
    min_pearson_r: float = 0.999


@dataclass(frozen=True)
class SimilarityReport:
    """Measured similarity summary between two arrays."""

    max_abs_error: float
    mean_abs_error: float
    pearson_r: float
    shape: tuple[int, ...]


def compare_arrays(actual: np.ndarray, reference: np.ndarray) -> SimilarityReport:
    """Compute stable similarity metrics used by regression tests."""
    actual = np.asarray(actual, dtype=float)
    reference = np.asarray(reference, dtype=float)

    if actual.shape != reference.shape:
        raise ValueError(f"Shape mismatch: actual={actual.shape}, reference={reference.shape}")

    diff = np.abs(actual - reference)
    flat_a = actual.ravel()
    flat_r = reference.ravel()

    if flat_a.size <= 1:
        corr = 1.0 if np.allclose(flat_a, flat_r) else 0.0
    elif np.std(flat_a) == 0 and np.std(flat_r) == 0:
        corr = 1.0 if np.allclose(flat_a, flat_r) else 0.0
    else:
        corr = float(np.corrcoef(flat_a, flat_r)[0, 1])

    return SimilarityReport(
        max_abs_error=float(diff.max()) if diff.size else 0.0,
        mean_abs_error=float(diff.mean()) if diff.size else 0.0,
        pearson_r=corr,
        shape=actual.shape,
    )


def assert_similarity(
    actual: np.ndarray,
    reference: np.ndarray,
    thresholds: SimilarityThresholds,
) -> SimilarityReport:
    """Assert actual results are numerically close to reference within configured tolerances."""
    report = compare_arrays(actual, reference)
    assert report.max_abs_error <= thresholds.max_abs_error, report
    assert report.mean_abs_error <= thresholds.mean_abs_error, report
    assert report.pearson_r >= thresholds.min_pearson_r, report
    return report


def load_supported_artifact(path: Path) -> np.ndarray:
    """Load an artifact from `.npy` or `.csv` into a NumPy array for cross-run comparison."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path))
    if suffix == ".csv":
        return pd.read_csv(path).to_numpy()
    raise ValueError(f"Unsupported artifact extension for similarity comparison: {suffix}")
