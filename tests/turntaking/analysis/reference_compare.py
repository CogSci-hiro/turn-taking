
"""Reusable helpers for numerical similarity checks against reference artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _numeric_similarity(actual: np.ndarray, reference: np.ndarray) -> SimilarityReport:
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


def compare_arrays(actual: Any, reference: Any) -> SimilarityReport:
    """Compute similarity metrics for numeric arrays or mixed-type DataFrames."""
    if isinstance(actual, pd.DataFrame) or isinstance(reference, pd.DataFrame):
        if not isinstance(actual, pd.DataFrame) or not isinstance(reference, pd.DataFrame):
            raise ValueError("Both artifacts must be DataFrames when one artifact is a DataFrame.")

        if list(actual.columns) != list(reference.columns):
            raise ValueError(
                f"CSV column mismatch: actual={list(actual.columns)}, reference={list(reference.columns)}"
            )
        if actual.shape != reference.shape:
            raise ValueError(f"Shape mismatch: actual={actual.shape}, reference={reference.shape}")

        numeric_cols: list[str] = []
        non_numeric_cols: list[str] = []
        for col in actual.columns:
            is_num = pd.api.types.is_numeric_dtype(actual[col]) and pd.api.types.is_numeric_dtype(reference[col])
            if is_num:
                numeric_cols.append(col)
            else:
                non_numeric_cols.append(col)

        for col in non_numeric_cols:
            if not actual[col].equals(reference[col]):
                raise ValueError(f"Non-numeric column mismatch in '{col}'.")

        if not numeric_cols:
            return SimilarityReport(
                max_abs_error=0.0,
                mean_abs_error=0.0,
                pearson_r=1.0,
                shape=actual.shape,
            )

        report = _numeric_similarity(
            actual[numeric_cols].to_numpy(dtype=float),
            reference[numeric_cols].to_numpy(dtype=float),
        )
        return SimilarityReport(
            max_abs_error=report.max_abs_error,
            mean_abs_error=report.mean_abs_error,
            pearson_r=report.pearson_r,
            shape=actual.shape,
        )

    return _numeric_similarity(np.asarray(actual), np.asarray(reference))


def assert_similarity(
    actual: Any,
    reference: Any,
    thresholds: SimilarityThresholds,
) -> SimilarityReport:
    """Assert actual results are numerically close to reference within configured tolerances."""
    report = compare_arrays(actual, reference)
    assert report.max_abs_error <= thresholds.max_abs_error, report
    assert report.mean_abs_error <= thresholds.mean_abs_error, report
    assert report.pearson_r >= thresholds.min_pearson_r, report
    return report


def load_supported_artifact(path: Path) -> np.ndarray | pd.DataFrame:
    """Load an artifact from `.npy` or `.csv` for cross-run comparison."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path))
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported artifact extension for similarity comparison: {suffix}")
