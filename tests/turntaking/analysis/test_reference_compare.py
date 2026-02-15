
"""Reference-metric unit tests for analysis regression tooling."""

import numpy as np
import pandas as pd
import pytest

from .reference_compare import (
    SimilarityThresholds,
    assert_similarity,
    compare_arrays,
)


def test_compare_arrays_reports_expected_metrics():
    """Validates similarity metric calculations so regression failures are interpretable and stable."""
    actual = np.array([1.0, 2.0, 3.0])
    reference = np.array([1.0, 2.1, 2.9])

    report = compare_arrays(actual, reference)
    assert report.shape == (3,)
    assert report.max_abs_error == pytest.approx(0.1)
    assert report.mean_abs_error == pytest.approx((0.0 + 0.1 + 0.1) / 3)
    assert 0.99 <= report.pearson_r <= 1.0


def test_assert_similarity_passes_within_thresholds():
    """Confirms threshold assertions succeed when numeric drift remains inside accepted limits."""
    actual = np.array([0.5, 0.7, 0.9])
    reference = np.array([0.500001, 0.700001, 0.900001])
    thresholds = SimilarityThresholds(max_abs_error=1e-3, mean_abs_error=1e-3, min_pearson_r=0.99)

    report = assert_similarity(actual, reference, thresholds)
    assert report.max_abs_error < 1e-3


def test_assert_similarity_fails_when_thresholds_violated():
    """Ensures large deviations are caught, preventing unnoticed behavioral regressions."""
    actual = np.array([0.0, 0.0, 0.0])
    reference = np.array([1.0, 1.0, 1.0])
    thresholds = SimilarityThresholds(max_abs_error=0.1, mean_abs_error=0.1, min_pearson_r=0.99)

    with pytest.raises(AssertionError):
        assert_similarity(actual, reference, thresholds)


def test_assert_similarity_supports_mixed_dataframe_values():
    """Allows CSV-style tables with string labels while applying numeric similarity checks to numeric columns."""
    actual = pd.DataFrame(
        {
            "score": [0.10, 0.20, 0.30],
            "condition": ["fast", "slow", "slow"],
            "subject": ["sub-001", "sub-001", "sub-002"],
        }
    )
    reference = pd.DataFrame(
        {
            "score": [0.1005, 0.2005, 0.3005],
            "condition": ["fast", "slow", "slow"],
            "subject": ["sub-001", "sub-001", "sub-002"],
        }
    )
    thresholds = SimilarityThresholds(max_abs_error=0.01, mean_abs_error=0.01, min_pearson_r=0.99)
    report = assert_similarity(actual, reference, thresholds)
    assert report.shape == (3, 3)


def test_assert_similarity_rejects_non_numeric_column_mismatch():
    """Fails fast when categorical/string columns differ even if numeric columns are close."""
    actual = pd.DataFrame({"score": [0.1, 0.2], "condition": ["fast", "slow"]})
    reference = pd.DataFrame({"score": [0.1, 0.2], "condition": ["fast", "fast"]})
    thresholds = SimilarityThresholds()
    with pytest.raises(ValueError, match="Non-numeric column mismatch"):
        assert_similarity(actual, reference, thresholds)
