
"""Tests for sample-based time-margin cropping utilities."""

import numpy as np
import pytest

from turntaking.stats.cropping import crop_time_margins_samples


def test_crop_time_margins_samples_returns_expected_slice_and_indices():
    """Validates exact sample-index cropping math so ERP/TFR windows stay aligned with requested margins."""
    X = np.arange(2 * 10 * 3).reshape(2, 10, 3)
    out, start, end = crop_time_margins_samples(
        X, sfreq=10.0, left_margin=0.2, right_margin=0.3
    )

    assert start == 2
    assert end == 7
    np.testing.assert_array_equal(out, X[:, 2:7, :])


def test_crop_time_margins_samples_handles_zero_right_margin():
    """Confirms right margin of zero keeps the right edge intact, avoiding accidental truncation."""
    X = np.zeros((1, 8, 2))
    out, start, end = crop_time_margins_samples(
        X, sfreq=4.0, left_margin=0.5, right_margin=0.0
    )

    assert start == 2
    assert end == 8
    assert out.shape == (1, 6, 2)


def test_crop_time_margins_samples_rejects_invalid_input_shapes():
    """Prevents silent misuse by requiring the documented 3D array contract."""
    with pytest.raises(ValueError, match="X must be 3D"):
        crop_time_margins_samples(
            np.zeros((4, 5)),
            sfreq=10.0,
            left_margin=0.0,
            right_margin=0.0,
        )


def test_crop_time_margins_samples_rejects_negative_margins():
    """Protects against invalid negative windows that would make cropping semantics ambiguous."""
    with pytest.raises(ValueError, match="must be >= 0"):
        crop_time_margins_samples(
            np.zeros((1, 10, 1)),
            sfreq=10.0,
            left_margin=-0.1,
            right_margin=0.0,
        )


def test_crop_time_margins_samples_rejects_full_removal():
    """Ensures errors are raised when requested margins would remove all time samples."""
    with pytest.raises(ValueError, match="removes all data"):
        crop_time_margins_samples(
            np.zeros((1, 10, 1)),
            sfreq=10.0,
            left_margin=0.4,
            right_margin=0.6,
        )
