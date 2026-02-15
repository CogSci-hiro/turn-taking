
"""Tests for mixed-effect EEG feature helper functions."""

import mne
import numpy as np
import pandas as pd
import pytest

from turntaking.analysis.mixed_effect.eeg_features import (
    _time_window_mean,
    _time_window_mean_uV,
    compute_run_eeg_features,
)


def _epochs() -> mne.Epochs:
    info = mne.create_info(["Fz", "Pz"], sfreq=10.0, ch_types=["eeg", "eeg"])
    data = np.array(
        [
            [[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]],
            [[2, 3, 4, 5, 6], [1, 3, 5, 7, 9]],
        ],
        dtype=float,
    )
    return mne.EpochsArray(data, info, tmin=0.0, verbose=False)


def test_time_window_mean_and_uv_scaling():
    """Validates per-trial ROI/window averaging and microvolt conversion."""
    epochs = _epochs()
    vals = _time_window_mean(epochs, 0.1, 0.3, ["Fz"])
    assert vals == [3.0, 4.0]

    vals_uv = _time_window_mean_uV(epochs, 0.1, 0.3, ["Fz"])
    np.testing.assert_allclose(vals_uv, np.array([3.0, 4.0]) * 1e6)


def test_time_window_mean_validates_window_and_picks():
    """Ensures invalid windows or missing channels are rejected explicitly."""
    epochs = _epochs()
    with pytest.raises(ValueError, match="No samples in time window"):
        _time_window_mean(epochs, 2.0, 3.0, ["Fz"])
    with pytest.raises(ValueError, match="No requested picks"):
        _time_window_mean(epochs, 0.1, 0.2, ["Oz"])


def test_compute_run_eeg_features_returns_expected_columns(monkeypatch):
    """Checks final mixed-effect EEG feature table schema remains stable."""
    epochs = _epochs()

    def fake_band_power_epochs(ep, *, fmin, fmax):
        # keep deterministic and avoid expensive filtering in unit test
        return ep

    monkeypatch.setattr("turntaking.analysis.mixed_effect.eeg_features._band_power_epochs", fake_band_power_epochs)
    out = compute_run_eeg_features(
        epochs,
        tw1_tmin=0.0,
        tw1_tmax=0.1,
        tw2_tmin=0.2,
        tw2_tmax=0.3,
        baseline_tmin=0.0,
        baseline_tmax=0.0,
        anterior_picks=["Fz"],
        posterior_picks=["Pz"],
    )

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 2
    expected_cols = {
        "tw1_mean_anterior",
        "tw1_mean_posterior",
        "tw2_mean_anterior",
        "tw2_mean_posterior",
        "tw1_alpha_anterior",
        "tw2_beta_posterior",
        "baseline_mean_anterior",
        "baseline_mean_posterior",
    }
    assert expected_cols.issubset(set(out.columns))
