from __future__ import annotations

"""Tests for ROI ERP time-window averaging helper."""

import mne
import numpy as np
import pytest

from turntaking.analysis.features.erp import roi_time_window_mean_uv


def _epochs() -> mne.Epochs:
    info = mne.create_info(["Fz", "Pz"], sfreq=10.0, ch_types=["eeg", "eeg"])
    # 2 epochs, 2 channels, 5 times -> times: [0.0, 0.1, ..., 0.4]
    data = np.array(
        [
            [[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]],
            [[2, 3, 4, 5, 6], [1, 3, 5, 7, 9]],
        ],
        dtype=float,
    )
    return mne.EpochsArray(data, info, tmin=0.0, verbose=False)


def test_roi_time_window_mean_uv_computes_expected_values():
    """Checks mean over ROI+time converts volts to microvolts exactly."""
    epochs = _epochs()
    out = roi_time_window_mean_uv(epochs, tmin=0.1, tmax=0.3, roi_channels=["Fz"])

    # For epoch1 Fz at t=[0.1,0.2,0.3] => [2,3,4] mean=3 ; epoch2 => [3,4,5] mean=4
    np.testing.assert_allclose(out, np.array([3.0, 4.0]) * 1e6)


def test_roi_time_window_mean_uv_validates_time_window():
    """Ensures out-of-range windows fail clearly rather than returning empty means."""
    epochs = _epochs()
    with pytest.raises(ValueError, match="outside epochs.times range"):
        roi_time_window_mean_uv(epochs, tmin=2.0, tmax=3.0, roi_channels=["Fz"])
