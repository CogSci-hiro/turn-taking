"""turntaking.analysis.features.erp

ERP feature extraction.
"""

from __future__ import annotations

from typing import Sequence

import mne
import numpy as np

from turntaking.analysis.features.roi import pick_roi


def roi_time_window_mean_uv(
    epochs: mne.BaseEpochs,
    *,
    tmin: float,
    tmax: float,
    roi_channels: Sequence[str],
) -> np.ndarray:
    """Mean ERP amplitude (µV) per epoch within ROI and time window.

    Parameters
    ----------
    epochs
        Input epochs.
    tmin, tmax
        Time window in seconds (inclusive).
    roi_channels
        Channel names defining the ROI.

    Returns
    -------
    values_uv
        Array of shape ``(n_epochs,)`` containing mean amplitude in µV.

    Usage example
    -------------
        n1 = roi_time_window_mean_uv(
            epochs,
            tmin=0.08,
            tmax=0.12,
            roi_channels=["Fz", "FCz"],
        )
    """
    roi_picks = pick_roi(epochs, roi_channels)

    data = epochs.get_data(copy=True)  # (n_epochs, n_channels, n_times)
    time_mask = (epochs.times >= tmin) & (epochs.times <= tmax)
    if not np.any(time_mask):
        raise ValueError(
            f"Time window [{tmin}, {tmax}] does not overlap epochs.times "
            f"range [{epochs.times.min()}, {epochs.times.max()}]."
        )

    values_volts = data[:, roi_picks, :][:, :, time_mask].mean(axis=(1, 2))
    values_uv = values_volts * 1e6
    return values_uv
