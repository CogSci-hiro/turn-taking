
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
    """Mean ERP amplitude (µV) per epoch within ROI and time window."""
    picks = pick_roi(epochs, roi_channels)
    data = epochs.get_data(copy=True)  # (n_epochs, n_ch, n_times)
    tmask = (epochs.times >= tmin) & (epochs.times <= tmax)
    if not np.any(tmask):
        raise ValueError(f"Time window [{tmin}, {tmax}] is outside epochs.times range.")
    return data[:, picks][:, :, tmask].mean(axis=(1, 2)) * 1e6
