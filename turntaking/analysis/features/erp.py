"""turntaking.analysis.features.erp

ERP feature extraction.
"""

from __future__ import annotations

from typing import Sequence

import mne
import numpy as np


def roi_time_window_mean_uv(
    epochs: mne.BaseEpochs,
    *,
    tmin: float,
    tmax: float,
    roi_channels: Sequence[str],
) -> np.ndarray:
    """Mean ERP amplitude (µV) per epoch within ROI and time window.

    Returns
    -------
    values
        Shape (n_epochs,)

    Usage example
    -------------
        x = roi_time_window_mean_uv(epochs, tmin=0.1, tmax=0.2, roi_channels=["Cz"])
    """
    raise NotImplementedError
