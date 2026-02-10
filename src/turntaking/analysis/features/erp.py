"""
ERP feature extraction.
"""


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
    """
    Mean ERP amplitude (µV) per epoch within ROI and time window.
    """
    raise NotImplementedError
