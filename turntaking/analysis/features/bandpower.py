"""turntaking.analysis.features.bandpower

Band-limited power features.
"""


from typing import Sequence

import mne
import numpy as np


def compute_bandpower_hilbert(
    epochs: mne.BaseEpochs,
    *,
    fmin: float,
    fmax: float,
) -> mne.BaseEpochs:
    """Band-pass + Hilbert envelope (copy-safe)."""
    raise NotImplementedError


def roi_time_window_mean_power(
    epochs: mne.BaseEpochs,
    *,
    tmin: float,
    tmax: float,
    roi_channels: Sequence[str],
) -> np.ndarray:
    """Mean bandpower per epoch within ROI and time window."""
    raise NotImplementedError
