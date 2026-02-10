
from typing import Sequence

import mne


def pick_roi(epochs: mne.BaseEpochs, roi_channels: Sequence[str]) -> list[int]:
    """Return channel indices for a given ROI."""
    picks = mne.pick_channels(epochs.ch_names, include=list(roi_channels))
    if len(picks) == 0:
        raise ValueError("ROI picks are empty. Check channel names/ROI definition.")
    return list(picks)
