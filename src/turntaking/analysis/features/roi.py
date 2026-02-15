from typing import Sequence

import mne


def pick_roi(epochs: mne.BaseEpochs, roi_channels: Sequence[str]) -> list[int]:
    """Return channel indices for a given ROI."""
    # Use ordered=False so missing ROI channels result in an empty selection
    # that we can validate with a stable project-specific error message.
    picks = mne.pick_channels(epochs.ch_names, include=list(roi_channels), ordered=False)
    if len(picks) == 0:
        raise ValueError("ROI picks are empty. Check channel names/ROI definition.")
    return list(picks)
