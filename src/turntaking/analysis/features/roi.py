"""turntaking.analysis.features.roi

ROI utilities and validation.
"""

from __future__ import annotations

from typing import Sequence

import mne


def pick_roi(epochs: mne.BaseEpochs, roi_channels: Sequence[str]) -> list[int]:
    """Return channel indices for a given ROI.

    Parameters
    ----------
    epochs
        Input epochs.
    roi_channels
        Channel names defining the ROI.

    Returns
    -------
    picks
        Indices into ``epochs.ch_names`` corresponding to the ROI.

    Raises
    ------
    ValueError
        If none of the requested ROI channels are present.

    Usage example
    -------------
        picks = pick_roi(epochs, ["Fz", "FCz"])
    """
    picks = mne.pick_channels(ch_names=epochs.ch_names, include=list(roi_channels))
    if len(picks) == 0:
        raise ValueError(
            "ROI picks are empty. None of these channels were found: "
            f"{list(roi_channels)}. Available channels: {epochs.ch_names}"
        )
    return list(picks)
