"""turntaking.analysis.features.roi

ROI utilities and validation.
"""


from typing import Sequence

import mne


def pick_roi(epochs: mne.BaseEpochs, roi_channels: Sequence[str]) -> list[int]:
    """Return channel indices for a given ROI.

    Usage example
    -------------
        picks = pick_roi(epochs, ["Fz", "FCz"])  # indices
    """
    raise NotImplementedError
