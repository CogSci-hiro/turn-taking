"""
Epoch selection and condition splitting.
"""


from dataclasses import dataclass
from typing import Literal

import mne

Contrast = Literal["latency", "duration"]


@dataclass(frozen=True)
class SelectionParams:
    """Epoch inclusion criteria."""
    min_latency: float
    max_latency: float
    min_self_duration: float


def select_epochs(
    epochs: mne.BaseEpochs,
    params: SelectionParams,
) -> mne.BaseEpochs:
    """Apply metadata-based epoch selection."""
    raise NotImplementedError


def split_epochs_median(
    epochs: mne.BaseEpochs,
    contrast: Contrast,
) -> tuple[mne.BaseEpochs, mne.BaseEpochs, dict[str, str]]:
    """
    Median split epochs into two conditions.

    Returns
    -------
    cond_1, cond_2
    labels : dict
        Mapping {"cond_1": name1, "cond_2": name2}
    """
    raise NotImplementedError
