"""turntaking.analysis.selection

Epoch selection and condition splitting.

Assumes epochs.metadata contains at least:
- latency
- self_duration

(Other columns may be present and are preserved.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mne

Contrast = Literal["latency", "duration"]


@dataclass(frozen=True)
class SelectionParams:
    """Epoch inclusion criteria.

    Parameters
    ----------
    min_latency, max_latency
        Keep epochs with ``min_latency < latency < max_latency`` (seconds).
    min_self_duration
        Keep epochs with ``self_duration > min_self_duration`` (seconds).
    """

    min_latency: float
    max_latency: float
    min_self_duration: float


def select_epochs(epochs: mne.BaseEpochs, params: SelectionParams) -> mne.BaseEpochs:
    """Apply metadata-based epoch selection."""
    raise NotImplementedError


def split_epochs_median(
    epochs: mne.BaseEpochs,
    contrast: Contrast,
) -> tuple[mne.BaseEpochs, mne.BaseEpochs, dict[str, str]]:
    """Median-split epochs into two conditions.

    Returns
    -------
    cond_1, cond_2
        For ``contrast='latency'``: (fast, slow)
        For ``contrast='duration'``: (long, short)
    labels
        Mapping {"cond_1": name1, "cond_2": name2}

    Usage example
    -------------
        cond_1, cond_2, labels = split_epochs_median(epochs, contrast="latency")
    """
    raise NotImplementedError
