
from dataclasses import dataclass
from typing import Literal

import mne
import numpy as np

Contrast = Literal["latency", "duration"]


@dataclass(frozen=True)
class SelectionParams:
    """Epoch inclusion criteria.

    Parameters
    ----------
    min_latency, max_latency
        Keep epochs with `min_latency < latency < max_latency` (seconds).
    min_self_duration
        Keep epochs with `self_duration > min_self_duration` (seconds).

    Notes
    -----
    Requires `epochs.metadata` to include at least:
    - latency
    - self_duration
    """
    min_latency: float
    max_latency: float
    min_self_duration: float


def select_epochs(epochs: mne.BaseEpochs, params: SelectionParams) -> mne.BaseEpochs:
    """Apply metadata-based epoch selection."""
    if epochs.metadata is None:
        raise ValueError("epochs.metadata is required for selection.")

    required_cols = {"latency", "self_duration"}
    missing = required_cols.difference(set(epochs.metadata.columns))
    if missing:
        raise ValueError(f"epochs.metadata missing required columns: {sorted(missing)}")

    out = epochs[f"{params.min_latency} < latency < {params.max_latency}"]
    out = out[f"self_duration > {params.min_self_duration}"]
    return out


def split_epochs_median(
    epochs: mne.BaseEpochs,
    contrast: Contrast,
) -> tuple[mne.BaseEpochs, mne.BaseEpochs, dict[str, str]]:
    if epochs.metadata is None:
        raise ValueError("epochs.metadata is required for splitting.")

    md = epochs.metadata

    if contrast == "latency":
        values = md["latency"].astype(float).to_numpy()
        median = float(np.median(values))
        mask_1 = values < median
        mask_2 = values > median
        labels = {"cond_1": "fast", "cond_2": "slow"}
    else:
        values = md["self_duration"].astype(float).to_numpy()
        median = float(np.median(values))
        mask_1 = values > median   # long
        mask_2 = values < median   # short
        labels = {"cond_1": "long", "cond_2": "short"}

    cond_1 = epochs[mask_1]
    cond_2 = epochs[mask_2]

    mne.epochs.equalize_epoch_counts([cond_1, cond_2])
    return cond_1, cond_2, labels
