"""
Epoch selection and contrast splitting.

This module defines the *behavioral* rules that turn a pool of epochs into the
two-class contrasts used throughout the pipeline.

Two orthogonal concepts are used repeatedly across domains:

- Selection: include/exclude epochs using metadata thresholds.
- Splitting: convert continuous metadata into two labels by median split.

Downstream modules assume:

- input is an MNE ``Epochs`` (or ``BaseEpochs``) object
- ``epochs.metadata`` exists and contains numeric columns required for the
  selected contrast
"""

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
    """
    Apply metadata-based epoch selection.

    Parameters
    ----------
    epochs
        Input epochs. Must include a ``pandas.DataFrame`` in ``epochs.metadata``.
    params
        Thresholds to apply.

    Returns
    -------
    selected
        A new ``BaseEpochs`` view with only epochs satisfying the constraints.
    """
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
    """
    Split epochs into two classes by median split on metadata.

    Parameters
    ----------
    epochs
        Input epochs with ``epochs.metadata`` present.
    contrast
        Which metadata dimension to split on:

        - ``"latency"``: split on the ``latency`` column (fast vs slow).
        - ``"duration"``: split on the ``self_duration`` column (long vs short).

    Returns
    -------
    cond_1
        First class epochs (fast or long).
    cond_2
        Second class epochs (slow or short).
    labels
        Mapping with keys ``cond_1`` and ``cond_2`` for consistent filenames and
        figure labels.

    Notes
    -----
    Epochs with values exactly equal to the median are excluded to avoid label
    ambiguity, which can slightly reduce trial counts.
    """
    if epochs.metadata is None:
        raise ValueError("epochs.metadata is required for splitting.")

    md = epochs.metadata

    if contrast == "latency":
        # Smaller latency -> "fast", larger latency -> "slow"
        values = md["latency"].astype(float).to_numpy()
        median = float(np.median(values))
        mask_1 = values < median
        mask_2 = values > median
        labels = {"cond_1": "fast", "cond_2": "slow"}
    else:
        # Larger self-duration -> "long", smaller -> "short"
        values = md["self_duration"].astype(float).to_numpy()
        median = float(np.median(values))
        mask_1 = values > median   # long
        mask_2 = values < median   # short
        labels = {"cond_1": "long", "cond_2": "short"}

    cond_1 = epochs[mask_1]
    cond_2 = epochs[mask_2]

    mne.epochs.equalize_epoch_counts([cond_1, cond_2])
    return cond_1, cond_2, labels
