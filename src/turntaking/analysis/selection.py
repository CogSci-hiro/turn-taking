"""turntaking.analysis.selection

Epoch selection and condition splitting.

Assumes ``epochs.metadata`` contains at least:
- ``latency`` (seconds)
- ``self_duration`` (seconds)

Other columns are preserved.

Notes
-----
Selection and splitting operate on metadata using MNE's pandas-query selection
syntax (e.g., ``epochs["latency < 0.5"]``). This keeps the logic transparent
and easy to reproduce from logs/configs.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mne
import pandas as pd

Contrast = Literal["latency", "duration"]


@dataclass(frozen=True)
class SelectionParams:
    """Epoch inclusion criteria.

    Parameters
    ----------
    min_latency
        Keep epochs with ``latency > min_latency`` (seconds).
    max_latency
        Keep epochs with ``latency < max_latency`` (seconds).
    min_self_duration
        Keep epochs with ``self_duration > min_self_duration`` (seconds).

    Usage example
    -------------
        params = SelectionParams(min_latency=0.05, max_latency=1.0, min_self_duration=0.2)
        epochs_sel = select_epochs(epochs, params)
    """

    min_latency: float
    max_latency: float
    min_self_duration: float


def _require_metadata_columns(metadata: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in metadata.columns]
    if missing:
        raise ValueError(
            "epochs.metadata is missing required columns: "
            f"{missing}. Available columns: {list(metadata.columns)}"
        )


def select_epochs(epochs: mne.BaseEpochs, params: SelectionParams) -> mne.BaseEpochs:
    """Apply metadata-based epoch selection.

    Parameters
    ----------
    epochs
        Input epochs with metadata.
    params
        Selection parameters.

    Returns
    -------
    selected_epochs
        A (potentially) subset of the input epochs.

    Usage example
    -------------
        epochs_sel = select_epochs(
            epochs,
            SelectionParams(min_latency=0.05, max_latency=1.0, min_self_duration=0.2),
        )
    """
    if epochs.metadata is None:
        raise ValueError("epochs.metadata is required for selection.")
    _require_metadata_columns(epochs.metadata, ["latency", "self_duration"])

    epochs = epochs[f"latency > {params.min_latency}"]
    epochs = epochs[f"latency < {params.max_latency}"]
    epochs = epochs[f"self_duration > {params.min_self_duration}"]
    return epochs


def split_epochs_median(
    epochs: mne.BaseEpochs,
    contrast: Contrast,
) -> tuple[mne.BaseEpochs, mne.BaseEpochs, dict[str, str]]:
    """Median split epochs into two conditions.

    Parameters
    ----------
    epochs
        Input epochs (typically already selected).
    contrast
        Which metadata variable to split on:
        - ``"latency"`` -> returns (fast, slow)
        - ``"duration"`` -> returns (long, short)

    Returns
    -------
    cond_1
        For latency: ``fast``. For duration: ``long``.
    cond_2
        For latency: ``slow``. For duration: ``short``.
    labels
        Mapping with keys ``{"cond_1", "cond_2"}`` for stable naming.

    Notes
    -----
    Epoch counts are equalized across conditions via
    :func:`mne.epochs.equalize_epoch_counts`.

    Usage example
    -------------
        fast, slow, labels = split_epochs_median(epochs_sel, contrast="latency")
    """
    if epochs.metadata is None:
        raise ValueError("epochs.metadata is required for splitting.")

    if contrast == "latency":
        _require_metadata_columns(epochs.metadata, ["latency"])
        median = float(epochs.metadata["latency"].median())
        cond_1 = epochs[f"latency < {median}"]
        cond_2 = epochs[f"latency > {median}"]
        labels = {"cond_1": "fast", "cond_2": "slow"}
    else:
        _require_metadata_columns(epochs.metadata, ["self_duration"])
        median = float(epochs.metadata["self_duration"].median())
        cond_1 = epochs[f"self_duration > {median}"]
        cond_2 = epochs[f"self_duration < {median}"]
        labels = {"cond_1": "long", "cond_2": "short"}

    mne.epochs.equalize_epoch_counts([cond_1, cond_2])
    return cond_1, cond_2, labels
