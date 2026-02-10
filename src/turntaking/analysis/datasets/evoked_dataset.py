"""turntaking.analysis.datasets.evoked_dataset

ERP / TFR group-level dataset construction.

This module builds group-level datasets from lists of epochs files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mne
import pandas as pd

from turntaking.analysis.io import load_epochs, parse_epochs_filepath
from turntaking.analysis.selection import SelectionParams, select_epochs, split_epochs_median

Kind = Literal["erp", "tfr"]


@dataclass
class EvokedDatasetResult:
    """Outputs of an evoked dataset build.

    Attributes
    ----------
    evokeds_cond1
        Per-file evokeds for condition 1 (e.g., fast or long).
    evokeds_cond2
        Per-file evokeds for condition 2 (e.g., slow or short).
    difference
        Grand-average difference (cond2 - cond1) across files.
    metadata
        Concatenated epoch metadata with added columns:
        - ``subject`` (str)
        - ``run`` (int)
        - ``condition`` (str)

        Example table (columns truncated)::

            subject  run condition  latency  self_duration  other_duration  timestamp
            005      1   fast       0.12     1.80           2.10           123.45
            005      1   slow       0.56     1.20           1.30           130.02

    n_trials
        Trial counts per file and condition.

        Example table::

            subject  run condition  n_epochs
            005      1   fast       120
            005      1   slow       120
    """

    evokeds_cond1: list[mne.Evoked]
    evokeds_cond2: list[mne.Evoked]
    difference: mne.Evoked
    metadata: pd.DataFrame
    n_trials: pd.DataFrame


def _difference_evoked(evoked_cond1: mne.Evoked, evoked_cond2: mne.Evoked) -> mne.Evoked:
    """Compute cond2 - cond1 as an Evoked."""
    diff = mne.combine_evoked([evoked_cond2, evoked_cond1], weights=[1.0, -1.0])
    diff.comment = "cond2_minus_cond1"
    return diff


def build_evoked_dataset(
    epoch_paths: list[Path],
    *,
    kind: Kind,
    contrast: Literal["latency", "duration"],
    selection_params: SelectionParams,
) -> EvokedDatasetResult:
    """Build a group-level ERP dataset from epoch files.

    Parameters
    ----------
    epoch_paths
        List of epochs FIF files.
    kind
        ``"erp"`` (implemented) or ``"tfr"`` (reserved for later).
    contrast
        ``"latency"`` or ``"duration"``; determines the median split.
    selection_params
        Metadata-based selection parameters.

    Returns
    -------
    result
        :class:`~turntaking.analysis.datasets.evoked_dataset.EvokedDatasetResult`

    Notes
    -----
    This function intentionally does *not* write to disk. Use CLI wrappers or
    :mod:`turntaking.analysis.io` for saving.

    Usage example
    -------------
        result = build_evoked_dataset(
            epoch_paths,
            kind="erp",
            contrast="latency",
            selection_params=SelectionParams(0.05, 1.0, 0.2),
        )
    """
    if kind != "erp":
        raise NotImplementedError("Only kind='erp' is implemented in the first vertical slice.")

    evokeds_cond1: list[mne.Evoked] = []
    evokeds_cond2: list[mne.Evoked] = []
    diffs: list[mne.Evoked] = []
    metadata_rows: list[pd.DataFrame] = []
    n_trials_rows: list[dict[str, object]] = []

    for epoch_path in epoch_paths:
        file_info = parse_epochs_filepath(epoch_path)

        epochs = load_epochs(epoch_path)
        epochs_sel = select_epochs(epochs, selection_params)
        cond_1, cond_2, labels = split_epochs_median(epochs_sel, contrast=contrast)

        evoked_1 = cond_1.average()
        evoked_2 = cond_2.average()
        evoked_1.comment = f"{labels['cond_1']}"
        evoked_2.comment = f"{labels['cond_2']}"

        evokeds_cond1.append(evoked_1)
        evokeds_cond2.append(evoked_2)

        diff = _difference_evoked(evoked_1, evoked_2)
        diffs.append(diff)

        if cond_1.metadata is None or cond_2.metadata is None:
            raise RuntimeError("Selection/splitting should preserve metadata, but it is None.")

        md1 = cond_1.metadata.copy()
        md1["subject"] = file_info.subject
        md1["run"] = file_info.run
        md1["condition"] = labels["cond_1"]

        md2 = cond_2.metadata.copy()
        md2["subject"] = file_info.subject
        md2["run"] = file_info.run
        md2["condition"] = labels["cond_2"]

        metadata_rows.append(pd.concat([md1, md2], ignore_index=True))

        n_trials_rows.append(
            {"subject": file_info.subject, "run": file_info.run, "condition": labels["cond_1"], "n_epochs": len(cond_1)}
        )
        n_trials_rows.append(
            {"subject": file_info.subject, "run": file_info.run, "condition": labels["cond_2"], "n_epochs": len(cond_2)}
        )

    if len(diffs) == 0:
        raise ValueError("No epochs files provided (epoch_paths is empty).")

    grand_diff = mne.grand_average(diffs)
    grand_diff.comment = "grand_diff_cond2_minus_cond1"

    metadata = pd.concat(metadata_rows, ignore_index=True)
    n_trials = pd.DataFrame(n_trials_rows)

    return EvokedDatasetResult(
        evokeds_cond1=evokeds_cond1,
        evokeds_cond2=evokeds_cond2,
        difference=grand_diff,
        metadata=metadata,
        n_trials=n_trials,
    )
