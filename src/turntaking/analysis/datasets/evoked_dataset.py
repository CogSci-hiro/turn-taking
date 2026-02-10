from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import mne
import pandas as pd

from turntaking.analysis.io import load_epochs, parse_epochs_filepath
from turntaking.analysis.selection import Contrast, SelectionParams, select_epochs, split_epochs_median

Kind = Literal["erp", "tfr"]


@dataclass
class EvokedDatasetResult:
    """Outputs of an evoked dataset build."""
    evokeds_cond1: list[mne.Evoked]
    evokeds_cond2: list[mne.Evoked]
    difference: mne.Evoked
    metadata: pd.DataFrame
    n_trials: pd.DataFrame


def build_evoked_dataset(
    epoch_paths: list[Path],
    *,
    kind: Kind,
    contrast: Contrast,
    selection_params: SelectionParams,
) -> EvokedDatasetResult:
    """Build group-level evoked dataset (ERP-only for now)."""
    if kind != "erp":
        raise NotImplementedError("Only kind='erp' is implemented in this vertical slice.")

    evokeds_1: list[mne.Evoked] = []
    evokeds_2: list[mne.Evoked] = []
    md_rows: list[pd.DataFrame] = []
    n_trials_rows: list[dict] = []

    for path in epoch_paths:
        info = parse_epochs_filepath(path)
        epochs = load_epochs(path)

        epochs_sel = select_epochs(epochs, selection_params)
        cond1, cond2, labels = split_epochs_median(epochs_sel, contrast=contrast)

        ev1 = cond1.average()
        ev2 = cond2.average()
        evokeds_1.append(ev1)
        evokeds_2.append(ev2)

        md1 = cond1.metadata.copy()
        md2 = cond2.metadata.copy()
        md1["condition"] = labels["cond_1"]
        md2["condition"] = labels["cond_2"]
        md = pd.concat([md1, md2], ignore_index=True)
        md["subject"] = info.subject
        md["run"] = info.run
        md_rows.append(md)

        n_trials_rows.append(
            {
                "subject": info.subject,
                "run": info.run,
                labels["cond_1"]: len(cond1),
                labels["cond_2"]: len(cond2),
            }
        )

    if len(evokeds_1) == 0:
        raise ValueError("No epoch files provided / no evokeds computed.")

    grand_1 = mne.grand_average(evokeds_1)
    grand_2 = mne.grand_average(evokeds_2)

    diff = grand_2.copy()
    diff.data = grand_2.data - grand_1.data
    diff.comment = f"{labels['cond_2']}-{labels['cond_1']}"

    metadata = pd.concat(md_rows, ignore_index=True) if md_rows else pd.DataFrame()
    n_trials = pd.DataFrame(n_trials_rows)

    return EvokedDatasetResult(
        evokeds_cond1=evokeds_1,
        evokeds_cond2=evokeds_2,
        difference=diff,
        metadata=metadata,
        n_trials=n_trials,
    )
