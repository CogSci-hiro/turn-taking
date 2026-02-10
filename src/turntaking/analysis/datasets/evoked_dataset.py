
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import mne
import numpy as np
import pandas as pd

from turntaking.analysis.io.epochs import load_epochs, parse_epochs_filepath
from turntaking.analysis.selection import Contrast, SelectionParams, select_epochs, split_epochs_median

Kind = Literal["erp", "tfr"]


@dataclass(frozen=True)
class EvokedDatasetResult:
    """
    Outputs of per-contrast ERP dataset construction (per subject).

    Notes
    -----
    This object is intentionally *write-ready*: it contains everything required
    to create the Snakemake-tracked artifacts for a single contrast.

    DataFrame formats
    -----------------
    n_trials
        | subject  | n_cond_1 | n_cond_2 |
        |----------|----------|----------|
        | sub-004  | 120      | 118      |

    offsets
        Legacy table (copied from old script). Must include a 'condition' column:
        | timestamp | self_duration | ... | condition |
        |-----------|---------------|-----|-----------|
        | ...       | ...           | ... | long      |

    Usage example
    -------------
        result = build_evoked_dataset(...)
        # result is then passed to write_erp_outputs(...)
    """

    # Per-subject evokeds (these become *_ave.fif files)
    evokeds_cond_1: list[mne.Evoked]
    evokeds_cond_2: list[mne.Evoked]
    evokeds_difference: list[mne.Evoked]

    # NPY payload (saved verbatim; you define shape/semantics upstream)
    evoked_data: np.ndarray

    # Tables
    n_trials: pd.DataFrame
    offsets: pd.DataFrame

    # HDF5 payload
    results: Mapping[str, Any]


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
    evokeds_diff: list[mne.Evoked] = []

    offsets_rows: list[pd.DataFrame] = []
    n_trials_rows: list[dict[str, Any]] = []

    labels: dict[str, str] | None = None

    for path in epoch_paths:
        info = parse_epochs_filepath(path)
        epochs = load_epochs(path)

        epochs_sel = select_epochs(epochs, selection_params)
        cond1, cond2, split_labels = split_epochs_median(epochs_sel, contrast=contrast)

        # Keep the last labels (they should be identical across files for a given contrast)
        labels = split_labels

        ev1 = cond1.average()
        ev2 = cond2.average()

        # Per-subject difference (cond_2 - cond_1), matching your earlier convention
        evd = ev2.copy()
        evd.data = ev2.data - ev1.data
        evd.comment = f"{split_labels['cond_2']}-{split_labels['cond_1']}"

        evokeds_1.append(ev1)
        evokeds_2.append(ev2)
        evokeds_diff.append(evd)

        # Legacy offsets.csv structure = metadata for both conditions with condition labels
        md1 = cond1.metadata.copy()
        md2 = cond2.metadata.copy()
        md1["condition"] = split_labels["cond_1"]
        md2["condition"] = split_labels["cond_2"]

        md = pd.concat([md1, md2], ignore_index=True)
        md["subject"] = info.subject
        md["run"] = info.run
        offsets_rows.append(md)

        # n_trials.csv
        n_trials_rows.append(
            {
                "subject": info.subject,
                "run": info.run,
                split_labels["cond_1"]: int(len(cond1)),
                split_labels["cond_2"]: int(len(cond2)),
            }
        )

    if len(evokeds_1) == 0:
        raise ValueError("No epoch files provided / no evokeds computed.")
    if labels is None:
        raise RuntimeError("Internal error: labels were never set.")

    # Evoked-data NPY: store per-subject condition averages + difference
    # Shape: (n_subjects, 3, n_channels, n_times) with order [cond_1, cond_2, diff]
    evoked_data = np.stack(
        [
            np.stack([ev.data for ev in evokeds_1], axis=0),
            np.stack([ev.data for ev in evokeds_2], axis=0),
            np.stack([ev.data for ev in evokeds_diff], axis=0),
        ],
        axis=1,
    )

    offsets = pd.concat(offsets_rows, ignore_index=True) if offsets_rows else pd.DataFrame()
    n_trials = pd.DataFrame(n_trials_rows)

    # results.hdf5 payload (keep it simple for now; expand later)
    results: dict[str, Any] = {
        "contrast": str(contrast),
        "cond_1": labels["cond_1"],
        "cond_2": labels["cond_2"],
        "times": evokeds_1[0].times,
        "ch_names": np.array(evokeds_1[0].ch_names, dtype=object),
        "evoked_data_shape": np.array(evoked_data.shape, dtype=int),
    }

    return EvokedDatasetResult(
        evokeds_cond_1=evokeds_1,
        evokeds_cond_2=evokeds_2,
        evokeds_difference=evokeds_diff,
        evoked_data=evoked_data,
        n_trials=n_trials,
        offsets=offsets,
        results=results,
    )
