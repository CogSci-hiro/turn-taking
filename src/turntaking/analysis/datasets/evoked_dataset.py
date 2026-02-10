
from collections import defaultdict
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


def _stable_sort_key(path: Path) -> tuple:
    info = parse_epochs_filepath(path)

    # We only rely on attributes that might exist; fall back to path name.
    # run is often int-like; normalize to string for safety.
    run = getattr(info, "run", None)
    run_key = str(run) if run is not None else ""

    # Optional fields if they exist (won't crash if they don't)
    task = getattr(info, "task", None)
    task_key = str(task) if task is not None else ""

    return (task_key, run_key, path.name)


def build_evoked_dataset(
    epoch_paths: list[Path],
    *,
    kind: Kind,
    contrast: Contrast,
    selection_params: SelectionParams,
) -> EvokedDatasetResult:
    """
    Build ERP evoked dataset using OLD (subject-level) logic inside the NEW API.

    Old behavior restored:
    - concatenate all runs per subject
    - select on concatenated epochs
    - median split once per subject
    - equalize epoch counts between conditions
    - difference = cond_1 - cond_2 (old combine_evoked weights [1, -1])
    """
    if kind != "erp":
        raise NotImplementedError("Only kind='erp' is implemented in this vertical slice.")
    if len(epoch_paths) == 0:
        raise ValueError("No epoch files provided.")

    # Group epoch files by subject
    paths_by_subject: dict[str, list[Path]] = defaultdict(list)
    for path in epoch_paths:
        info = parse_epochs_filepath(path)
        paths_by_subject[info.subject].append(path)

    # Deterministic subject order (and deterministic run order within subject)
    subjects = sorted(paths_by_subject.keys())
    for subject in subjects:
        paths_by_subject[subject] = sorted(paths_by_subject[subject], key=_stable_sort_key)

    evokeds_1: list[mne.Evoked] = []
    evokeds_2: list[mne.Evoked] = []
    evokeds_diff: list[mne.Evoked] = []

    offsets_rows: list[pd.DataFrame] = []
    n_trials_rows: list[dict[str, Any]] = []

    labels: dict[str, str] | None = None
    reference_ch_names: list[str] | None = None
    reference_times: np.ndarray | None = None

    for subject in subjects:
        paths_by_subject[subject] = sorted(paths_by_subject[subject], key=_stable_sort_key)

        # Load + concatenate runs for this subject
        epochs_list: list[mne.BaseEpochs] = [load_epochs(p) for p in paths_by_subject[subject]]
        if len(epochs_list) == 1:
            epochs = epochs_list[0]
        else:
            # Keep everything consistent; if there are mismatched channel sets, MNE will complain.
            epochs = mne.concatenate_epochs(epochs_list)

        # Select epochs (constraints) on concatenated subject epochs
        epochs_sel = select_epochs(epochs, selection_params)

        # Split once per subject
        cond1, cond2, split_labels = split_epochs_median(epochs_sel, contrast=contrast)
        labels = split_labels  # they should be identical across subjects for a given contrast

        # Equalize counts like old logic
        mne.epochs.equalize_epoch_counts([cond1, cond2])

        ev1 = cond1.average()
        ev2 = cond2.average()

        # OLD convention: difference = cond_1 - cond_2
        evd = ev1.copy()
        evd.data = ev1.data - ev2.data
        evd.comment = f"{split_labels['cond_1']}-{split_labels['cond_2']}"

        # Cross-subject invariants (fail fast if violated)
        if reference_ch_names is None:
            reference_ch_names = list(ev1.ch_names)
        else:
            if list(ev1.ch_names) != reference_ch_names:
                raise ValueError(
                    f"Channel order mismatch for subject={subject}. "
                    "This would silently break group stacking; fix upstream."
                )

        if reference_times is None:
            reference_times = ev1.times.copy()
        else:
            if ev1.times.shape != reference_times.shape or not np.allclose(ev1.times, reference_times, atol=0.0, rtol=0.0):
                raise ValueError(
                    f"Time axis mismatch for subject={subject}. "
                    "This would silently break group stacking; fix upstream."
                )

        evokeds_1.append(ev1)
        evokeds_2.append(ev2)
        evokeds_diff.append(evd)

        # offsets.csv (legacy): metadata + condition + subject
        md1 = cond1.metadata.copy()
        md2 = cond2.metadata.copy()
        md1["condition"] = split_labels["cond_1"]
        md2["condition"] = split_labels["cond_2"]
        md = pd.concat([md1, md2], ignore_index=True)
        md["subject"] = subject
        offsets_rows.append(md)

        # n_trials.csv (subject-level)
        n_trials_rows.append(
            {
                "subject": subject,
                split_labels["cond_1"]: int(len(cond1)),
                split_labels["cond_2"]: int(len(cond2)),
            }
        )

    if labels is None:
        raise RuntimeError("Internal error: labels were never set.")
    if len(evokeds_1) == 0:
        raise ValueError("No evokeds computed (maybe selection removed all epochs).")

    # NEW NPY payload shape kept: (n_subjects, 3, n_channels, n_times) order [cond_1, cond_2, diff]
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

    results: dict[str, Any] = {
        "contrast": str(contrast),
        "cond_1": labels["cond_1"],
        "cond_2": labels["cond_2"],
        "subjects": np.array(subjects, dtype=object),
        "n_subjects": int(len(subjects)),
        "times": evokeds_1[0].times,
        "ch_names": np.array(evokeds_1[0].ch_names, dtype=object),
        "evoked_data_shape": np.array(evoked_data.shape, dtype=int),
        "difference_definition": f"{labels['cond_1']}-{labels['cond_2']}",
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
