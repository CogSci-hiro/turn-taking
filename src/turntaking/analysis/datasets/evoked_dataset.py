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


# =============================================================================
# Band definitions
# =============================================================================
_BAND_LIMITS_HZ: dict[str, tuple[float, float]] = {
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
}


def _band_limits_hz(band: str) -> tuple[float, float]:
    """
    Return frequency limits for a predefined frequency band.

    Parameters
    ----------
    band : str
        Name of the frequency band.

    Returns
    -------
    tuple of float
        (l_freq, h_freq) in Hz.

    Raises
    ------
    ValueError
        If the band is not defined in `_BAND_LIMITS_HZ`.

    Notes
    -----
    This function enforces explicit band definitions to avoid silent
    mismatches between analysis configuration and implementation.
    """
    if band not in _BAND_LIMITS_HZ:
        raise ValueError(
            f"Unknown band={band!r}. Known: {sorted(_BAND_LIMITS_HZ.keys())}. "
            "Add it to _BAND_LIMITS_HZ or wire config-based bands."
        )
    return _BAND_LIMITS_HZ[band]


def _compute_induced_envelope_epochs(
    epochs: mne.BaseEpochs,
    *,
    band: str,
) -> np.ndarray:
    """
    Compute band-limited induced envelope per epoch using Hilbert transform.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epoched data.
    band : str
        Frequency band name (must exist in `_BAND_LIMITS_HZ`).

    Returns
    -------
    np.ndarray
        Array of shape (n_epochs, n_channels, n_times)
        containing the Hilbert envelope of the band-passed signal.

    Notes
    -----
    Processing steps:
        1. Band-pass filter (FIR, zero-phase).
        2. Apply analytic Hilbert transform.
        3. Extract amplitude envelope.

    This returns *induced* activity (no phase locking across trials).
    """
    l_freq, h_freq = _band_limits_hz(band)

    epochs = epochs.copy()

    # Bandpass
    epochs.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose="ERROR",
    )

    # Hilbert envelope
    epochs.apply_hilbert(envelope=True)

    return epochs.get_data()  # (E, C, T)


def _induced_evoked_from_epochs(
    epochs: mne.BaseEpochs,
    *,
    band: str,
    comment: str,
) -> mne.Evoked:
    """
    Convert induced envelope data into an ERP-like Evoked object.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epoched data.
    band : str
        Frequency band name.
    comment : str
        Label stored in the Evoked object.

    Returns
    -------
    mne.Evoked
        Evoked object with data shape (n_channels, n_times).

    Notes
    -----
    The envelope is computed per epoch and then averaged:

        mean_env(c, t) = mean_e envelope(e, c, t)

    This preserves channel/time structure while discarding
    phase-locked information.
    """
    env = _compute_induced_envelope_epochs(epochs, band=band)  # (E,C,T)
    mean_env = env.mean(axis=0)  # (C, T)

    evoked = mne.EvokedArray(
        mean_env,
        info=epochs.info.copy(),
        tmin=float(epochs.times[0]),
        comment=comment,
    )
    return evoked


def _stable_sort_key(path: Path) -> tuple:
    """
    Construct a deterministic sort key for epoch file paths.

    Parameters
    ----------
    path : Path
        Path to epoch file.

    Returns
    -------
    tuple
        Sorting key based on (task, run, filename).

    Notes
    -----
    Ensures consistent concatenation order across subjects.
    """
    info = parse_epochs_filepath(path)

    run = getattr(info, "run", None)
    run_key = str(run) if run is not None else ""

    task = getattr(info, "task", None)
    task_key = str(task) if task is not None else ""

    return task_key, run_key, path.name


# =============================================================================
# Dataset result (reused for ERP and induced-TFR)
# =============================================================================
@dataclass(frozen=True)
class EvokedDatasetResult:
    """
    Container for subject-level evoked dataset outputs.

    Attributes
    ----------
    evokeds_cond_1 : list of mne.Evoked
        Per-subject Evoked objects for condition 1.
    evokeds_cond_2 : list of mne.Evoked
        Per-subject Evoked objects for condition 2.
    evokeds_difference : list of mne.Evoked
        Per-subject difference wave (cond_1 - cond_2).
    evoked_data : np.ndarray
        Array of shape (n_subjects, 3, n_channels, n_times),
        ordered as [cond_1, cond_2, diff].
    n_trials : pd.DataFrame
        Per-subject trial counts.
    offsets : pd.DataFrame
        Concatenated metadata table for selected epochs.
    results : Mapping[str, Any]
        Metadata dictionary describing the dataset.
    """

    evokeds_cond_1: list[mne.Evoked]
    evokeds_cond_2: list[mne.Evoked]
    evokeds_difference: list[mne.Evoked]

    # (N,3,C,T) order [cond_1, cond_2, diff]
    evoked_data: np.ndarray
    n_trials: pd.DataFrame
    offsets: pd.DataFrame

    # Metadata payload (written as metadata.hdf5)
    results: Mapping[str, Any]


def build_evoked_dataset(
    epoch_paths: list[Path],
    *,
    kind: Kind,
    contrast: Contrast,
    selection_params: SelectionParams,
    band: str | None = None,
    sfreq: float | None = None,
) -> EvokedDatasetResult:
    """
    Construct subject-level ERP or induced-TFR dataset.

    Parameters
    ----------
    epoch_paths : list of Path
        Paths to epoch files (possibly multiple runs per subject).
    kind : {"erp", "tfr"}
        Type of analysis.
    contrast : Contrast
        Contrast definition used for median split.
    selection_params : SelectionParams
        Epoch selection criteria.
    band : str | None
        Required if kind="tfr". Frequency band name.
    sfreq : float | None
        Optional resampling frequency.

    Returns
    -------
    EvokedDatasetResult
        Structured container with evoked objects and stacked data.

    Notes
    -----
    Processing steps per subject:
        1. Load and concatenate epochs.
        2. Optional resampling.
        3. Apply selection.
        4. Split via median (cond_1 vs cond_2).
        5. Equalize trial counts.
        6. Compute evoked averages.
        7. Compute difference wave (cond_1 - cond_2).

    Cross-subject invariants:
        - Identical channel ordering.
        - Identical time axis.
        - Identical data shapes.

    Data layout:
        evoked_data shape = (N_subjects, 3, C, T)
        order = [cond_1, cond_2, difference]
    """
    if len(epoch_paths) == 0:
        raise ValueError("No epoch files provided.")
    if kind == "tfr" and band is None:
        raise ValueError("kind='tfr' requires band=...")

    paths_by_subject: dict[str, list[Path]] = defaultdict(list)
    for path in epoch_paths:
        info = parse_epochs_filepath(path)
        paths_by_subject[info.subject].append(path)

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
        epochs_list = [load_epochs(p) for p in paths_by_subject[subject]]
        epochs = epochs_list[0] if len(epochs_list) == 1 else mne.concatenate_epochs(epochs_list)

        # Optional resample (useful for induced envelopes, and matches your config pattern)
        if sfreq is not None:
            epochs = epochs.copy().resample(float(sfreq))

        epochs_sel = select_epochs(epochs, selection_params)
        cond1, cond2, split_labels = split_epochs_median(epochs_sel, contrast=contrast)
        labels = split_labels

        mne.epochs.equalize_epoch_counts([cond1, cond2])

        if kind == "erp":
            ev1 = cond1.average()
            ev2 = cond2.average()
        elif kind == "tfr":
            assert band is not None
            ev1 = _induced_evoked_from_epochs(cond1, band=band, comment=split_labels["cond_1"])
            ev2 = _induced_evoked_from_epochs(cond2, band=band, comment=split_labels["cond_2"])
        else:
            raise ValueError(f"Unknown kind={kind!r}")

        evd = ev1.copy()
        evd.data = ev1.data - ev2.data
        evd.comment = f"{split_labels['cond_1']}-{split_labels['cond_2']}"

        # Cross-subject invariants
        if reference_ch_names is None:
            reference_ch_names = list(ev1.ch_names)
        elif list(ev1.ch_names) != reference_ch_names:
            raise ValueError(f"Channel order mismatch for subject={subject}.")

        if reference_times is None:
            reference_times = ev1.times.copy()
        else:
            if ev1.times.shape != reference_times.shape or not np.allclose(ev1.times, reference_times, atol=0.0, rtol=0.0):
                raise ValueError(f"Time axis mismatch for subject={subject}.")

        evokeds_1.append(ev1)
        evokeds_2.append(ev2)
        evokeds_diff.append(evd)

        # Offsets table (kept for parity)
        md1 = cond1.metadata.copy()
        md2 = cond2.metadata.copy()
        md1["condition"] = split_labels["cond_1"]
        md2["condition"] = split_labels["cond_2"]
        md = pd.concat([md1, md2], ignore_index=True)
        md["subject"] = subject
        offsets_rows.append(md)

        n_trials_rows.append(
            {
                "subject": subject,
                split_labels["cond_1"]: int(len(cond1)),
                split_labels["cond_2"]: int(len(cond2)),
            }
        )

    if labels is None or len(evokeds_1) == 0:
        raise ValueError("No evokeds computed (maybe selection removed all epochs).")

    # (N,3,C,T) order [cond_1, cond_2, diff]
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
        "kind": str(kind),
        "contrast": str(contrast),
        "cond_1": labels["cond_1"],
        "cond_2": labels["cond_2"],
        "subjects": np.array(subjects, dtype=object),
        "n_subjects": int(len(subjects)),
        "times": evokeds_1[0].times,
        "ch_names": np.array(evokeds_1[0].ch_names, dtype=object),
        "data_shape": np.array(evoked_data.shape, dtype=int),
        "difference_definition": f"{labels['cond_1']}-{labels['cond_2']}",
    }
    if kind == "tfr":
        results["band"] = str(band)

    return EvokedDatasetResult(
        evokeds_cond_1=evokeds_1,
        evokeds_cond_2=evokeds_2,
        evokeds_difference=evokeds_diff,
        evoked_data=evoked_data,
        n_trials=n_trials,
        offsets=offsets,
        results=results,
    )
