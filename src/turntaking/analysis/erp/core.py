from __future__ import annotations

from typing import Any

import mne
import numpy as np
import pandas as pd

from turntaking.analysis.datasets.evoked_dataset import EvokedDatasetRaw, EvokedDatasetResult
from turntaking.analysis.selection import Contrast


def compute_erp_average(
    data: np.ndarray,
    trial_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute ERP average for a set of trials.

    Parameters
    ----------
    data
        3D array [n_trials, n_channels, n_times]
    trial_mask
        boolean array of shape [n_trials] indicating which trials to include

    Returns
    -------
    erp : ndarray [n_channels, n_times]
    """
    if data.ndim != 3:
        raise ValueError(f"`data` must be 3D [n_trials, n_channels, n_times], got shape={data.shape}.")

    mask = np.asarray(trial_mask)
    if mask.ndim != 1:
        raise ValueError(f"`trial_mask` must be 1D [n_trials], got shape={mask.shape}.")
    if mask.shape[0] != data.shape[0]:
        raise ValueError(
            f"`trial_mask` length ({mask.shape[0]}) must match n_trials ({data.shape[0]})."
        )
    if mask.dtype != np.bool_:
        mask = mask.astype(bool, copy=False)
    if not np.any(mask):
        raise ValueError("`trial_mask` selects zero trials; cannot compute ERP average.")
    return data[mask].mean(axis=0)


def compute_contrast(
    erp_condition1: np.ndarray,
    erp_condition2: np.ndarray,
) -> np.ndarray:
    """Compute simple contrast: condition1 - condition2."""
    if erp_condition1.ndim != 2 or erp_condition2.ndim != 2:
        raise ValueError(
            "Both ERP inputs must be 2D arrays [n_channels, n_times]; "
            f"got {erp_condition1.shape} and {erp_condition2.shape}."
        )
    if erp_condition1.shape != erp_condition2.shape:
        raise ValueError(
            f"ERP shapes must match for contrast, got {erp_condition1.shape} vs {erp_condition2.shape}."
        )
    return erp_condition1 - erp_condition2


def apply_baseline(
    erp: np.ndarray,
    times: np.ndarray,
    baseline: tuple[float, float] | None,
) -> np.ndarray:
    """Apply baseline correction to ERP."""
    if erp.ndim != 2:
        raise ValueError(f"`erp` must be 2D [n_channels, n_times], got shape={erp.shape}.")

    t = np.asarray(times)
    if t.ndim != 1:
        raise ValueError(f"`times` must be 1D [n_times], got shape={t.shape}.")
    if t.shape[0] != erp.shape[1]:
        raise ValueError(
            f"`times` length ({t.shape[0]}) must match ERP n_times ({erp.shape[1]})."
        )
    if baseline is None:
        return erp.copy()

    tmin, tmax = float(baseline[0]), float(baseline[1])
    if tmin > tmax:
        raise ValueError(f"Invalid baseline window: start ({tmin}) is greater than end ({tmax}).")

    baseline_mask = (t >= tmin) & (t <= tmax)
    if not np.any(baseline_mask):
        raise ValueError(
            "Baseline window does not overlap `times`: "
            f"baseline=({tmin}, {tmax}), times=[{float(t[0])}, {float(t[-1])}]."
        )
    baseline_mean = erp[:, baseline_mask].mean(axis=1, keepdims=True)
    return erp - baseline_mean


def summarize_erp(
    erp: np.ndarray,
    times: np.ndarray,
    summary_window: tuple[float, float],
) -> dict[str, float]:
    """
    Compute selected summary metrics:
    - mean amplitude in window
    - peak latency in window
    - peak amplitude in window
    """
    if erp.ndim != 2:
        raise ValueError(f"`erp` must be 2D [n_channels, n_times], got shape={erp.shape}.")

    t = np.asarray(times)
    if t.ndim != 1:
        raise ValueError(f"`times` must be 1D [n_times], got shape={t.shape}.")
    if t.shape[0] != erp.shape[1]:
        raise ValueError(
            f"`times` length ({t.shape[0]}) must match ERP n_times ({erp.shape[1]})."
        )

    win_tmin, win_tmax = float(summary_window[0]), float(summary_window[1])
    if win_tmin > win_tmax:
        raise ValueError(
            f"Invalid summary window: start ({win_tmin}) is greater than end ({win_tmax})."
        )

    mask = (t >= win_tmin) & (t <= win_tmax)
    if not np.any(mask):
        raise ValueError(
            "Summary window does not overlap `times`: "
            f"summary_window=({win_tmin}, {win_tmax}), times=[{float(t[0])}, {float(t[-1])}]."
        )

    window_erp = erp[:, mask]
    window_times = t[mask]
    channel_mean_waveform = window_erp.mean(axis=0)
    peak_idx = int(np.argmax(channel_mean_waveform))
    return {
        "mean_amplitude": float(window_erp.mean()),
        "peak_latency": float(window_times[peak_idx]),
        "peak_amplitude": float(channel_mean_waveform[peak_idx]),
    }


def _build_evoked(data: np.ndarray, info: mne.Info, times: np.ndarray, *, comment: str) -> mne.Evoked:
    return mne.EvokedArray(data, info=info.copy(), tmin=float(times[0]), comment=comment)


def _build_offsets(
    subject: str,
    labels: dict[str, str],
    cond1_metadata: pd.DataFrame,
    cond2_metadata: pd.DataFrame,
) -> pd.DataFrame:
    md1 = cond1_metadata.copy()
    md2 = cond2_metadata.copy()
    md1["condition"] = labels["cond_1"]
    md2["condition"] = labels["cond_2"]
    offsets = pd.concat([md1, md2], ignore_index=True)
    offsets["subject"] = subject
    return offsets


def _stack_evoked_data(
    evokeds_cond_1: list[mne.Evoked],
    evokeds_cond_2: list[mne.Evoked],
    evokeds_difference: list[mne.Evoked],
) -> np.ndarray:
    return np.stack(
        [
            np.stack([ev.data for ev in evokeds_cond_1], axis=0),
            np.stack([ev.data for ev in evokeds_cond_2], axis=0),
            np.stack([ev.data for ev in evokeds_difference], axis=0),
        ],
        axis=1,
    )


def _build_dataset_metadata(
    *,
    contrast: Contrast,
    labels: dict[str, str],
    subject_ids: list[str],
    times: np.ndarray,
    ch_names: list[str],
    evoked_data: np.ndarray,
) -> dict[str, Any]:
    return {
        "kind": "erp",
        "contrast": str(contrast),
        "cond_1": labels["cond_1"],
        "cond_2": labels["cond_2"],
        "subjects": np.array(subject_ids, dtype=object),
        "n_subjects": int(len(subject_ids)),
        "times": times,
        "ch_names": np.array(ch_names, dtype=object),
        "data_shape": np.array(evoked_data.shape, dtype=int),
        "difference_definition": f"{labels['cond_1']}-{labels['cond_2']}",
    }


def _compute_subject_evokeds(
    raw_dataset: EvokedDatasetRaw,
    index: int,
) -> tuple[mne.Evoked, mne.Evoked, mne.Evoked]:
    cond1_data = raw_dataset.cond1_epochs[index]
    cond2_data = raw_dataset.cond2_epochs[index]
    mean1 = compute_erp_average(cond1_data, np.ones(cond1_data.shape[0], dtype=bool))
    mean2 = compute_erp_average(cond2_data, np.ones(cond2_data.shape[0], dtype=bool))
    diff = compute_contrast(mean1, mean2)
    ev1 = _build_evoked(mean1, raw_dataset.infos[index], raw_dataset.times, comment=raw_dataset.labels["cond_1"])
    ev2 = _build_evoked(mean2, raw_dataset.infos[index], raw_dataset.times, comment=raw_dataset.labels["cond_2"])
    evd = _build_evoked(
        diff,
        raw_dataset.infos[index],
        raw_dataset.times,
        comment=f"{raw_dataset.labels['cond_1']}-{raw_dataset.labels['cond_2']}",
    )
    return ev1, ev2, evd


def _subject_trial_counts(raw_dataset: EvokedDatasetRaw, index: int, subject: str) -> dict[str, Any]:
    return {
        "subject": subject,
        raw_dataset.labels["cond_1"]: int(raw_dataset.cond1_epochs[index].shape[0]),
        raw_dataset.labels["cond_2"]: int(raw_dataset.cond2_epochs[index].shape[0]),
    }


def compute_evoked_dataset_result(
    raw_dataset: EvokedDatasetRaw,
    *,
    contrast: Contrast,
) -> EvokedDatasetResult:
    """Compute ERP evoked outputs from raw split/equalized epoch arrays."""
    evokeds_cond_1: list[mne.Evoked] = []
    evokeds_cond_2: list[mne.Evoked] = []
    evokeds_difference: list[mne.Evoked] = []
    offsets_rows: list[pd.DataFrame] = []
    n_trials_rows: list[dict[str, Any]] = []

    for idx, subject in enumerate(raw_dataset.subject_ids):
        ev1, ev2, evd = _compute_subject_evokeds(raw_dataset, idx)
        evokeds_cond_1.append(ev1)
        evokeds_cond_2.append(ev2)
        evokeds_difference.append(evd)
        offsets_rows.append(
            _build_offsets(
                subject,
                raw_dataset.labels,
                raw_dataset.cond1_metadata[idx],
                raw_dataset.cond2_metadata[idx],
            )
        )
        n_trials_rows.append(_subject_trial_counts(raw_dataset, idx, subject))

    evoked_data = _stack_evoked_data(evokeds_cond_1, evokeds_cond_2, evokeds_difference)
    metadata = _build_dataset_metadata(
        contrast=contrast,
        labels=raw_dataset.labels,
        subject_ids=raw_dataset.subject_ids,
        times=raw_dataset.times,
        ch_names=raw_dataset.ch_names,
        evoked_data=evoked_data,
    )
    offsets = pd.concat(offsets_rows, ignore_index=True) if offsets_rows else pd.DataFrame()
    n_trials = pd.DataFrame(n_trials_rows)
    return EvokedDatasetResult(
        evokeds_cond_1=evokeds_cond_1,
        evokeds_cond_2=evokeds_cond_2,
        evokeds_difference=evokeds_difference,
        evoked_data=evoked_data,
        n_trials=n_trials,
        offsets=offsets,
        results=metadata,
    )
