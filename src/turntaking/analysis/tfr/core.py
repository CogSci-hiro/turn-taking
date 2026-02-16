
from typing import Any

import mne
import numpy as np
import pandas as pd

from turntaking.analysis.datasets.evoked_dataset import EvokedDatasetRaw, EvokedDatasetResult
from turntaking.analysis.erp.core import compute_contrast
from turntaking.analysis.selection import Contrast

_BAND_LIMITS_HZ: dict[str, tuple[float, float]] = {
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
}


def _band_limits_hz(band: str) -> tuple[float, float]:
    if band not in _BAND_LIMITS_HZ:
        raise ValueError(
            f"Unknown band={band!r}. Known: {sorted(_BAND_LIMITS_HZ.keys())}. "
            "Add it to _BAND_LIMITS_HZ or wire config-based bands."
        )
    return _BAND_LIMITS_HZ[band]


def _compute_induced_envelope_epochs(
    data: np.ndarray,
    *,
    info: mne.Info,
    times: np.ndarray,
    band: str,
) -> np.ndarray:
    l_freq, h_freq = _band_limits_hz(band)
    epochs = mne.EpochsArray(data.copy(), info.copy(), tmin=float(times[0]), verbose=False)
    epochs.load_data()
    epochs.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose="ERROR",
    )
    epochs.apply_hilbert(envelope=True)
    return epochs.get_data(copy=True)


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
    band: str,
) -> dict[str, Any]:
    return {
        "kind": "tfr",
        "contrast": str(contrast),
        "cond_1": labels["cond_1"],
        "cond_2": labels["cond_2"],
        "subjects": np.array(subject_ids, dtype=object),
        "n_subjects": int(len(subject_ids)),
        "times": times,
        "ch_names": np.array(ch_names, dtype=object),
        "data_shape": np.array(evoked_data.shape, dtype=int),
        "difference_definition": f"{labels['cond_1']}-{labels['cond_2']}",
        "band": str(band),
    }


def _sfreq_from_times(times: np.ndarray) -> float:
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a 1D array with at least two samples.")
    dt = np.diff(times)
    if not np.allclose(dt, dt[0], atol=1e-12, rtol=0.0):
        raise ValueError("times must be evenly sampled for induced envelope computation.")
    return float(round(1.0 / float(dt[0]), 12))


def _compute_subject_induced_evokeds(
    raw_dataset: EvokedDatasetRaw,
    index: int,
    *,
    band: str,
) -> tuple[mne.Evoked, mne.Evoked, mne.Evoked]:
    cond1_data = raw_dataset.cond1_epochs[index]
    cond2_data = raw_dataset.cond2_epochs[index]
    mean1 = _compute_induced_envelope_epochs(
        cond1_data,
        info=raw_dataset.infos[index],
        times=raw_dataset.times,
        band=band,
    ).mean(axis=0)
    mean2 = _compute_induced_envelope_epochs(
        cond2_data,
        info=raw_dataset.infos[index],
        times=raw_dataset.times,
        band=band,
    ).mean(axis=0)
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


def compute_induced_dataset_result(
    raw_dataset: EvokedDatasetRaw,
    *,
    band: str,
    contrast: Contrast,
) -> EvokedDatasetResult:
    """Compute induced-envelope evoked outputs from raw split/equalized arrays."""
    _sfreq_from_times(raw_dataset.times)
    evokeds_cond_1: list[mne.Evoked] = []
    evokeds_cond_2: list[mne.Evoked] = []
    evokeds_difference: list[mne.Evoked] = []
    offsets_rows: list[pd.DataFrame] = []
    n_trials_rows: list[dict[str, Any]] = []

    for idx, subject in enumerate(raw_dataset.subject_ids):
        ev1, ev2, evd = _compute_subject_induced_evokeds(raw_dataset, idx, band=band)
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
        band=band,
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
