"""
Decoding dataset construction (ERP features).

This module turns MNE epochs into the ``(X, y, times)`` tensors expected by the
temporal generalization decoder.

Workflow per subject
--------------------
1. Load epochs for that subject.
2. Apply metadata-based selection thresholds.
3. Resample to a lower sfreq for compute/memory efficiency.
4. Median-split into two classes (duration or latency).
5. Return trial tensors + binary labels.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Tuple

import mne
import numpy as np

from turntaking.analysis.selection import SelectionParams, select_epochs, split_epochs_median

Contrast = Literal["latency", "duration"]


@dataclass(frozen=True)
class DecodingDatasetParams:
    """
    Parameters controlling decoding dataset construction.

    Attributes
    ----------
    contrast
        Which median-split rule to use (duration or latency).
    selection
        Inclusion thresholds applied before splitting.
    sfreq_hz
        Resampling frequency (Hz) for decoding features.
    """

    contrast: Contrast
    selection: SelectionParams
    sfreq_hz: float


LoadSubjectEpochsFn = Callable[[str, Path], mne.BaseEpochs]


def make_subject_split(
    *,
    subject: str,
    epoch_dir: Path,
    params: DecodingDatasetParams,
    load_subject_epochs_fn: LoadSubjectEpochsFn,
) -> Tuple[mne.BaseEpochs, mne.BaseEpochs]:
    """
    Load epochs for a subject, apply selection, resample, and split into 2 classes.

    Returns
    -------
    cond_1, cond_2
        Two Epochs objects with equalized epoch counts.
    """
    epochs = load_subject_epochs_fn(subject, epoch_dir)

    # Apply inclusion criteria
    epochs = select_epochs(epochs, params.selection)

    # Resample before splitting (keeps time axis consistent and reduces compute).
    # (If your epochs are already at sfreq_hz, MNE will be a no-op.)
    epochs = epochs.copy().resample(sfreq=params.sfreq_hz, npad="auto")

    cond_1, cond_2, _labels = split_epochs_median(epochs, params.contrast)

    return cond_1, cond_2


def make_decoding_data(
    *,
    subject: str,
    epoch_dir: Path,
    params: DecodingDatasetParams,
    load_subject_epochs_fn: LoadSubjectEpochsFn,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make X, y, times for temporal generalization decoding.

    Returns
    -------
    X
        (n_trials, n_channels, n_times)
    y
        (n_trials,), values in {0,1}
    times_s
        (n_times,)
    """
    cond_1, cond_2 = make_subject_split(
        subject=subject,
        epoch_dir=epoch_dir,
        params=params,
        load_subject_epochs_fn=load_subject_epochs_fn,
    )

    # Data
    X_1 = cond_1.get_data()
    X_2 = cond_2.get_data()
    X = np.vstack((X_1, X_2))

    y = np.hstack(
        (
            np.zeros((X_1.shape[0],), dtype=np.int64),
            np.ones((X_2.shape[0],), dtype=np.int64),
        )
    )

    times_1 = np.asarray(cond_1.times, dtype=np.float64)
    times_2 = np.asarray(cond_2.times, dtype=np.float64)
    if times_1.shape != times_2.shape or not np.allclose(times_1, times_2, atol=1e-12):
        raise ValueError(
            f"Time axis mismatch after split for subject={subject}: "
            f"{times_1.shape} vs {times_2.shape}"
        )

    return X, y, times_1
