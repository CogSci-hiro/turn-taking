"""
Temporal generalization decoding orchestration.

The decoding pipeline is ERP-based and produces a temporal generalization (TG)
matrix per subject, using MNE-Python's ``GeneralizingEstimator`` around a linear
classifier. Group outputs are stacked into a single 4D array:

``scores.shape == (n_subjects, n_splits, n_times, n_times)``.

This module is intentionally model-centric; dataset construction lives in
``turntaking.analysis.decoding.dataset`` and persistence lives in
``turntaking.analysis.decoding.io``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, Tuple

import numpy as np
from mne.decoding import GeneralizingEstimator, cross_val_multiscore
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from turntaking.analysis.decoding.dataset import DecodingDatasetParams, LoadSubjectEpochsFn, make_decoding_data


@dataclass(frozen=True)
class DecodingRunParams:
    """Model fitting parameters for decoding."""

    n_splits: int
    seed: int
    n_jobs: int


def decode_subject_temporal_generalization(
    X: np.ndarray,
    y: np.ndarray,
    params: DecodingRunParams,
) -> np.ndarray:
    """
    Decode a single subject with temporal generalization.

    Parameters
    ----------
    X
        Feature tensor ``(n_trials, n_channels, n_times)``.
    y
        Class labels ``(n_trials,)`` with two classes.
    params
        Cross-validation and parallelism controls.

    Returns
    -------
    scores : np.ndarray
        TG scores ``(n_splits, n_times, n_times)`` (train time x test time).
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X as (trials, channels, times), got {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"Expected y as (trials,), got {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Trial mismatch: X has {X.shape[0]}, y has {y.shape[0]}.")

    classifier = make_pipeline(StandardScaler(), LinearSVC())

    decoder = GeneralizingEstimator(
        classifier,
        n_jobs=params.n_jobs,
        scoring="roc_auc",
        verbose="ERROR",
    )

    cv = StratifiedKFold(
        n_splits=params.n_splits,
        shuffle=True,
        random_state=params.seed,
    )

    scores = cross_val_multiscore(
        decoder,
        X,
        y,
        cv=cv,
        n_jobs=params.n_jobs,
        verbose="ERROR",
    )
    return scores


def run_group_decoding(
    *,
    subjects: Sequence[str],
    epoch_dir: Path,
    dataset_params: DecodingDatasetParams,
    run_params: DecodingRunParams,
    load_subject_epochs_fn: LoadSubjectEpochsFn,
    load_cached_features_fn: Callable[[str], Tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    save_cached_features_fn: Callable[[str, np.ndarray, np.ndarray, np.ndarray], None] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run temporal generalization decoding for multiple subjects and stack results.

    Parameters
    ----------
    subjects
        Iterable of subject IDs such as ``["sub-004", "sub-005"]``.
    epoch_dir
        Directory containing epoch FIF files.
    dataset_params
        Selection/splitting/resampling parameters.
    run_params
        Model + CV parameters.
    load_subject_epochs_fn
        Callback that loads epochs for a subject (allows swapping discovery logic).
    load_cached_features_fn, save_cached_features_fn
        Optional callbacks for caching decoded features on disk (e.g. HDF5).

    Returns
    -------
    scores
        Stacked TG scores ``(n_subjects, n_splits, n_times, n_times)``.
    times_s
        Shared time vector in seconds, shape ``(n_times,)``.
    """
    subjects_sorted = sorted(subjects)

    scores_list: list[np.ndarray] = []
    times_ref: np.ndarray | None = None

    for subject in subjects_sorted:
        if load_cached_features_fn is not None:
            X, y, times_s = load_cached_features_fn(subject)
        else:
            X, y, times_s = make_decoding_data(
                subject=subject,
                epoch_dir=epoch_dir,
                params=dataset_params,
                load_subject_epochs_fn=load_subject_epochs_fn,
            )
            if save_cached_features_fn is not None:
                save_cached_features_fn(subject, X, y, times_s)

        scores_sub = decode_subject_temporal_generalization(X, y, run_params)

        if times_ref is None:
            times_ref = times_s
        else:
            if times_s.shape != times_ref.shape or not np.allclose(times_s, times_ref, atol=1e-12):
                raise ValueError(
                    f"Time axis mismatch: subject={subject}, times shape={times_s.shape}, "
                    f"expected={times_ref.shape}"
                )

        if scores_sub.shape[0] != run_params.n_splits:
            raise ValueError(
                f"Unexpected n_splits for subject={subject}: got {scores_sub.shape[0]}, "
                f"expected {run_params.n_splits}"
            )

        scores_list.append(scores_sub)

    if times_ref is None:
        raise RuntimeError("No subjects provided; cannot run decoding.")

    scores = np.stack(scores_list, axis=0)
    return scores, times_ref
