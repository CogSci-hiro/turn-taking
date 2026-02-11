
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Tuple

import numpy as np


ContrastName = Literal["latency", "duration"]


@dataclass(frozen=True)
class DecodingDatasetParams:
    """
    Parameters for making trial-level decoding data.

    Attributes
    ----------
    contrast
        Contrast to decode ("latency" or "duration").
    min_latency_s
        Minimum response latency allowed.
    max_latency_s
        Maximum response latency allowed.
    min_response_duration_s
        Minimum response duration allowed.
    sfreq_hz
        Target sampling frequency (resampling happens in split function).
    """

    contrast: ContrastName
    min_latency_s: float
    max_latency_s: float
    min_response_duration_s: float
    sfreq_hz: float


MakeSubjectSplitFn = Callable[
    [str, Path, ContrastName, float, float, float, float],
    Tuple["mne.Epochs", "mne.Epochs"],
]


# =============================================================================
#                     ########################################
#                     #       DECODING DATASET (ERP)         #
#                     ########################################
# =============================================================================


def make_decoding_data(
    *,
    subject: str,
    epoch_dir: Path,
    params: DecodingDatasetParams,
    make_subject_split_fn: MakeSubjectSplitFn,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make (X, y, times) for temporal generalization decoding.

    This function assumes `make_subject_split_fn` performs:
    - loading epochs (potentially from multiple runs)
    - trial filtering (latency bounds, min duration, etc.)
    - median split into two classes
    - equalization of class trial counts (recommended)
    - resampling to `params.sfreq_hz`

    Parameters
    ----------
    subject
        Subject ID, e.g. "sub-001".
    epoch_dir
        Directory containing epochs.
    params
        Dataset creation parameters.
    make_subject_split_fn
        Function that returns (cond_1_epochs, cond_2_epochs).

    Returns
    -------
    X
        Neural data, shape (n_trials, n_channels, n_times).
    y
        Labels, shape (n_trials,), values in {0, 1}.
    times_s
        Time vector in seconds, shape (n_times,).

    Usage example
    -------------
        params = DecodingDatasetParams(
            contrast="duration",
            min_latency_s=-1.0,
            max_latency_s=1.0,
            min_response_duration_s=0.01,
            sfreq_hz=64.0,
        )

        X, y, times_s = make_decoding_data(
            subject="sub-001",
            epoch_dir=Path("derivatives/epochs"),
            params=params,
            make_subject_split_fn=make_subject_split,
        )
    """
    cond_1, cond_2 = make_subject_split_fn(
        subject,
        epoch_dir,
        params.contrast,
        params.min_latency_s,
        params.max_latency_s,
        params.min_response_duration_s,
        params.sfreq_hz,
    )

    # Neural data
    X_1 = cond_1.get_data()  # (n_trials_1, n_channels, n_times)
    X_2 = cond_2.get_data()  # (n_trials_2, n_channels, n_times)
    X = np.vstack((X_1, X_2))

    # Labels
    y = np.hstack(
        (
            np.zeros((X_1.shape[0],), dtype=np.int64),
            np.ones((X_2.shape[0],), dtype=np.int64),
        )
    )

    # Time axis consistency
    times_1 = np.asarray(cond_1.times, dtype=np.float64)
    times_2 = np.asarray(cond_2.times, dtype=np.float64)

    if times_1.shape != times_2.shape or not np.allclose(times_1, times_2, atol=1e-12):
        raise ValueError(
            f"Time axis mismatch after split for subject={subject}: "
            f"times_1 shape={times_1.shape}, times_2 shape={times_2.shape}."
        )

    return X, y, times_1
