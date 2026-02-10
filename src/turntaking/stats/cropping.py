import numpy as np


def crop_time_margins_samples(
    X: np.ndarray,
    *,
    sfreq: float,
    left_margin: float,
    right_margin: float,
) -> tuple[np.ndarray, int, int]:
    """
    Crop a time axis by margins expressed in seconds, using sample indices.

    Parameters
    ----------
    X
        Array with shape (n_observations, n_times, n_space).
        (Space can be channels, channels×freq, etc.)
    sfreq
        Sampling frequency (Hz).
    left_margin, right_margin
        Seconds to remove from the left and right edges.

    Returns
    -------
    X_crop
        Cropped array.
    start_idx, end_idx
        Indices used for cropping in the original time axis.
        end_idx follows Python slicing conventions (exclusive), can be negative or None-equivalent.
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3D (N,T,S). Got {X.shape}")
    if left_margin < 0 or right_margin < 0:
        raise ValueError("left_margin and right_margin must be >= 0.")

    n_times = X.shape[1]
    start_idx = int(round(left_margin * sfreq))
    right_samp = int(round(right_margin * sfreq))

    if start_idx >= n_times:
        raise ValueError(
            f"left_margin={left_margin}s too large for n_times={n_times} at sfreq={sfreq}."
        )

    if right_samp == 0:
        end_idx = n_times
    else:
        end_idx = n_times - right_samp
        if end_idx <= start_idx:
            raise ValueError(
                f"Cropping removes all data: start_idx={start_idx}, end_idx={end_idx}, n_times={n_times}."
            )

    X_crop = X[:, start_idx:end_idx, :]
    return X_crop, start_idx, end_idx
