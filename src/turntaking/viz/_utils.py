"""Internal helpers shared across visualization modules.

These functions are intentionally *not* exported as part of the public API.
They exist solely to keep the figure-producing functions in `viz.figures`
small and readable while preserving identical outputs.
"""

from __future__ import annotations

from typing import List, Tuple

import mne
import numpy as np
import scipy


def _get_mask(t_values: np.ndarray, p_values: np.ndarray, cluster_list: List[Tuple], p_threshold: float) -> np.ndarray:
    """
    Create a significance mask from t-values, p-values, clusters and p threshold

    Parameters
    ----------
    t_values: np.ndarray
        t values (per cluster), (n_times, n_channels)

    p_values: np.ndarray
        p values (per cluster), (n_cluster,)

    cluster_list: List[Tuple]
        clusters (list of indices), n_cluster list of tuples of (n_times, n_channels)

    p_threshold: float
        p significance threshold

    Returns
    -------
    mask: np.ndarray
        significance mask (n_times, n_channels)
    """

    mask = np.zeros_like(t_values).astype(bool)
    for idx, cluster in enumerate(cluster_list):

        if p_values[idx] > p_threshold:
            continue

        mask[cluster] = True

    return mask




def _ci(data: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute confidence interval

    Parameters
    ----------
    data: np.ndarray
        1D, data along sample axis

    confidence: float
        confidence level

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        mean: mean time series
        lower: lower confidence interval
        upper: upper confidence interval
    """

    n = data.size
    mean = data.mean()
    se = scipy.stats.sem(data)

    height = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, mean - height, mean + height




def _get_dummy_raw() -> mne.io.BaseRaw:
    """
    Make a dummy raw containing no data

    Returns
    -------
    raw: mne.io.BaseRaw
        dummy raw object for plotting
    """

    montage = mne.channels.make_standard_montage("biosemi64")
    info = mne.create_info(ch_names=montage.ch_names, sfreq=514)
    dummy = np.zeros((64, 1))
    raw = mne.io.RawArray(dummy, info)
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    raw.set_montage(montage, on_missing="ignore")

    return raw




def _get_targets(evokeds: List[mne.Evoked], picks: List[str]) -> np.ndarray:
    """
    Get the data for the picked channels per subject and concatenate into a single numpy array

    Parameters
    ----------
    evokeds: List[mne.Evoked]
        list of subject level evoked objects

    picks: List[str]
        list of channels to pick

    Returns
    -------
    data: np.ndarray
        picked data, (n_subjects, n_channels, n_times)
    """

    data_list = []
    for evoked in evokeds:
        data = evoked.copy().pick(picks).get_data()  # (n_channels, n_times)
        data_list.append(data)

    data = np.stack(data_list)
    return data


