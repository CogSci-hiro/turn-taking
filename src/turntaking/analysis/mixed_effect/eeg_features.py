from typing import Sequence

import numpy as np
import pandas as pd
import mne

from .constants import ALPHA_BAND, BETA_BAND


def compute_run_eeg_features(
    epochs: mne.BaseEpochs,
    *,
    tw1_tmin: float,
    tw1_tmax: float,
    tw2_tmin: float,
    tw2_tmax: float,
    baseline_tmin: float,
    baseline_tmax: float,
    anterior_picks: Sequence[str],
    posterior_picks: Sequence[str],
) -> pd.DataFrame:
    """
    Compute ERP mean amplitude and band-power summaries per epoch for ROIs + time windows.

    Parameters
    ----------
    epochs
        Epoched EEG.
    tw1_tmin, tw1_tmax, tw2_tmin, tw2_tmax, baseline_tmin, baseline_tmax
        Time windows (seconds).
    anterior_picks, posterior_picks
        Channel name lists for ROIs.

    Returns
    -------
    pd.DataFrame
        Trial-level features table with columns:
        - tw1_mean_anterior, tw1_mean_posterior, tw2_mean_anterior, tw2_mean_posterior
        - tw1_alpha_anterior, ..., tw2_alpha_posterior
        - tw1_beta_anterior,  ..., tw2_beta_posterior
        - baseline_mean_anterior, baseline_mean_posterior

    Example table
    -------------
    | tw1_mean_anterior | tw1_alpha_posterior | baseline_mean_posterior |
    |---:|---:|---:|
    | -0.83 | 2.11 | 0.05 |
    |  0.12 | 1.44 | -0.02 |

    Usage example
    -------------
        df = compute_run_eeg_features(
            epochs,
            tw1_tmin=0.05, tw1_tmax=0.15,
            tw2_tmin=0.20, tw2_tmax=0.35,
            baseline_tmin=-0.20, baseline_tmax=0.0,
            anterior_picks=["Fz", "FCz"],
            posterior_picks=["Pz", "POz"],
        )
    """
    tw1_mean_anterior = _time_window_mean_uV(epochs, tw1_tmin, tw1_tmax, anterior_picks)
    tw1_mean_posterior = _time_window_mean_uV(epochs, tw1_tmin, tw1_tmax, posterior_picks)
    tw2_mean_anterior = _time_window_mean_uV(epochs, tw2_tmin, tw2_tmax, anterior_picks)
    tw2_mean_posterior = _time_window_mean_uV(epochs, tw2_tmin, tw2_tmax, posterior_picks)

    alpha_epochs = _band_power_epochs(epochs.copy(), fmin=ALPHA_BAND[0], fmax=ALPHA_BAND[1])
    beta_epochs = _band_power_epochs(epochs.copy(), fmin=BETA_BAND[0], fmax=BETA_BAND[1])

    tw1_alpha_anterior = _time_window_mean(alpha_epochs, tw1_tmin, tw1_tmax, anterior_picks)
    tw1_alpha_posterior = _time_window_mean(alpha_epochs, tw1_tmin, tw1_tmax, posterior_picks)
    tw2_alpha_anterior = _time_window_mean(alpha_epochs, tw2_tmin, tw2_tmax, anterior_picks)
    tw2_alpha_posterior = _time_window_mean(alpha_epochs, tw2_tmin, tw2_tmax, posterior_picks)

    tw1_beta_anterior = _time_window_mean(beta_epochs, tw1_tmin, tw1_tmax, anterior_picks)
    tw1_beta_posterior = _time_window_mean(beta_epochs, tw1_tmin, tw1_tmax, posterior_picks)
    tw2_beta_anterior = _time_window_mean(beta_epochs, tw2_tmin, tw2_tmax, anterior_picks)
    tw2_beta_posterior = _time_window_mean(beta_epochs, tw2_tmin, tw2_tmax, posterior_picks)

    baseline_mean_anterior = _time_window_mean_uV(epochs, baseline_tmin, baseline_tmax, anterior_picks)
    baseline_mean_posterior = _time_window_mean_uV(epochs, baseline_tmin, baseline_tmax, posterior_picks)

    return pd.DataFrame(
        {
            "tw1_mean_anterior": tw1_mean_anterior,
            "tw1_mean_posterior": tw1_mean_posterior,
            "tw2_mean_anterior": tw2_mean_anterior,
            "tw2_mean_posterior": tw2_mean_posterior,
            "tw1_alpha_anterior": tw1_alpha_anterior,
            "tw1_alpha_posterior": tw1_alpha_posterior,
            "tw2_alpha_anterior": tw2_alpha_anterior,
            "tw2_alpha_posterior": tw2_alpha_posterior,
            "tw1_beta_anterior": tw1_beta_anterior,
            "tw1_beta_posterior": tw1_beta_posterior,
            "tw2_beta_anterior": tw2_beta_anterior,
            "tw2_beta_posterior": tw2_beta_posterior,
            "baseline_mean_anterior": baseline_mean_anterior,
            "baseline_mean_posterior": baseline_mean_posterior,
        }
    )


def _time_window_mean_uV(
    epochs: mne.BaseEpochs,
    tmin: float,
    tmax: float,
    picks: Sequence[str],
) -> list[float]:
    values = _time_window_mean(epochs, tmin, tmax, picks)
    return (np.array(values) * 1e6).tolist()


def _time_window_mean(
    epochs: mne.BaseEpochs,
    tmin: float,
    tmax: float,
    picks: Sequence[str],
) -> list[float]:
    data = epochs.get_data(copy=True)  # (n_epochs, n_channels, n_times)
    times = epochs.times

    t_mask = (times >= tmin) & (times <= tmax)
    if not np.any(t_mask):
        raise ValueError(f"No samples in time window [{tmin}, {tmax}]")

    ch_idx = [epochs.ch_names.index(ch) for ch in picks if ch in epochs.ch_names]
    if len(ch_idx) == 0:
        raise ValueError("No requested picks found in epochs.ch_names.")

    windowed = data[:, ch_idx, :][:, :, t_mask]          # (n_epochs, n_picks, n_t)
    per_trial = windowed.mean(axis=2).mean(axis=1)       # mean over time then channels
    return per_trial.tolist()


def _band_power_epochs(epochs: mne.BaseEpochs, *, fmin: float, fmax: float) -> mne.BaseEpochs:
    epochs.load_data().filter(l_freq=fmin, h_freq=fmax)
    epochs.apply_hilbert(envelope=True)
    return epochs
