"""ERP-related figures."""


from typing import List, Tuple

import matplotlib as mpl
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from .._style import (
    FACE_COLOR,
    MARKER_SIZE,
    WIDTH,
    DURATION_COLOR_1,
    DURATION_COLOR_2,
    LATENCY_COLOR_1,
    LATENCY_COLOR_2,
    SMALLER_MARKER_SIZE
)
from .._utils import _get_mask
from ..components.electrodes import _plot_selection_electrode_time_course


def plot_topo_selection(t: np.ndarray, p: np.ndarray, cluster: List[Tuple], info: mne.Info, data_tmin: float,
                        time: float, p_threshold: float = 0.01,
                        lim_val: float | None = None) -> Tuple[plt.Figure, float]:

    # Make significance masks
    mask = _get_mask(t, p, cluster, p_threshold)

    # Get the largest absolute t value as the limit
    if lim_val is None:
        lim_val = max(t.max(), abs(t.min()))

    # Convert to MNE evoked: crop the margin
    t = mne.EvokedArray(t.T * 1e-6, info, tmin=data_tmin)  # µV t-values → V for MNE topomap

    # Plot topographies
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    mask_params = {"marker": "o", "markerfacecolor": FACE_COLOR, "markeredgecolor": "k",
                   "linewidth": 0, "markersize": MARKER_SIZE}
    t.plot_topomap(axes=ax, times=[time], colorbar=False, show=False, mask=mask.T,
                   vlim=(-lim_val, lim_val), time_unit="ms", time_format="", mask_params=mask_params)

    ax.set_xlabel("")
    ax.set_ylabel("")

    return fig, lim_val




def plot_erp_topo(duration_t: np.ndarray, latency_t: np.ndarray,
                  duration_p: np.ndarray, latency_p: np.ndarray,
                  duration_cluster: List[Tuple], latency_cluster: List[Tuple], info: mne.Info,
                  data_tmin: float, tmin: float, tmax: float, n_topo: int, p_threshold: float = 0.01) -> plt.Figure:
    """
    Plot two rows of topography maps, one for each time step (specified by 'tmin', 'tmax' and 'n_topo')
    First row is duration comparison, the second row is latency comparison

    Parameters
    ----------
    duration_t: np.ndarray
        duration t values

    latency_t: np.ndarray
        latency t values

    duration_p: np.ndarray
        duration p values

    latency_p: np.ndarray
        latency p values

    duration_cluster: List[Tuple]
        clusters for the duration comparison

    latency_cluster: List[Tuple]
        clusters for the latency comparison

    info: mne.Info
        info object from the evoked object

    data_tmin: float
        start of epoch data

    tmin: float
        start of the timestamp to plot topography for

    tmax: float
        end of the timestamp to plot topography for

    n_topo: int
        total number of topographies to plot

    p_threshold: float
        significance threshold

    Returns
    -------
    plt.Figure
        figure
    """

    # Make significance masks
    duration_mask = _get_mask(duration_t, duration_p, duration_cluster, p_threshold)
    latency_mask = _get_mask(latency_t, latency_p, latency_cluster, p_threshold)

    # Get the largest absolute t value as the limit
    lim_val = max(duration_t.max(), latency_t.max(), abs(duration_t.min()), abs(latency_t.min()))

    # Convert to MNE evoked: crop the margin
    duration_t = mne.EvokedArray(duration_t.T * 1e-6, info, tmin=data_tmin)  # µV t-values → V for MNE topomap
    latency_t = mne.EvokedArray(latency_t.T * 1e-6, info, tmin=data_tmin)  # µV t-values → V for MNE topomap

    # Timesteps, include 0.0
    timesteps = np.linspace(tmin, tmax, n_topo)

    # Plot topographies
    fig, axes = plt.subplots(2, timesteps.size, figsize=(15, 3))

    mask_params = {"marker": "o", "markerfacecolor": FACE_COLOR, "markeredgecolor": "k",
                   "linewidth": 0, "markersize": SMALLER_MARKER_SIZE}
    duration_t.plot_topomap(axes=axes[0, :], times=timesteps, colorbar=False, show=False, mask=duration_mask.T,
                            vlim=(-lim_val, lim_val), time_unit="ms", mask_params=mask_params)
    latency_t.plot_topomap(axes=axes[1, :], times=timesteps, colorbar=False, show=False, mask=latency_mask.T,
                           vlim=(-lim_val, lim_val), time_unit="ms", time_format="",
                           mask_params=mask_params)  # no time label

    # Left label
    axes[0, 0].set_ylabel("Duration")
    axes[1, 0].set_ylabel("Latency")

    # Colorbar
    fig.subplots_adjust(left=0.02, right=0.95, top=0.9, bottom=0.05, hspace=0.05, wspace=0.0)
    cbar_ax = fig.add_axes((0.97, 0.15, 0.01, 0.7))  # (left, bottom, width, height)
    cbar_ax.set_ylabel("t values", rotation=270)
    norm = mpl.colors.Normalize(vmin=-lim_val, vmax=lim_val)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

    return fig




def plot_electrode_time_course(long_list: List[mne.Evoked], short_list: List[mne.Evoked],
                               fast_list: List[mne.Evoked], slow_list: List[mne.Evoked],
                               xmin: float = -1500, xmax: float = 500,
                               ymin: float = -2.8, ymax: float = 1.9) -> plt.Figure:
    """
    Plot time course of ERP amplitudes for Fz and Pz for duration/latency comparison

    Parameters
    ----------
    long_list: List[mne.Evoked]
        list of subject level evoked for long condition

    short_list: List[mne.Evoked]
        list of subject level evoked for short condition

    fast_list: List[mne.Evoked]
        list of subject level evoked for fast condition

    slow_list: List[mne.Evoked]
        list of subject level evoked for slow condition

    xmin: float
        x limit minimum to plot

    xmax: float
        x limit maximum to plot

    ymin: float
        y limit minimum to plot

    ymax: float
        y limit maximum to plot

    Returns
    -------
    plt.Figure
        figure
    """

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, WIDTH * 0.78))

    # Duration FC6
    _plot_selection_electrode_time_course(long_list, short_list, axes[0, 0], electrode="Fz",
                                          label_1="long", label_2="short",
                                          color_1=DURATION_COLOR_1, color_2=DURATION_COLOR_2,
                                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                          y_large_label="Fz",
                                          title="Duration", xlabel="Time (ms)", ylabel="Amplitude ($\mu$V)")

    # Duration Pz
    _plot_selection_electrode_time_course(long_list, short_list, axes[1, 0], electrode="Pz",
                                          label_1="long", label_2="short",
                                          color_1=DURATION_COLOR_1, color_2=DURATION_COLOR_2,
                                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                          y_large_label="Pz",
                                          title=None, xlabel="Time (ms)", ylabel="Amplitude ($\mu$V)")

    # Latency FC6
    _plot_selection_electrode_time_course(fast_list, slow_list, axes[0, 1], electrode="Fz",
                                          label_1="fast", label_2="slow",
                                          color_1=LATENCY_COLOR_1, color_2=LATENCY_COLOR_2,
                                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                          title="Latency", xlabel="Time (ms)", ylabel="Amplitude ($\mu$V)")

    # Latency Pz
    _plot_selection_electrode_time_course(fast_list, slow_list, axes[1, 1], electrode="Pz",
                                          label_1="fast", label_2="slow",
                                          color_1=LATENCY_COLOR_1, color_2=LATENCY_COLOR_2,
                                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                          title=None, xlabel="Time (ms)", ylabel="Amplitude ($\mu$V)")

    return fig


def plot_latency_erp_with_histograms(fast_list: List[mne.Evoked], slow_list: List[mne.Evoked],
                                     df: pd.DataFrame, ymax: float = 2000) -> plt.Figure:
    """
    Plot time course of latency ERP comparison for two electrodes Fz/Pz together with speech offset histogram

    Parameters
    ----------
    fast_list: List[mne.Evoked]
        list of subject level evokeds for fast condition

    slow_list: List[mne.Evoked]
        list of subject level evokeds for slow condition

    df: pd.DataFrame
        metadata

    ymax: float
        y axis limit

    Returns
    -------
    plt.Figure
        figure
    """

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, WIDTH * 0.5))

    fig.suptitle("Latency effect and histogram")

    # Fz
    _plot_selection_electrode_time_course(fast_list, slow_list, axes[0], electrode="Fz",
                                          label_1="fast", label_2="slow",
                                          color_1=LATENCY_COLOR_1, color_2=LATENCY_COLOR_2,
                                          xmin=-1500, xmax=500, ymin=-2.8, ymax=2.8,
                                          title="Fz",
                                          xlabel="Time (ms)", ylabel="Amplitude ($\mu$V)", legend=False)

    ax0 = axes[0].twinx()
    # NOTE: this mutates `df` in-place (preserved for backward compatibility)
    df["latency"] = -df["latency"] * 1e3
    sns.histplot(df, x="latency", hue="condition", ax=ax0, palette=[LATENCY_COLOR_1, LATENCY_COLOR_2])
    axes[0].legend()

    legend = ax0.legend()
    legend.remove()

    ax0.set_ylim(0, ymax)

    ax0.set_yticks([])
    ax0.set_ylabel("")

    # Pz
    _plot_selection_electrode_time_course(fast_list, slow_list, axes[1], electrode="Pz",
                                          label_1="fast", label_2="slow",
                                          color_1=LATENCY_COLOR_1, color_2=LATENCY_COLOR_2,
                                          xmin=-1500, xmax=500, ymin=-2.8, ymax=2.8,
                                          title="Pz",
                                          xlabel="Time (ms)", ylabel="Amplitude ($\mu$V)", legend=False)

    ax1 = axes[1].twinx()
    sns.histplot(df, x="latency", hue="condition", ax=ax1, palette=[LATENCY_COLOR_1, LATENCY_COLOR_2])
    axes[1].legend()

    legend = ax1.legend()
    legend.remove()

    ax1.set_ylim(0, ymax)

    axes[1].set_yticks([])
    axes[1].set_ylabel("")

    return fig




def plot_joint_erps(long_list: List[mne.Evoked], short_list: List[mne.Evoked],
                    fast_list: List[mne.Evoked], slow_list: List[mne.Evoked],
                    times: List[float] =
                    (-2.0, -0.8, -0.5, 0.0)) -> Tuple[plt.Figure, plt.Figure, plt.Figure, plt.Figure]:
    """
    Plot joint plots for individual conditions (long, short, fast, slow)

    Parameters
    ----------
    long_list: List[mne.Evoked]
        list of subject evoked for long condition

    short_list: List[mne.Evoked]
        list of subject evoked for short condition

    fast_list: List[mne.Evoked]
        list of subject evoked for fast condition

    slow_list: List[mne.Evoked]
        list of subject evoked for slow condition

    times: List[float]
        tuple of time points to plot topography of

    Returns
    -------
    Tuple[plt.Figure, plt.Figure, plt.Figure, plt.Figure]
        figures for each condition
    """

    # Combine subject level evokeds
    long = mne.combine_evoked(long_list, weights="equal")
    short = mne.combine_evoked(short_list, weights="equal")
    fast = mne.combine_evoked(fast_list, weights="equal")
    slow = mne.combine_evoked(slow_list, weights="equal")

    # Plot joints
    long = long.plot_joint(times, title="Long responses", show=False)
    short = short.plot_joint(times, title="Short responses", show=False)
    fast = fast.plot_joint(times, title="Fast responses", show=False)
    slow = slow.plot_joint(times, title="Slow responses", show=False)

    return long, short, fast, slow

