"""ERP-related figures."""

from pathlib import Path
from typing import List, Tuple

import matplotlib as mpl
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from turntaking.viz._style import (
    FACE_COLOR,
    MARKER_SIZE,
    WIDTH,
    DURATION_COLOR_1,
    DURATION_COLOR_2,
    LATENCY_COLOR_1,
    LATENCY_COLOR_2,
    SMALLER_MARKER_SIZE,
    save_figure
)
from .._utils import _get_mask
from ..components.electrodes import _plot_selection_electrode_time_course


def plot_topo_selection(
    t: np.ndarray,
    p: np.ndarray,
    cluster: List[Tuple],
    info: mne.Info,
    data_tmin: float,
    time: float,
    p_threshold: float = 0.01,
    lim_val: float | None = None,
    *,
    figure_profile: str = "jneuro_1col",
    save_basepath: str | Path | None = None,
) -> Tuple[plt.Figure, float]:

    # Make significance masks
    mask = _get_mask(t, p, cluster, p_threshold)

    # Get the largest absolute t value as the limit
    if lim_val is None:
        lim_val = max(t.max(), abs(t.min()))

    # Convert to MNE evoked: crop the margin
    t = mne.EvokedArray(t.T * 1e-6, info, tmin=data_tmin)  # µV t-values → V for MNE topomap

    # Plot topographies
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    mask_params = {
        "marker": "o",
        "markerfacecolor": FACE_COLOR,
        "markeredgecolor": "k",
        "linewidth": 0,
        "markersize": MARKER_SIZE,
    }
    t.plot_topomap(
        axes=ax,
        times=[time],
        colorbar=False,
        show=False,
        mask=mask.T,
        vlim=(-lim_val, lim_val),
        time_unit="ms",
        time_format="",
        mask_params=mask_params,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    _maybe_save(fig, save_basepath=save_basepath, figure_profile=figure_profile)
    return fig, lim_val


def plot_erp_topo(
    duration_t: np.ndarray,
    latency_t: np.ndarray,
    duration_p: np.ndarray,
    latency_p: np.ndarray,
    duration_cluster: List[Tuple],
    latency_cluster: List[Tuple],
    info: mne.Info,
    data_tmin: float,
    tmin: float,
    tmax: float,
    n_topo: int,
    p_threshold: float = 0.01,
    *,
    figure_profile: str = "jneuro_2col",
    save_basepath: str | Path | None = None,
) -> plt.Figure:
    """
    Plot two rows of topography maps, one for each time step (specified by 'tmin', 'tmax' and 'n_topo')
    First row is duration comparison, the second row is latency comparison
    """

    # Make significance masks
    duration_mask = _get_mask(duration_t, duration_p, duration_cluster, p_threshold)
    latency_mask = _get_mask(latency_t, latency_p, latency_cluster, p_threshold)

    # Get the largest absolute t value as the limit
    lim_val = max(duration_t.max(), latency_t.max(), abs(duration_t.min()), abs(latency_t.min()))

    # Convert to MNE evoked: crop the margin
    duration_t = mne.EvokedArray(duration_t.T * 1e-6, info, tmin=data_tmin)
    latency_t = mne.EvokedArray(latency_t.T * 1e-6, info, tmin=data_tmin)

    req_tmin_s, req_tmax_s = _clip_time_range_to_evoked(
        duration_t,
        tmin_s=tmin,
        tmax_s=tmax,
    )

    # Timesteps
    timesteps = np.linspace(req_tmin_s, req_tmax_s, n_topo)

    # Plot topographies
    fig, axes = plt.subplots(2, timesteps.size, figsize=(15, 3))

    mask_params = {
        "marker": "o",
        "markerfacecolor": FACE_COLOR,
        "markeredgecolor": "k",
        "linewidth": 0,
        "markersize": SMALLER_MARKER_SIZE,
    }
    duration_t.plot_topomap(
        axes=axes[0, :],
        times=timesteps,
        colorbar=False,
        show=False,
        mask=duration_mask.T,
        vlim=(-lim_val, lim_val),
        time_unit="ms",
        mask_params=mask_params,
    )
    latency_t.plot_topomap(
        axes=axes[1, :],
        times=timesteps,
        colorbar=False,
        show=False,
        mask=latency_mask.T,
        vlim=(-lim_val, lim_val),
        time_unit="ms",
        time_format="",
        mask_params=mask_params,
    )

    axes[0, 0].set_ylabel("Duration")
    axes[1, 0].set_ylabel("Latency")

    # Colorbar
    fig.subplots_adjust(left=0.02, right=0.95, top=0.9, bottom=0.05, hspace=0.05, wspace=0.0)
    cbar_ax = fig.add_axes((0.97, 0.15, 0.01, 0.7))
    cbar_ax.set_ylabel("t values", rotation=270)
    norm = mpl.colors.Normalize(vmin=-lim_val, vmax=lim_val)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

    _maybe_save(fig, save_basepath=save_basepath, figure_profile=figure_profile)
    return fig


def plot_electrode_time_course(
    long_list: List[mne.Evoked],
    short_list: List[mne.Evoked],
    fast_list: List[mne.Evoked],
    slow_list: List[mne.Evoked],
    xmin: float = -1500,
    xmax: float = 500,
    ymin: float = -2.8,
    ymax: float = 1.9,
    *,
    figure_profile: str = "jneuro_2col",
    save_basepath: str | Path | None = None,
) -> plt.Figure:
    """
    Plot time course of ERP amplitudes for Fz and Pz for duration/latency comparison
    """

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, WIDTH * 0.78))

    _plot_selection_electrode_time_course(
        long_list,
        short_list,
        axes[0, 0],
        electrode="Fz",
        label_1="long",
        label_2="short",
        color_1=DURATION_COLOR_1,
        color_2=DURATION_COLOR_2,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        y_large_label="Fz",
        title="Duration",
        xlabel="Time (ms)",
        ylabel="Amplitude ($\\mu$V)",
    )

    _plot_selection_electrode_time_course(
        long_list,
        short_list,
        axes[1, 0],
        electrode="Pz",
        label_1="long",
        label_2="short",
        color_1=DURATION_COLOR_1,
        color_2=DURATION_COLOR_2,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        y_large_label="Pz",
        title=None,
        xlabel="Time (ms)",
        ylabel="Amplitude ($\\mu$V)",
    )

    _plot_selection_electrode_time_course(
        fast_list,
        slow_list,
        axes[0, 1],
        electrode="Fz",
        label_1="fast",
        label_2="slow",
        color_1=LATENCY_COLOR_1,
        color_2=LATENCY_COLOR_2,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        title="Latency",
        xlabel="Time (ms)",
        ylabel="Amplitude ($\\mu$V)",
    )

    _plot_selection_electrode_time_course(
        fast_list,
        slow_list,
        axes[1, 1],
        electrode="Pz",
        label_1="fast",
        label_2="slow",
        color_1=LATENCY_COLOR_1,
        color_2=LATENCY_COLOR_2,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        title=None,
        xlabel="Time (ms)",
        ylabel="Amplitude ($\\mu$V)",
    )

    # After all panels are drawn
    fig.align_ylabels(axes[:, 0])  # aligns left-column ylabels

    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        bottom=0.10,
        top=0.92,
        wspace=0.25,
        hspace=0.25,
    )

    _maybe_save(fig, save_basepath=save_basepath, figure_profile=figure_profile)
    return fig


def plot_latency_erp_with_histograms(
    fast_list: List[mne.Evoked],
    slow_list: List[mne.Evoked],
    df: pd.DataFrame,
    ymax: float = 2000,
    *,
    figure_profile: str = "jneuro_2col",
    save_basepath: str | Path | None = None,
) -> plt.Figure:
    """
    Plot time course of latency ERP comparison for two electrodes Fz/Pz together with speech offset histogram
    """

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, WIDTH * 0.5))
    fig.suptitle("Latency effect and histogram")

    _plot_selection_electrode_time_course(
        fast_list,
        slow_list,
        axes[0],
        electrode="Fz",
        label_1="fast",
        label_2="slow",
        color_1=LATENCY_COLOR_1,
        color_2=LATENCY_COLOR_2,
        xmin=-1500,
        xmax=500,
        ymin=-2.8,
        ymax=2.8,
        title="Fz",
        xlabel="Time (ms)",
        ylabel="Amplitude ($\\mu$V)",
        legend=False,
    )

    ax0 = axes[0].twinx()

    # IMPORTANT: avoid mutating caller's DataFrame in-place
    latency_df = df.copy()
    latency_df["latency"] = -latency_df["latency"] * 1e3

    sns.histplot(latency_df, x="latency", hue="condition", ax=ax0, palette=[LATENCY_COLOR_1, LATENCY_COLOR_2])
    axes[0].legend()

    legend = ax0.legend()
    legend.remove()

    ax0.set_ylim(0, ymax)
    ax0.set_yticks([])
    ax0.set_ylabel("")

    _plot_selection_electrode_time_course(
        fast_list,
        slow_list,
        axes[1],
        electrode="Pz",
        label_1="fast",
        label_2="slow",
        color_1=LATENCY_COLOR_1,
        color_2=LATENCY_COLOR_2,
        xmin=-1500,
        xmax=500,
        ymin=-2.8,
        ymax=2.8,
        title="Pz",
        xlabel="Time (ms)",
        ylabel="Amplitude ($\\mu$V)",
        legend=False,
    )

    ax1 = axes[1].twinx()
    sns.histplot(latency_df, x="latency", hue="condition", ax=ax1, palette=[LATENCY_COLOR_1, LATENCY_COLOR_2])
    axes[1].legend()

    legend = ax1.legend()
    legend.remove()

    ax1.set_ylim(0, ymax)
    axes[1].set_yticks([])
    axes[1].set_ylabel("")

    _maybe_save(fig, save_basepath=save_basepath, figure_profile=figure_profile)
    return fig


def plot_joint_erps(
    long_list: List[mne.Evoked],
    short_list: List[mne.Evoked],
    fast_list: List[mne.Evoked],
    slow_list: List[mne.Evoked],
    times: List[float] = (-2.0, -0.8, -0.5, 0.0),
    *,
    figure_profile: str = "jneuro_2col",
    save_basepath: str | Path | None = None,
) -> Tuple[plt.Figure, plt.Figure, plt.Figure, plt.Figure]:
    """
    Plot joint plots for individual conditions (long, short, fast, slow)
    """

    long = mne.combine_evoked(long_list, weights="equal")
    short = mne.combine_evoked(short_list, weights="equal")
    fast = mne.combine_evoked(fast_list, weights="equal")
    slow = mne.combine_evoked(slow_list, weights="equal")

    long_fig = long.plot_joint(times, title="Long responses", show=False)
    short_fig = short.plot_joint(times, title="Short responses", show=False)
    fast_fig = fast.plot_joint(times, title="Fast responses", show=False)
    slow_fig = slow.plot_joint(times, title="Slow responses", show=False)

    if save_basepath is not None:
        base = Path(save_basepath)
        save_figure(long_fig, base.with_name(base.name + "_long"), profile_name=figure_profile)
        save_figure(short_fig, base.with_name(base.name + "_short"), profile_name=figure_profile)
        save_figure(fast_fig, base.with_name(base.name + "_fast"), profile_name=figure_profile)
        save_figure(slow_fig, base.with_name(base.name + "_slow"), profile_name=figure_profile)

    return long_fig, short_fig, fast_fig, slow_fig


def _maybe_save(
    fig: plt.Figure,
    save_basepath: str | Path | None,
    figure_profile: str,
) -> None:
    """
    Save the figure if a save path is provided.

    Parameters
    ----------
    fig : plt.Figure
        Figure object.
    save_basepath : str | Path | None
        Base path (no extension). If None, do not save.
    figure_profile : str
        Figure sizing/export profile name.

    Usage example
    -------------
        _maybe_save(fig, "out/figures/erp_topos", "jneuro_1col")
    """
    if save_basepath is None:
        return
    save_figure(fig, save_basepath=save_basepath, profile_name=figure_profile)


def _clip_time_range_to_evoked(
    evoked: mne.Evoked,
    *,
    tmin_s: float,
    tmax_s: float,
) -> tuple[float, float]:
    t0 = float(evoked.times[0])
    t1 = float(evoked.times[-1])

    # clip to available range
    tmin_s = max(float(tmin_s), t0)
    tmax_s = min(float(tmax_s), t1)

    # snap to nearest sample (avoids "0.0" when last sample is -0.00195)
    sfreq = float(evoked.info["sfreq"])
    i_min = int(np.round((tmin_s - t0) * sfreq))
    i_max = int(np.round((tmax_s - t0) * sfreq))

    i_min = max(0, min(i_min, len(evoked.times) - 1))
    i_max = max(0, min(i_max, len(evoked.times) - 1))

    tmin_s_snapped = float(evoked.times[i_min])
    tmax_s_snapped = float(evoked.times[i_max])
    return tmin_s_snapped, tmax_s_snapped

