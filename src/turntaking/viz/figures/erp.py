"""ERP-related figures."""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib as mpl
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
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


# =============================================================================
# Topomap grid (generic: ERP/TFR)
# =============================================================================
def _make_step_times_s(
    evoked: mne.Evoked,
    *,
    tmin_s: float,
    tmax_s: float,
    step_ms: float,
) -> np.ndarray:
    """
    Create step-based time points (in seconds), clipped and snapped to evoked samples.
    """
    if step_ms <= 0:
        raise ValueError(f"step_ms must be > 0, got {step_ms}.")

    req_tmin_s, req_tmax_s = _clip_time_range_to_evoked(evoked, tmin_s=tmin_s, tmax_s=tmax_s)

    # build ms grid (avoids awkward fractions); include endpoint
    tmin_ms = req_tmin_s * 1e3
    tmax_ms = req_tmax_s * 1e3
    times_ms = np.arange(tmin_ms, tmax_ms + 0.5 * step_ms, step_ms)

    # snap each time to nearest sample (keeps titles clean & consistent)
    t0 = float(evoked.times[0])
    sfreq = float(evoked.info["sfreq"])
    idx = np.round((times_ms / 1e3 - t0) * sfreq).astype(int)
    idx = np.clip(idx, 0, len(evoked.times) - 1)

    times_s = evoked.times[idx].astype(float)

    # de-duplicate in case snapping collapses neighbors
    times_s = np.unique(times_s)
    return times_s


def plot_stat_topomaps_grid(
    *,
    stat: np.ndarray,
    mask: Optional[np.ndarray],
    info: mne.Info,
    data_tmin: float,
    tmin: float,
    tmax: float,
    step_ms: float,
    title: str,
    lim_val: float | None = None,
    max_cols: int = 10,
    cmap: str = "RdBu_r",
    cbar_label: str = "t values",
    time_unit: str = "ms",
    time_format: str = "%0.0f ms",
    mask_marker_size: float = SMALLER_MARKER_SIZE,
    figure_profile: str = "jneuro_2col",
    save_basepath: str | Path | None = None,
    time_fontsize: float = 6
) -> plt.Figure:
    """
    Generic grid plotter for topomap time series (ERP/TFR compatible).

    Parameters
    ----------
    stat
        Array of shape (n_times, n_channels), typically t-values.
    mask
        Boolean array of shape (n_times, n_channels) or None.
        NOTE: this supports only a single marker style. If you want different
        clusters to have different markers, pass cluster identities and overlay
        markers (see extension note below).
    info
        MNE Info with montage.
    data_tmin
        Start time (seconds) corresponding to stat[0, :].
    tmin, tmax
        Requested time window (seconds) to display.
    step_ms
        Step size in milliseconds.
    title
        Figure title.
    lim_val
        If None, uses max abs(stat).
    max_cols
        Max number of columns before wrapping.
    """

    if stat.ndim != 2:
        raise ValueError(f"stat must be 2D (n_times, n_channels), got shape={stat.shape}.")
    n_times, n_ch = stat.shape

    if mask is not None and mask.shape != stat.shape:
        raise ValueError(f"mask must match stat shape. stat={stat.shape}, mask={mask.shape}")

    if step_ms <= 0:
        raise ValueError(f"step_ms must be > 0, got {step_ms}")

    # Color limits
    if lim_val is None:
        lim_val = float(np.max(np.abs(stat)))
    lim_val = float(lim_val)

    # MNE expects (n_channels, n_times) in Volts for EvokedArray
    evoked = mne.EvokedArray(stat.T * 1e-6, info, tmin=float(data_tmin))

    times_s = _make_step_times_s(evoked, tmin_s=float(tmin), tmax_s=float(tmax), step_ms=float(step_ms))
    if times_s.size == 0:
        raise ValueError("No time points selected after clipping/snapping. Check tmin/tmax/step_ms.")

    # Wrap into grid
    n_maps = int(times_s.size)
    n_cols = int(min(max_cols, n_maps))
    n_rows = int(np.ceil(n_maps / n_cols))

    # Figure size: keep panels big; width grows with columns, height with rows
    # You can tune these constants, but this is a sane starting point.
    panel_w = 1.35
    panel_h = 1.35
    cbar_col_w = 0.55  # inches-ish worth of space, controlled by GridSpec ratio below

    fig_w = max(9.0, panel_w * n_cols + cbar_col_w)
    fig_h = max(5.0, panel_h * n_rows + 0.8)

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Dedicated colorbar column keeps topomaps from shrinking
    gs = GridSpec(
        n_rows,
        n_cols + 1,
        figure=fig,
        width_ratios=[1.0] * n_cols + [0.08],  # last column is narrow colorbar
        wspace=0.05,
        hspace=0.10,
    )

    axes_flat: list[plt.Axes] = []
    for r in range(n_rows):
        for c in range(n_cols):
            axes_flat.append(fig.add_subplot(gs[r, c]))

    cbar_ax = fig.add_subplot(gs[:, -1])

    # Hide unused axes
    for ax in axes_flat[n_maps:]:
        ax.set_visible(False)

    mask_params = {
        "marker": "o",
        "markerfacecolor": FACE_COLOR,
        "markeredgecolor": "k",
        "linewidth": 0,
        "markersize": float(mask_marker_size),
    }

    # Draw topomaps (no built-in colorbar; we add our own in cbar_ax)
    evoked.plot_topomap(
        axes=axes_flat[:n_maps],
        times=times_s,
        colorbar=False,
        show=False,
        mask=None if mask is None else mask.T,  # MNE expects (n_channels, n_times)
        vlim=(-lim_val, lim_val),
        time_unit=time_unit,
        time_format=time_format,
        mask_params=mask_params,
        cmap=cmap,
    )

    # Move timestamps inside each panel so they don't "float" into the row above
    for ax in axes_flat[:n_maps]:
        t = ax.get_title()
        ax.set_title("")  # remove default MNE title

        ax.text(
            0.5,
            1.2,  # inside the axes (top)
            t,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=time_fontsize,  # your desired timestamp size
            color="0.2",
            zorder=20,
        )

    for ax in axes_flat[:n_maps]:
        ax.title.set_fontsize(time_fontsize)

    fig.suptitle(title)

    # Our dedicated, slim colorbar
    norm = mpl.colors.Normalize(vmin=-lim_val, vmax=lim_val)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=14)
    cbar.ax.tick_params(labelsize=8)

    _maybe_save(fig, save_basepath=save_basepath, figure_profile=figure_profile)
    return fig


# =============================================================================
# ERP wrappers: separate figures
# =============================================================================
def plot_erp_topo_duration(
    duration_t: np.ndarray,
    duration_p: np.ndarray,
    duration_cluster: List[Tuple],
    info: mne.Info,
    data_tmin: float,
    tmin: float,
    tmax: float,
    step_ms: float,
    p_threshold: float = 0.01,
    *,
    max_cols: int = 10,
    figure_profile: str = "jneuro_2col",
    save_basepath: str | Path | None = None,
) -> plt.Figure:
    mask = _get_mask(duration_t, duration_p, duration_cluster, p_threshold)
    lim_val = float(np.max(np.abs(duration_t)))
    return plot_stat_topomaps_grid(
        stat=duration_t,
        mask=mask,
        info=info,
        data_tmin=data_tmin,
        tmin=tmin,
        tmax=tmax,
        step_ms=step_ms,
        title="ERP topographies (Duration)",
        lim_val=lim_val,
        max_cols=max_cols,
        figure_profile=figure_profile,
        save_basepath=save_basepath,
    )


def plot_erp_topo_latency(
    latency_t: np.ndarray,
    latency_p: np.ndarray,
    latency_cluster: List[Tuple],
    info: mne.Info,
    data_tmin: float,
    tmin: float,
    tmax: float,
    step_ms: float,
    p_threshold: float = 0.01,
    *,
    max_cols: int = 10,
    figure_profile: str = "jneuro_2col",
    save_basepath: str | Path | None = None,
) -> plt.Figure:
    mask = _get_mask(latency_t, latency_p, latency_cluster, p_threshold)
    lim_val = float(np.max(np.abs(latency_t)))
    return plot_stat_topomaps_grid(
        stat=latency_t,
        mask=mask,
        info=info,
        data_tmin=data_tmin,
        tmin=tmin,
        tmax=tmax,
        step_ms=step_ms,
        title="ERP topographies (Latency)",
        lim_val=lim_val,
        max_cols=max_cols,
        figure_profile=figure_profile,
        save_basepath=save_basepath,
    )


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

