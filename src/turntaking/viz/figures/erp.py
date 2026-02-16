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
    mask = _get_mask(t, p, cluster, p_threshold)
    if lim_val is None:
        lim_val = max(t.max(), abs(t.min()))
    t = mne.EvokedArray(t.T * 1e-6, info, tmin=data_tmin)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    t.plot_topomap(
        axes=ax,
        times=[time],
        colorbar=False,
        show=False,
        mask=mask.T,
        vlim=(-lim_val, lim_val),
        time_unit="ms",
        time_format="",
        mask_params=_topomap_mask_params(float(MARKER_SIZE)),
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    _maybe_save(fig, save_basepath=save_basepath, figure_profile=figure_profile)
    return fig, lim_val


# =============================================================================
# Topomap grid (generic: ERP/TFR)
# =============================================================================
def _topomap_mask_params(marker_size: float) -> dict[str, float | str]:
    return {
        "marker": "o",
        "markerfacecolor": FACE_COLOR,
        "markeredgecolor": "k",
        "linewidth": 0,
        "markersize": float(marker_size),
    }


def _make_step_times_s_and_labels_ms(
    evoked: mne.Evoked,
    *,
    tmin_s: float,
    tmax_s: float,
    step_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return snapped times (seconds) for plotting and clean labels (ms) for display.
    """
    if step_ms <= 0:
        raise ValueError(f"step_ms must be > 0, got {step_ms}.")

    req_tmin_s, req_tmax_s = _clip_time_range_to_evoked(evoked, tmin_s=tmin_s, tmax_s=tmax_s)

    # clean grid in ms (no weird decimals)
    tmin_ms = req_tmin_s * 1e3
    tmax_ms = req_tmax_s * 1e3
    grid_ms = np.arange(tmin_ms, tmax_ms + 0.5 * step_ms, step_ms)

    # snap to sample indices
    t0 = float(evoked.times[0])
    sfreq = float(evoked.info["sfreq"])
    idx = np.round((grid_ms / 1e3 - t0) * sfreq).astype(int)
    idx = np.clip(idx, 0, len(evoked.times) - 1)

    # de-duplicate (snapping can collapse neighbors); keep first occurrence label
    keep = np.concatenate(([True], idx[1:] != idx[:-1]))
    idx = idx[keep]
    grid_ms = grid_ms[keep]

    times_s = evoked.times[idx].astype(float)

    # make labels *exact* multiples of step_ms (avoids 1699)
    labels_ms = np.round(grid_ms / step_ms) * step_ms

    return times_s, labels_ms.astype(int)


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
    _validate_topomap_inputs(stat=stat, mask=mask, step_ms=step_ms)
    lim_val = _resolve_lim_val(stat=stat, lim_val=lim_val)
    evoked = mne.EvokedArray(stat.T * 1e-6, info, tmin=float(data_tmin))
    times_s, labels_ms = _make_step_times_s_and_labels_ms(evoked, tmin_s=tmin, tmax_s=tmax, step_ms=step_ms)
    if times_s.size == 0:
        raise ValueError("No time points selected after clipping/snapping. Check tmin/tmax/step_ms.")

    fig, axes_flat, cbar_ax = _make_topomap_grid(n_maps=int(times_s.size), max_cols=max_cols)
    draw_axes = axes_flat[: int(times_s.size)]
    _plot_topomap_series(
        evoked=evoked,
        axes=draw_axes,
        times_s=times_s,
        mask=mask,
        lim_val=lim_val,
        time_unit=time_unit,
        time_format=time_format,
        mask_marker_size=mask_marker_size,
        cmap=cmap,
    )
    _annotate_topomap_times(draw_axes, labels_ms=labels_ms, time_fontsize=time_fontsize)
    fig.suptitle(title)
    _add_topomap_colorbar(fig, cbar_ax=cbar_ax, lim_val=lim_val, cmap=cmap, cbar_label=cbar_label)
    _maybe_save(fig, save_basepath=save_basepath, figure_profile=figure_profile)
    return fig


def _validate_topomap_inputs(*, stat: np.ndarray, mask: Optional[np.ndarray], step_ms: float) -> None:
    if stat.ndim != 2:
        raise ValueError(f"stat must be 2D (n_times, n_channels), got shape={stat.shape}.")
    if mask is not None and mask.shape != stat.shape:
        raise ValueError(f"mask must match stat shape. stat={stat.shape}, mask={mask.shape}")
    if step_ms <= 0:
        raise ValueError(f"step_ms must be > 0, got {step_ms}")


def _resolve_lim_val(*, stat: np.ndarray, lim_val: float | None) -> float:
    if lim_val is None:
        return float(np.max(np.abs(stat)))
    return float(lim_val)


def _make_topomap_grid(n_maps: int, max_cols: int) -> tuple[plt.Figure, list[plt.Axes], plt.Axes]:
    n_cols = int(min(max_cols, n_maps))
    n_rows = int(np.ceil(n_maps / n_cols))
    fig_w = max(9.0, 1.35 * n_cols + 0.55)
    fig_h = max(5.0, 1.35 * n_rows + 0.8)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        n_rows,
        n_cols + 1,
        figure=fig,
        width_ratios=[1.0] * n_cols + [0.08],
        wspace=0.05,
        hspace=0.10,
    )
    axes_flat = [fig.add_subplot(gs[r, c]) for r in range(n_rows) for c in range(n_cols)]
    cbar_ax = fig.add_subplot(gs[:, -1])
    for ax in axes_flat[n_maps:]:
        ax.set_visible(False)
    return fig, axes_flat, cbar_ax


def _plot_topomap_series(
    *,
    evoked: mne.Evoked,
    axes: list[plt.Axes],
    times_s: np.ndarray,
    mask: Optional[np.ndarray],
    lim_val: float,
    time_unit: str,
    time_format: str,
    mask_marker_size: float,
    cmap: str,
) -> None:
    evoked.plot_topomap(
        axes=axes,
        times=times_s,
        colorbar=False,
        show=False,
        mask=None if mask is None else mask.T,
        vlim=(-lim_val, lim_val),
        time_unit=time_unit,
        time_format=time_format,
        mask_params=_topomap_mask_params(mask_marker_size),
        cmap=cmap,
    )


def _annotate_topomap_times(
    axes: list[plt.Axes],
    *,
    labels_ms: np.ndarray,
    time_fontsize: float,
) -> None:
    for ax, label_ms in zip(axes, labels_ms):
        ax.set_title("")
        ax.text(
            0.5,
            1.2,
            f"{label_ms:d} ms",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=time_fontsize,
            color="0.2",
            zorder=20,
        )
    for ax in axes:
        ax.title.set_fontsize(time_fontsize)


def _add_topomap_colorbar(
    fig: plt.Figure,
    *,
    cbar_ax: plt.Axes,
    lim_val: float,
    cmap: str,
    cbar_label: str,
) -> None:
    norm = mpl.colors.Normalize(vmin=-lim_val, vmax=lim_val)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=14)
    cbar.ax.tick_params(labelsize=8)


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
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, WIDTH * 0.78))
    _plot_duration_column(axes, long_list=long_list, short_list=short_list, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    _plot_latency_column(axes, fast_list=fast_list, slow_list=slow_list, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    _format_electrode_panel_layout(fig, axes)
    _maybe_save(fig, save_basepath=save_basepath, figure_profile=figure_profile)
    return fig


def _plot_duration_column(
    axes: np.ndarray,
    *,
    long_list: List[mne.Evoked],
    short_list: List[mne.Evoked],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> None:
    _plot_electrode_panel(
        long_list=long_list,
        short_list=short_list,
        ax=axes[0, 0],
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
    )
    _plot_electrode_panel(
        long_list=long_list,
        short_list=short_list,
        ax=axes[1, 0],
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
    )


def _plot_latency_column(
    axes: np.ndarray,
    *,
    fast_list: List[mne.Evoked],
    slow_list: List[mne.Evoked],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> None:
    _plot_electrode_panel(
        long_list=fast_list,
        short_list=slow_list,
        ax=axes[0, 1],
        electrode="Fz",
        label_1="fast",
        label_2="slow",
        color_1=LATENCY_COLOR_1,
        color_2=LATENCY_COLOR_2,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        y_large_label=None,
        title="Latency",
    )
    _plot_electrode_panel(
        long_list=fast_list,
        short_list=slow_list,
        ax=axes[1, 1],
        electrode="Pz",
        label_1="fast",
        label_2="slow",
        color_1=LATENCY_COLOR_1,
        color_2=LATENCY_COLOR_2,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        y_large_label=None,
        title=None,
    )


def _plot_electrode_panel(
    *,
    long_list: List[mne.Evoked],
    short_list: List[mne.Evoked],
    ax: plt.Axes,
    electrode: str,
    label_1: str,
    label_2: str,
    color_1: str,
    color_2: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    y_large_label: str | None,
    title: str | None,
) -> None:
    _plot_selection_electrode_time_course(
        long_list,
        short_list,
        ax,
        electrode=electrode,
        label_1=label_1,
        label_2=label_2,
        color_1=color_1,
        color_2=color_2,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        y_large_label=y_large_label,
        title=title,
        xlabel="Time (ms)",
        ylabel="Amplitude ($\\mu$V)",
    )


def _format_electrode_panel_layout(fig: plt.Figure, axes: np.ndarray) -> None:
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.10, top=0.92, wspace=0.25, hspace=0.25)


def plot_latency_erp_with_histograms(
    fast_list: List[mne.Evoked],
    slow_list: List[mne.Evoked],
    df: pd.DataFrame,
    ymax: float = 2000,
    *,
    figure_profile: str = "jneuro_2col",
    save_basepath: str | Path | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, WIDTH * 0.5))
    fig.suptitle("Latency effect and histogram")
    latency_df = df.copy()
    latency_df["latency"] = -latency_df["latency"] * 1e3
    _plot_latency_hist_panel(axes[0], fast_list=fast_list, slow_list=slow_list, latency_df=latency_df, ymax=ymax, electrode="Fz")
    _plot_latency_hist_panel(
        axes[1],
        fast_list=fast_list,
        slow_list=slow_list,
        latency_df=latency_df,
        ymax=ymax,
        electrode="Pz",
        clear_main_axis_ticks=True,
    )
    _maybe_save(fig, save_basepath=save_basepath, figure_profile=figure_profile)
    return fig


def _plot_latency_hist_panel(
    ax: plt.Axes,
    *,
    fast_list: List[mne.Evoked],
    slow_list: List[mne.Evoked],
    latency_df: pd.DataFrame,
    ymax: float,
    electrode: str,
    clear_main_axis_ticks: bool = False,
) -> None:
    _plot_selection_electrode_time_course(
        fast_list,
        slow_list,
        ax,
        electrode=electrode,
        label_1="fast",
        label_2="slow",
        color_1=LATENCY_COLOR_1,
        color_2=LATENCY_COLOR_2,
        xmin=-1500,
        xmax=500,
        ymin=-2.8,
        ymax=2.8,
        title=electrode,
        xlabel="Time (ms)",
        ylabel="Amplitude ($\\mu$V)",
        legend=False,
    )
    hist_ax = ax.twinx()
    sns.histplot(latency_df, x="latency", hue="condition", ax=hist_ax, palette=[LATENCY_COLOR_1, LATENCY_COLOR_2])
    ax.legend()
    legend = hist_ax.legend()
    legend.remove()
    hist_ax.set_ylim(0, ymax)
    hist_ax.set_yticks([])
    hist_ax.set_ylabel("")
    if clear_main_axis_ticks:
        ax.set_yticks([])
        ax.set_ylabel("")


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
