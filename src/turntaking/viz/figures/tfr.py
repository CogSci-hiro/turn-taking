"""Time–frequency representation figures."""


from pathlib import Path
from typing import List, Tuple

import matplotlib as mpl
import mne
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import inset_locator

from .erp import plot_stat_topomaps_grid

from .._style import (
    FACE_COLOR,
    FONT_SIZE,
    MARKER_SIZE,
    P_THRESHOLD,
    SMALLER_MARKER_SIZE,
    TITLE_FONT_SIZE,
)
from .._utils import _get_dummy_raw, _get_mask, _get_targets
from ..components.electrodes import _plot_selection_electrode_time_course

def plot_tfr_topo(duration_alpha_t: np.ndarray, duration_beta_t: np.ndarray,
                  latency_alpha_t: np.ndarray, latency_beta_t: np.ndarray,
                  duration_alpha_p: np.ndarray, duration_beta_p: np.ndarray,
                  latency_alpha_p: np.ndarray, latency_beta_p: np.ndarray,
                  duration_alpha_cluster: List[Tuple], duration_beta_cluster: List[Tuple],
                  latency_alpha_cluster: List[Tuple], latency_beta_cluster: List[Tuple],
                  info: mne.Info, data_tmin: float, tmin: float, tmax: float, n_topo: int,
                  p_threshold: float = 0.01) -> plt.Figure:
    """
    Plot fours rows of topography maps, one for each time step (specified by 'tmin', 'tmax' and 'n_topo')
    the first row is duration alpha band comparison,
    the second row is duration beta band comparison,
    the third row is latency alpha band comparison,
    the fourth row is latency beta band comparison

    Parameters
    ----------
    duration_alpha_t: np.ndarray
        t value for alpha band duration comparison

    duration_beta_t: np.ndarray
        t value for beta band duration comparison

    latency_alpha_t: np.ndarray
        t value for alpha band latency comparison

    latency_beta_t: np.ndarray
        t value for beta band latency comparison

    duration_alpha_p: np.ndarray
        p value for alpha band duration comparison

    duration_beta_p: np.ndarray
        p value for beta band duration comparison

    latency_alpha_p: np.ndarray
        p value for alpha band latency comparison

    latency_beta_p: np.ndarray
        p value for beta band latency comparison

    duration_alpha_cluster: List[Tuple]
        clusters for alpha band duration comparison

    duration_beta_cluster: List[Tuple]
        clusters for beta band duration comparison

    latency_alpha_cluster: List[Tuple]
        clusters for alpha band latency comparison

    latency_beta_cluster: List[Tuple]
        clusters for beta band latency comparison

    info: mne.Info
        info objects for evoked object

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
    duration_alpha_mask = _get_mask(duration_alpha_t, duration_alpha_p, duration_alpha_cluster, p_threshold)
    duration_beta_mask = _get_mask(duration_beta_t, duration_beta_p, duration_beta_cluster, p_threshold)
    latency_alpha_mask = _get_mask(latency_alpha_t, latency_alpha_p, latency_alpha_cluster, p_threshold)
    latency_beta_mask = _get_mask(latency_beta_t, latency_beta_p, latency_beta_cluster, p_threshold)

    # Get the largest absolute t value as the limit
    lim_val = max(duration_alpha_t.max(), duration_beta_t.max(), latency_alpha_t.max(), latency_beta_t.max(),
                  abs(duration_alpha_t.min()), abs(duration_beta_t.min()), abs(latency_alpha_t.min()),
                  abs(latency_beta_t.min()))

    # Convert to MNE evoked
    duration_alpha_t = mne.EvokedArray(duration_alpha_t.T * 1e-6, info, tmin=data_tmin)  # expecting uV
    duration_beta_t = mne.EvokedArray(duration_beta_t.T * 1e-6, info, tmin=data_tmin)  # expecting uV
    latency_alpha_t = mne.EvokedArray(latency_alpha_t.T * 1e-6, info, tmin=data_tmin)  # expecting uV
    latency_beta_t = mne.EvokedArray(latency_beta_t.T * 1e-6, info, tmin=data_tmin)  # expecting uV

    # Timesteps, include 0.0
    timesteps = np.linspace(tmin, tmax, n_topo)

    # Plot topographies
    fig, axes = plt.subplots(4, timesteps.size, figsize=(15, 6))

    mask_params = {"marker": "o", "markerfacecolor": FACE_COLOR, "markeredgecolor": "k",
                   "linewidth": 0, "markersize": SMALLER_MARKER_SIZE}
    duration_alpha_t.plot_topomap(axes=axes[0, :], times=timesteps, colorbar=False, show=False,
                                  mask=duration_alpha_mask.T,
                                  vlim=(-lim_val, lim_val), time_unit="ms", mask_params=mask_params)
    duration_beta_t.plot_topomap(axes=axes[1, :], times=timesteps, colorbar=False, show=False,
                                 mask=duration_beta_mask.T,
                                 vlim=(-lim_val, lim_val), time_unit="ms", time_format="", mask_params=mask_params)
    latency_alpha_t.plot_topomap(axes=axes[2, :], times=timesteps, colorbar=False, show=False,
                                 mask=latency_alpha_mask.T,
                                 vlim=(-lim_val, lim_val), time_unit="ms", time_format="", mask_params=mask_params)
    latency_beta_t.plot_topomap(axes=axes[3, :], times=timesteps, colorbar=False, show=False,
                                mask=latency_beta_mask.T,
                                vlim=(-lim_val, lim_val), time_unit="ms", time_format="", mask_params=mask_params)

    # Left label
    axes[0, 0].set_ylabel("Alpha", fontsize=FONT_SIZE)
    axes[1, 0].set_ylabel("Beta", fontsize=FONT_SIZE)
    axes[2, 0].set_ylabel("Alpha", fontsize=FONT_SIZE)
    axes[3, 0].set_ylabel("Beta", fontsize=FONT_SIZE)

    # Condition labels
    points = axes[0, 0].get_position()._points  # noqa
    x0 = points[0, 0]
    y0 = points[0, 1]
    axes[0, 0].text(x0 - 0.11, y0 - 0.12, "Duration", fontsize=FONT_SIZE,
                    transform=plt.gcf().transFigure, rotation=90)

    points = axes[2, 0].get_position()._points  # noqa
    x0 = points[0, 0]
    y0 = points[0, 1]
    axes[2, 0].text(x0 - 0.11, y0 - 0.12, "Latency", fontsize=FONT_SIZE,
                    transform=plt.gcf().transFigure, rotation=90)

    # Colorbar
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0.05, wspace=0.0)
    cbar_ax = fig.add_axes((0.97, 0.15, 0.01, 0.7))  # (left, bottom, width, height)
    norm = mpl.colors.Normalize(vmin=-lim_val, vmax=lim_val)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

    return fig



# =============================================================================
#                     ########################################
#                     #          TFR TOPO WRAPPERS           #
#                     ########################################
# =============================================================================
def plot_tfr_topo_alpha_duration(
    t_values: np.ndarray,
    p_values: np.ndarray,
    clusters: List[Tuple],
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
    mask = _get_mask(t_values, p_values, clusters, p_threshold)
    lim_val = float(np.max(np.abs(t_values)))
    return plot_stat_topomaps_grid(
        stat=t_values,
        mask=mask,
        info=info,
        data_tmin=data_tmin,
        tmin=tmin,
        tmax=tmax,
        step_ms=step_ms,
        title="TFR topographies (Alpha, Duration)",
        lim_val=lim_val,
        max_cols=max_cols,
        figure_profile=figure_profile,
        save_basepath=save_basepath,
    )


def plot_tfr_topo_alpha_latency(
    t_values: np.ndarray,
    p_values: np.ndarray,
    clusters: List[Tuple],
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
    mask = _get_mask(t_values, p_values, clusters, p_threshold)
    lim_val = float(np.max(np.abs(t_values)))
    return plot_stat_topomaps_grid(
        stat=t_values,
        mask=mask,
        info=info,
        data_tmin=data_tmin,
        tmin=tmin,
        tmax=tmax,
        step_ms=step_ms,
        title="TFR topographies (Alpha, Latency)",
        lim_val=lim_val,
        max_cols=max_cols,
        figure_profile=figure_profile,
        save_basepath=save_basepath,
    )


def plot_tfr_topo_beta_duration(
    t_values: np.ndarray,
    p_values: np.ndarray,
    clusters: List[Tuple],
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
    mask = _get_mask(t_values, p_values, clusters, p_threshold)
    lim_val = float(np.max(np.abs(t_values)))
    return plot_stat_topomaps_grid(
        stat=t_values,
        mask=mask,
        info=info,
        data_tmin=data_tmin,
        tmin=tmin,
        tmax=tmax,
        step_ms=step_ms,
        title="TFR topographies (Beta, Duration)",
        lim_val=lim_val,
        max_cols=max_cols,
        figure_profile=figure_profile,
        save_basepath=save_basepath,
    )


def plot_tfr_topo_beta_latency(
    t_values: np.ndarray,
    p_values: np.ndarray,
    clusters: List[Tuple],
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
    mask = _get_mask(t_values, p_values, clusters, p_threshold)
    lim_val = float(np.max(np.abs(t_values)))
    return plot_stat_topomaps_grid(
        stat=t_values,
        mask=mask,
        info=info,
        data_tmin=data_tmin,
        tmin=tmin,
        tmax=tmax,
        step_ms=step_ms,
        title="TFR topographies (Beta, Latency)",
        lim_val=lim_val,
        max_cols=max_cols,
        figure_profile=figure_profile,
        save_basepath=save_basepath,
    )


