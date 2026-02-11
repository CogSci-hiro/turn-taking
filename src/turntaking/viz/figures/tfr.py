"""Time–frequency representation figures."""


from pathlib import Path
from typing import List, Tuple

import matplotlib as mpl
import mne
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import inset_locator

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




def plot_tfr_electrode_time_course(long_list: List[mne.Evoked], short_list: List[mne.Evoked],
                                   fast_list: List[mne.Evoked], slow_list: List[mne.Evoked],
                                   xmin: float = -1500, xmax: float = 500,
                                   ymin: float = -1.3, ymax: float = 1.6) -> plt.Figure:
    """
    Plot time course of TFR amplitudes for Fz and Pz for duration/latency comparison for a given frequency band

    Note: subject mean is subtracted before plotting

    Parameters
    ----------
    long_list
        list of subject evoked for long condition

    short_list
        list of subject evoked for short condition

    fast_list
        list of subject evoked for fast condition

    slow_list
        list of subject evoked for slow condition

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

    electrode_1 = "FC4"
    electrode_2 = "Pz"
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    # Duration FC6
    _plot_selection_electrode_time_course(long_list, short_list, axes[0, 0], electrode=electrode_1,
                                          label_1="long", label_2="short",
                                          color_1=DURATION_COLOR_1, color_2=DURATION_COLOR_2,
                                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, reverse=False, average=True,
                                          title="Duration", xlabel="Time (ms)",
                                          y_large_label=electrode_1,
                                          ylabel=f"Power ($\mu V^2$)")

    # Duration Pz
    _plot_selection_electrode_time_course(long_list, short_list, axes[1, 0], electrode=electrode_2,
                                          label_1="long", label_2="short",
                                          color_1=DURATION_COLOR_1, color_2=DURATION_COLOR_2,
                                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, reverse=False, average=True,
                                          title=None, xlabel="Time (ms)",
                                          y_large_label=electrode_2,
                                          ylabel=f"Power ($\mu V^2$)")

    # Latency FC6
    _plot_selection_electrode_time_course(fast_list, slow_list, axes[0, 1], electrode=electrode_1,
                                          label_1="fast", label_2="slow",
                                          color_1=LATENCY_COLOR_1, color_2=LATENCY_COLOR_2,
                                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, reverse=False, average=True,
                                          title="Latency", xlabel="Time (ms)", ylabel="Power ($\mu V^2$)")

    # Latency Pz
    _plot_selection_electrode_time_course(fast_list, slow_list, axes[1, 1], electrode=electrode_2,
                                          label_1="fast", label_2="slow",
                                          color_1=LATENCY_COLOR_1, color_2=LATENCY_COLOR_2,
                                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, reverse=False, average=True,
                                          title=None, xlabel="Time (ms)", ylabel="Power ($\mu V^2$)")

    return fig

################################################################
#                           DECODING                           #
################################################################


