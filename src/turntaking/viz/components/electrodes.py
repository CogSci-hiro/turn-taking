"""Reusable plotting components.

The goal is to centralize shared plotting code (e.g., electrode time-courses)
so that figure functions remain small and consistent.
"""


from typing import List

import mne
import numpy as np
from matplotlib import pyplot as plt

from .._style import FONT_SIZE, TITLE_FONT_SIZE
from .._utils import _ci


def _plot_selection_electrode_time_course(data_1_list: List[mne.Evoked], data_2_list: List[mne.Evoked],
                                          ax: plt.axis, electrode: str,
                                          label_1: str, label_2: str, color_1: str, color_2: str,
                                          xmin: float, xmax: float, ymin: float, ymax: float,
                                          margin: float = 1.05, reverse: bool = True, average: bool = False,
                                          x_offset: float = -0.09, y_large_label: str | None = None,
                                          title: str | None = None,
                                          xlabel: str | None = None, ylabel: str | None = None,
                                          legend: bool = True) -> None:
    """
    Plot electrode time course for a particular electrode in two conditions

    Parameters
    ----------
    data_1_list: List[mne.Evoked]
        evoked data for the first condition

    data_2_list: List[mne.Evoked]
        evoked data for the second condition

    ax: plt.axis
        axis to plot to

    electrode: str
        electrode to plot

    label_1: str
        label for the first condition

    label_2: str
        label for the second condition

    color_1: str
        color of the first condition

    color_2: str
        color of the second condition

    xmin: float
        start value for the x axis

    xmax: float
        end value for the x axis

    ymin: float
        start value for the y axis

    ymax: float
        end value for the y axis

    margin: float
        margin for the y axis limits

    reverse: bool
        if true, negative is up (for ERP)

    average: bool
        if true, subtract subject mean (for TF)

    x_offset: float
        text location offset (for the electrode names)

    y_large_label: str | None
        if provided add label to the y axis (for the electrode name)

    title: str | None
        title

    xlabel: str | None
        x axis label

    ylabel: str | None
        y axis label

    legend: bool
        if false, disable legends

    Returns
    -------
    plt.Figure
        figure
    """

    # Time (ms)
    times = data_1_list[0].times * 1e3

    # Horizontal and vertical axes
    ax.hlines(0.0, times[0], times[-1], color="gray")
    ax.vlines(0.0, ymin * margin, ymax * margin, color="gray")

    # Make dummy raw
    raw = _get_dummy_raw()

    # Pick data
    data_1 = _get_targets(data_1_list, [electrode])
    data_2 = _get_targets(data_2_list, [electrode])

    # Remove subject mean (for TF)
    if average:
        mean = (data_1.mean(axis=-1).mean(axis=-1) + data_2.mean(axis=-1).mean(axis=-1)) / 2
        data_1 -= mean.reshape(-1, 1, 1)
        data_2 -= mean.reshape(-1, 1, 1)

    # Compute mean and confidence boundaries
    data_1_mean, data_1_lower, data_1_upper = np.apply_along_axis(_ci, 0, data_1.mean(axis=1) * 1e6)  # in uV
    data_2_mean, data_2_lower, data_2_upper = np.apply_along_axis(_ci, 0, data_2.mean(axis=1) * 1e6)  # in uV

    # Plot the mean
    ax.plot(times, data_1_mean, label=label_1, color=color_1)
    ax.plot(times, data_2_mean, label=label_2, color=color_2)

    # Fill the confidence interval
    ax.fill_between(times, data_1_lower, data_1_upper, color=color_1, alpha=.2)
    ax.fill_between(times, data_2_lower, data_2_upper, color=color_2, alpha=.2)

    # Plot picks
    ax_ins = inset_locator.inset_axes(ax, width="20%", height="30%", loc=3)
    raw.copy().pick(electrode).plot_sensors(axes=ax_ins, ch_type="eeg")

    # Set limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Positive is down
    if reverse:
        ax.yaxis.set_inverted(True)

    # Labels
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE)

    if y_large_label is not None:
        points = ax.get_position()._points  # noqa
        x0 = points[0, 0]
        y0 = points[0, 1]
        y1 = points[1, 1]
        ax.text(x0 + x_offset, (y0 + y1) / 2, y_large_label, fontsize=TITLE_FONT_SIZE,
                transform=plt.gcf().transFigure, rotation=90)

    # Ticks
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)

    if legend:
        ax.legend(loc="upper left", prop={"size": FONT_SIZE})


