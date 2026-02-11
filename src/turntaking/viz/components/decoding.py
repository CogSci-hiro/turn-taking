"""Decoding visualization components (private helpers)."""

from typing import List

from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
import numpy as np

from .._style import FONT_SIZE, TITLE_FONT_SIZE


def _plot_generalization(tmin: float, tmax: float, scores: np.ndarray,
                         cluster_list: List[np.ndarray], p_values: np.ndarray, ax: plt.axis, p_threshold: float,
                         y_axis: bool, lim: float = 0.04) -> AxesImage:
    """
    Plot temporal generalisation matrix

    Parameters
    ----------
    tmin: float
        start of the time window

    tmax: float
        end of the time window

    scores: np.ndarray

    cluster_list: List[np.ndarray]

    p_values: np.ndarray

    ax: plt.axis
        axis

    p_threshold: float
        significance threshold

    y_axis: bool

    lim: float

    Returns
    -------
    AxesImage
        AxesImage returned for colorbar
    """

    # Make mask
    mask = np.zeros((scores.shape[-1], scores.shape[-1])).astype(bool)
    for cluster, p in zip(cluster_list, p_values):
        if p < p_threshold:
            mask[cluster] = True

    # Times
    times = np.linspace(tmin, tmax, scores.shape[-1]) * 1e3

    # Plot image
    im = ax.imshow(scores.mean(axis=0).mean(axis=0),
                   origin="lower",
                   cmap="RdBu_r",
                   extent=times[[0, -1, 0, -1]],
                   vmin=0.5 - lim,
                   vmax=0.5 + lim)

    # Axes
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")

    # Mask
    big_mask = np.kron(mask, np.ones((10, 10)))
    ax.contour(big_mask,
               colors=["k"],
               extent=times[[0, -1, 0, -1]],
               linewidths=[0.75],
               corner_mask=False,
               antialiased=False,
               levels=[0.0])

    # Labels
    ax.set_xlabel("Testing Time (s)", fontsize=FONT_SIZE)
    if y_axis:
        ax.set_ylabel("Training Time (s)", fontsize=FONT_SIZE)
    else:
        ax.set_yticks([])

    return im




def _plot_diagonal(tmin: float, tmax: float, scores: np.ndarray, ax: plt.axis,
                   title: str, y_axis: bool, color: str, ymin: float = 0.45, ymax: float = 0.65) -> None:
    """
    Plot the diagonal of the temporal generalisation decoding results

    Parameters
    ----------
    tmin: float
        start of the epoch

    tmax: float
        end of the epoch

    scores: np.ndarray
        decoding score, (n_subjects, n_splits, n_times, n_times)

    ax: plt.axis
        axis to plot to

    title: str
        title

    y_axis: bool
        if true, add y ticks

    color: str
        color of the line plot

    ymin: float

    ymax: float

    Returns
    -------
    None
    """

    times = np.linspace(tmin, tmax, scores.shape[-1]) * 1e3

    diagonal = np.diagonal(scores, axis1=2, axis2=3).mean(axis=1)

    # Compute mean and confidence boundaries
    mean, lower, upper = np.apply_along_axis(_ci, 0, diagonal)

    # Plot the mean
    ax.plot(times, mean, color=color)

    # Fill the confidence interval
    ax.fill_between(times, lower, upper, color=color, alpha=.1)

    # Chance level
    ax.axhline(0.5, color="gray")

    # Limits
    ax.set_ylim(ymin, ymax)

    # Title
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    if not y_axis:
        ax.set_yticks([])


