"""Decoding figures (diagonal + temporal generalization)."""

from __future__ import annotations

from typing import List

import numpy as np
from matplotlib import pyplot as plt

from .._style import (
    DURATION_COLOR_1,
    LATENCY_COLOR_1,
    WIDTH,
)
from ..components.decoding import _plot_diagonal, _plot_generalization

def plot_decoding(tmin: float, tmax: float,
                  duration_scores: np.ndarray, latency_scores: np.ndarray,
                  duration_clusters: List[np.ndarray], latency_clusters: List[np.ndarray],
                  duration_p: np.ndarray, latency_p: np.ndarray,
                  p_threshold: float, ymax: float = 0.65) -> plt.Figure:
    """
    Plot decoding results

    On the left is the duration comparison, on the right is the latency comparison

    the top row is diagonal decoding, the bottom row is temporal generalisation
    Parameters
    ----------
    tmin: float
        start of the epoch in seconds

    tmax: float
        end of the epoch in seconds

    duration_scores: np.ndarray

    latency_scores: np.ndarray

    duration_clusters: List[np.ndarray]

    latency_clusters: List[np.ndarray]

    duration_p: np.ndarray

    latency_p: np.ndarray

    p_threshold: float
        significance threshold

    ymax: float

    Returns
    -------
    plt.Figure
        figure
    """

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, WIDTH * 0.7),
                             gridspec_kw={"height_ratios": [1, 4], "wspace": 0.1, "hspace": 0.1})

    # Duration comparison
    _plot_diagonal(tmin, tmax, duration_scores, axes[0, 0], title="Duration decoding",
                   color=DURATION_COLOR_1, y_axis=True, ymax=ymax)
    _plot_generalization(tmin, tmax, duration_scores, duration_clusters, duration_p, axes[1, 0],
                         p_threshold, y_axis=True)

    # Latency comparison
    _plot_diagonal(tmin, tmax, latency_scores, axes[0, 1], title="Latency decoding",
                   color=LATENCY_COLOR_1, y_axis=False, ymax=ymax)
    im = _plot_generalization(tmin, tmax, latency_scores, latency_clusters, latency_p, axes[1, 1],
                              p_threshold, y_axis=False)

    # Colorbar
    cbar = fig.add_axes([0.93, 0.1, 0.02, 0.8])  # noqa
    fig.colorbar(im, cax=cbar)
    cbar.set_label("AUC")

    fig.tight_layout()

    return fig


################################################################
#                        HELPER METHODS                        #
################################################################


