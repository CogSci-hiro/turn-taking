"""Decoding figures (diagonal + temporal generalization)."""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

from .._style import (
    DURATION_COLOR_1,
    LATENCY_COLOR_1,
    WIDTH,
)
from ..components.decoding import _plot_diagonal, _plot_generalization


ClusterLike = Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]


def _as_generalization_clusters(
    clusters: Sequence[ClusterLike],
) -> List[np.ndarray]:
    """
    Normalize cluster representations to the legacy format expected by _plot_generalization.

    Supported inputs
    ----------------
    - np.ndarray
        Legacy cluster representation (already in the expected format).
    - (train_idx, test_idx)
        MNE out_type="indices" representation for 2D TG clusters.

    Returns
    -------
    clusters_out
        List[np.ndarray] where each item is an (n_points, 2) int array of (train_i, test_i).
    """
    out: List[np.ndarray] = []

    for cl in clusters:
        if isinstance(cl, np.ndarray):
            # Assume legacy already in expected format.
            out.append(cl)
            continue

        train_idx, test_idx = cl
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        if train_idx.shape != test_idx.shape:
            raise ValueError(
                "Cluster index arrays must have the same shape. "
                f"Got train_idx.shape={train_idx.shape}, test_idx.shape={test_idx.shape}."
            )

        out.append(np.column_stack([train_idx, test_idx]))

    return out


def plot_decoding(
    tmin: float,
    tmax: float,
    duration_scores: np.ndarray,
    latency_scores: np.ndarray,
    duration_clusters: Sequence[ClusterLike],
    latency_clusters: Sequence[ClusterLike],
    duration_p: np.ndarray,
    latency_p: np.ndarray,
    p_threshold: float,
    ymax: float = 0.65,
) -> plt.Figure:
    """
    Plot decoding results.

    On the left is the duration comparison, on the right is the latency comparison.
    The top row is diagonal decoding, the bottom row is temporal generalisation.
    """
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(WIDTH, WIDTH * 0.7),
        gridspec_kw={"height_ratios": [1, 4], "wspace": 0.1, "hspace": 0.1},
    )

    duration_clusters_ = _as_generalization_clusters(duration_clusters)
    latency_clusters_ = _as_generalization_clusters(latency_clusters)

    # Duration comparison
    _plot_diagonal(
        tmin,
        tmax,
        duration_scores,
        axes[0, 0],
        title="Duration decoding",
        color=DURATION_COLOR_1,
        y_axis=True,
        ymax=ymax,
    )
    _plot_generalization(
        tmin,
        tmax,
        duration_scores,
        duration_clusters_,
        duration_p,
        axes[1, 0],
        p_threshold,
        y_axis=True,
    )

    # Latency comparison
    _plot_diagonal(
        tmin,
        tmax,
        latency_scores,
        axes[0, 1],
        title="Latency decoding",
        color=LATENCY_COLOR_1,
        y_axis=False,
        ymax=ymax,
    )
    im = _plot_generalization(
        tmin,
        tmax,
        latency_scores,
        latency_clusters_,
        latency_p,
        axes[1, 1],
        p_threshold,
        y_axis=False,
    )

    # Colorbar
    cbar = fig.add_axes([0.93, 0.1, 0.02, 0.8])  # noqa
    fig.colorbar(im, cax=cbar)
    cbar.set_label("AUC")

    fig.tight_layout()
    return fig
