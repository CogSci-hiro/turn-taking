"""Decoding visualization components (private helpers)."""

from typing import List, Tuple

from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
import numpy as np
import scipy

from .._style import FONT_SIZE, TITLE_FONT_SIZE


def _plot_generalization(
    tmin: float,
    tmax: float,
    scores: np.ndarray,
    cluster_list: List[Tuple[np.ndarray, np.ndarray]],
    p_values: np.ndarray,
    ax: plt.Axes,
    p_threshold: float,
    y_axis: bool,
    lim: float = 0.05,
) -> AxesImage:
    """
    Plot temporal generalisation matrix (legacy-matching).

    Notes
    -----
    - Expects clusters in MNE out_type="indices" format: (train_idx, test_idx).
    - Draws 0-time crosshair and black cluster outlines via upsampled mask + contour.
    """

    n_times = scores.shape[-1]

    # Make mask
    mask = np.zeros((scores.shape[-1], scores.shape[-1]), dtype=bool)

    for cluster, p in zip(cluster_list, p_values):
        if float(p) >= p_threshold:
            continue

        # Accept both:
        # 1) tuple of (row_idx, col_idx)  [MNE out_type="indices"]
        # 2) (n_points, 2) int array      [point list]
        if isinstance(cluster, tuple) and len(cluster) == 2:
            row_idx, col_idx = cluster
            mask[np.asarray(row_idx, dtype=int), np.asarray(col_idx, dtype=int)] = True
            continue

        cluster_arr = np.asarray(cluster)
        if cluster_arr.ndim == 2 and cluster_arr.shape[1] == 2:
            row_idx = cluster_arr[:, 0].astype(int, copy=False)
            col_idx = cluster_arr[:, 1].astype(int, copy=False)
            mask[row_idx, col_idx] = True
            continue

        raise TypeError(
            "Unsupported cluster format. Expected (row_idx, col_idx) tuple or (n_points,2) array; "
            f"got type={type(cluster)} shape={getattr(cluster_arr, 'shape', None)}."
        )

    # legacy time axis: uses linspace(tmin, tmax) and multiplies by 1e3 for plotting
    times = np.linspace(tmin, tmax, n_times) * 1e3

    im = ax.imshow(
        scores.mean(axis=0).mean(axis=0),
        origin="lower",
        cmap="RdBu_r",
        extent=times[[0, -1, 0, -1]],
        vmin=0.5 - lim,
        vmax=0.5 + lim,
        aspect="auto",
    )

    # 0-time crosshair (this is the "horizontal line" you noticed)
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")

    # black outline around significant clusters
    if mask.any():
        big_mask = np.kron(mask.astype(float), np.ones((10, 10)))
        ax.contour(
            big_mask,
            levels=[0.5],
            colors=["k"],
            linewidths=[0.75],
            origin="lower",
            extent=times[[0, -1, 0, -1]],
            corner_mask=False,
            antialiased=False,
            zorder=10,
        )
    ax.set_xlabel("Testing Time (s)")
    if y_axis:
        ax.set_ylabel("Training Time (s)")
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


def _ci(data: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute confidence interval

    Parameters
    ----------
    data: np.ndarray
        1D, data along sample axis

    confidence: float
        confidence level

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        mean: mean time series
        lower: lower confidence interval
        upper: upper confidence interval
    """

    n = data.size
    mean = data.mean()
    se = scipy.stats.sem(data)

    height = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, mean - height, mean + height
