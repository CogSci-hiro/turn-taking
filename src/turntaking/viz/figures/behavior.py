"""Behavioral figures."""


from math import floor

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .._style import (
    DURATION_COLOR_1,
    DURATION_COLOR_2,
    FONT_SIZE,
    LATENCY_COLOR_1,
    LATENCY_COLOR_2,
    TITLE_FONT_SIZE,
    WIDTH,
)

def plot_behavior(df: pd.DataFrame, n_bins: int = 100) -> plt.Figure:
    """
    Plot histogram of all durations and all latencies

    Parameters
    ----------
    df: pd.DataFrame
        'offset' file generated during ERP data generation

    n_bins: int
        number of bins

    Returns
    -------
    plt.Figure
        figure
    """

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, WIDTH * 0.5))

    # Crop a little
    min_val, max_val = -4, 4
    df = df[df["latency"] > min_val]
    df = df[df["latency"] < max_val]
    df = df[df["self_duration"] < 6]

    # Latency
    latency_median = df[(-1 < df["latency"]) & (df["latency"] < 1)]["latency"].median()
    _, bins, patches = axes[0].hist(df["latency"], n_bins)

    # Set colors
    upper, lower = 1., -1.
    length = max_val - min_val
    step = length / n_bins
    left_lim = floor((lower - min_val) / step)
    right_lim = n_bins - floor((max_val - upper) / step)

    for idx in range(left_lim):
        patches[idx].set_facecolor("lightgray")
    for idx in range(left_lim, right_lim):
        patches[idx].set_facecolor("gray")
    for idx in range(right_lim, n_bins):
        patches[idx].set_facecolor("lightgray")

    axes[0].vlines(x=latency_median, ymin=0, ymax=1600, colors="salmon", linestyles=":")
    axes[0].text(x=latency_median + 0.05, y=1500, s=f"{int(latency_median * 1e3)} ms", fontsize=FONT_SIZE)

    axes[0].set_title("Response latency", fontsize=TITLE_FONT_SIZE)
    axes[0].set_xlabel("Response latency (s)", fontsize=FONT_SIZE)
    axes[0].set_xlim(-4, 4)
    axes[0].set_xticks(np.arange(-4, 4.01, 1.0))
    axes[0].set_ylabel("Count", fontsize=FONT_SIZE)
    axes[0].tick_params(axis="both", which="major", labelsize=FONT_SIZE)

    # Duration
    duration_median = df[(-1 < df["latency"]) & (df["latency"] < 1)]["self_duration"].median()
    axes[1].hist(df["self_duration"], n_bins, facecolor="gray")

    axes[1].vlines(x=duration_median, ymin=0, ymax=1400, colors="salmon", linestyles=":")
    axes[1].text(x=duration_median + 0.05, y=1300, s=f"{int(duration_median * 1e3)} ms", fontsize=FONT_SIZE)

    axes[1].set_title("Response duration", fontsize=TITLE_FONT_SIZE)
    axes[1].set_xlabel("Response duration (s)", fontsize=FONT_SIZE)
    axes[1].set_ylabel("")
    axes[1].yaxis.tick_right()
    axes[1].tick_params(axis="both", which="major", labelsize=FONT_SIZE)

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()

    return fig




def plot_latencies(duration_df: pd.DataFrame, latency_df: pd.DataFrame) -> plt.Figure:
    """
    Histograms of latencies by conditions in both duration and latency comparisons

    Parameters
    ----------
    duration_df
        metadata for the duration comparison

    latency_df
        metadata for the latency comparison

    Returns
    -------
    plt.Figure
        figures
    """

    duration_palette = [DURATION_COLOR_1, DURATION_COLOR_2]
    latency_palette = [LATENCY_COLOR_1, LATENCY_COLOR_2]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sns.histplot(duration_df, x="latency", hue="condition", ax=axes[0], palette=duration_palette)
    sns.histplot(latency_df, x="latency", hue="condition", ax=axes[1], palette=latency_palette)

    axes[0].set_ylim(0, 700)
    axes[1].set_ylim(0, 700)

    axes[0].set_title("Duration comparison")
    axes[1].set_title("Latency comparison")

    axes[0].set_xlabel("Response latency (s)")
    axes[1].set_xlabel("Response latency (s)")

    axes[0].set_ylabel("Number of turns")
    axes[1].set_ylabel("")

    axes[1].set_yticks([])

    return fig




def plot_other_duration(duration_df: pd.DataFrame, latency_df: pd.DataFrame) -> plt.Figure:
    """
    Plot histograms of other participant IPU duration per condition

    Parameters
    ----------
    duration_df
        metadata for the duration comparison

    latency_df
        metadata for the latency comparison

    Returns
    -------
    plt.Figure
        figure
    """

    duration_palette = [DURATION_COLOR_1, DURATION_COLOR_2]
    latency_palette = [LATENCY_COLOR_1, LATENCY_COLOR_2]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sns.histplot(duration_df, x="other_duration", hue="condition", ax=axes[0], palette=duration_palette)
    axes[0].set_xlim(0, 8)
    axes[0].set_xlabel("Duration (s)")
    axes[0].set_title("Previous speech duration: duration contrast")

    sns.histplot(latency_df, x="other_duration", hue="condition", ax=axes[1], palette=latency_palette)
    axes[1].set_xlim(0, 8)
    axes[1].set_xlabel("Duration (s)")
    axes[1].set_title("Previous speech duration: latency contrast")

    return fig


################################################################
#                             ERP                              #
################################################################


