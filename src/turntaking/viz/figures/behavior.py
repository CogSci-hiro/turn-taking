"""Behavioral figures."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Optional

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
    save_figure,  # assumes this exists in your _style.py (it does for ERP timecourse usage)
)

# ======================================================================================================================
# Constants
# ======================================================================================================================

DEFAULT_N_BINS: int = 100
MAX_LATENCY_S: float = 4.0
MAX_SELF_DURATION_S: float = 6.0
INCLUSION_LATENCY_S: float = 1.0

JOINT_SCATTER_ALPHA: float = 0.25
JOINT_SCATTER_SIZE: float = 8.0


# ======================================================================================================================
# Column resolution
# ======================================================================================================================

@dataclass(frozen=True)
class BehaviorColumns:
    """
    Column mapping for behavior offsets files.

    Attributes
    ----------
    latency
        Latency column (seconds).
    self_duration
        Self response duration column (seconds).
    other_duration
        Previous/partner speech duration column (seconds).
    condition
        Condition label column (e.g., long/short or fast/slow). Optional.
    """
    latency: str = "latency"
    self_duration: str = "self_duration"
    other_duration: str = "other_duration"
    condition: str = "condition"


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")


def _crop_behavior_df(df: pd.DataFrame, cols: BehaviorColumns) -> pd.DataFrame:
    """
    Apply the same “publication crop” that your original Fig1 used.

    Notes
    -----
    - Does not mutate input DataFrame.
    - Keeps negative latencies (overlap).
    """
    _require_columns(df, [cols.latency, cols.self_duration])

    cropped = df.copy()
    cropped = cropped[cropped[cols.latency] > -MAX_LATENCY_S]
    cropped = cropped[cropped[cols.latency] < MAX_LATENCY_S]
    cropped = cropped[cropped[cols.self_duration] < MAX_SELF_DURATION_S]
    return cropped


def _median_in_inclusion_window(df: pd.DataFrame, cols: BehaviorColumns, value_col: str) -> float:
    """
    Median computed within the canonical inclusion latency window [-1, 1] s.
    """
    window = df[(-INCLUSION_LATENCY_S < df[cols.latency]) & (df[cols.latency] < INCLUSION_LATENCY_S)]
    if len(window) == 0:
        raise ValueError("No rows remain inside inclusion window for median computation.")
    return float(window[value_col].median())


# ======================================================================================================================
# Main Fig 1: overall latency + duration histograms
# ======================================================================================================================

def plot_behavior(df: pd.DataFrame, n_bins: int = DEFAULT_N_BINS) -> plt.Figure:
    """
    Plot histogram of all durations and all latencies.

    Parameters
    ----------
    df
        Offsets DataFrame (e.g., offsets.csv generated during ERP data generation).
        Must contain columns: 'latency', 'self_duration'.
    n_bins
        Number of histogram bins.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    cols = BehaviorColumns()
    df_cropped = _crop_behavior_df(df, cols)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, WIDTH * 0.5))

    # Latency
    latency_median = _median_in_inclusion_window(df_cropped, cols, cols.latency)
    _, _, patches = axes[0].hist(df_cropped[cols.latency], n_bins)

    # Set colors (keeps your original styling behavior)
    min_val, max_val = -MAX_LATENCY_S, MAX_LATENCY_S
    upper, lower = INCLUSION_LATENCY_S, -INCLUSION_LATENCY_S
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
    axes[0].text(
        x=latency_median + 0.05,
        y=1500,
        s=f"{int(latency_median * 1e3)} ms",
        fontsize=FONT_SIZE,
    )

    axes[0].set_title("Response latency", fontsize=TITLE_FONT_SIZE)
    axes[0].set_xlabel("Response latency (s)", fontsize=FONT_SIZE)
    axes[0].set_xlim(-MAX_LATENCY_S, MAX_LATENCY_S)
    axes[0].set_xticks(np.arange(-MAX_LATENCY_S, MAX_LATENCY_S + 0.01, 1.0))
    axes[0].set_ylabel("Count", fontsize=FONT_SIZE)
    axes[0].tick_params(axis="both", which="major", labelsize=FONT_SIZE)

    # Duration
    duration_median = _median_in_inclusion_window(df_cropped, cols, cols.self_duration)
    axes[1].hist(df_cropped[cols.self_duration], n_bins, facecolor="gray")

    axes[1].vlines(x=duration_median, ymin=0, ymax=1400, colors="salmon", linestyles=":")
    axes[1].text(
        x=duration_median + 0.05,
        y=1300,
        s=f"{int(duration_median * 1e3)} ms",
        fontsize=FONT_SIZE,
    )

    axes[1].set_title("Response duration", fontsize=TITLE_FONT_SIZE)
    axes[1].set_xlabel("Response duration (s)", fontsize=FONT_SIZE)
    axes[1].set_ylabel("")
    axes[1].yaxis.tick_right()
    axes[1].tick_params(axis="both", which="major", labelsize=FONT_SIZE)

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    return fig


# ======================================================================================================================
# Supplementary: S1/S2
# ======================================================================================================================

def plot_response_duration_hist(duration_df: pd.DataFrame, latency_df: pd.DataFrame, n_bins: int = DEFAULT_N_BINS) -> plt.Figure:
    """
    S1: Response duration histograms split by condition (duration contrast + latency contrast).

    Parameters
    ----------
    duration_df
        Offsets DataFrame for duration contrast (should have condition: long/short).
    latency_df
        Offsets DataFrame for latency contrast (should have condition: fast/slow).
    n_bins
        Number of bins.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with 1x2 panels.
    """
    cols = BehaviorColumns()
    _require_columns(duration_df, [cols.self_duration, cols.condition])
    _require_columns(latency_df, [cols.self_duration, cols.condition])

    duration_palette = [DURATION_COLOR_1, DURATION_COLOR_2]
    latency_palette = [LATENCY_COLOR_1, LATENCY_COLOR_2]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sns.histplot(duration_df, x=cols.self_duration, hue=cols.condition, ax=axes[0], bins=n_bins, palette=duration_palette)
    sns.histplot(latency_df, x=cols.self_duration, hue=cols.condition, ax=axes[1], bins=n_bins, palette=latency_palette)

    axes[0].set_title("Response duration: duration contrast")
    axes[1].set_title("Response duration: latency contrast")

    axes[0].set_xlabel("Duration (s)")
    axes[1].set_xlabel("Duration (s)")

    axes[0].set_ylabel("Number of turns")
    axes[1].set_ylabel("")
    axes[1].set_yticks([])

    return fig


def plot_other_duration(duration_df: pd.DataFrame, latency_df: pd.DataFrame) -> plt.Figure:
    """
    S2: Previous speech duration histograms by condition for both contrasts.

    Parameters
    ----------
    duration_df
        Metadata for the duration comparison.
    latency_df
        Metadata for the latency comparison.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with 1x2 panels.
    """
    cols = BehaviorColumns()
    _require_columns(duration_df, [cols.other_duration, cols.condition])
    _require_columns(latency_df, [cols.other_duration, cols.condition])

    duration_palette = [DURATION_COLOR_1, DURATION_COLOR_2]
    latency_palette = [LATENCY_COLOR_1, LATENCY_COLOR_2]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sns.histplot(duration_df, x=cols.other_duration, hue=cols.condition, ax=axes[0], palette=duration_palette)
    axes[0].set_xlim(0, 8)
    axes[0].set_xlabel("Duration (s)")
    axes[0].set_title("Previous speech duration: duration contrast")

    sns.histplot(latency_df, x=cols.other_duration, hue=cols.condition, ax=axes[1], palette=latency_palette)
    axes[1].set_xlim(0, 8)
    axes[1].set_xlabel("Duration (s)")
    axes[1].set_title("Previous speech duration: latency contrast")

    return fig


# ======================================================================================================================
# Supplementary: S3 joint plots (4 separate figures)
# ======================================================================================================================

def plot_joint(df: pd.DataFrame, title: str, color: str) -> plt.Figure:
    """
    Joint plot: other_duration vs self_duration with marginal histograms.

    Parameters
    ----------
    df
        Offsets DataFrame filtered to one condition (e.g., long only).
    title
        Figure title.
    color
        Color for points/hist.

    Returns
    -------
    matplotlib.figure.Figure
        Figure.
    """
    cols = BehaviorColumns()
    _require_columns(df, [cols.other_duration, cols.self_duration])

    # Seaborn JointGrid keeps it compact and consistent for publication.
    g = sns.JointGrid(data=df, x=cols.other_duration, y=cols.self_duration, height=6, ratio=4, space=0.1)
    g.plot_joint(
        sns.scatterplot,
        alpha=JOINT_SCATTER_ALPHA,
        s=JOINT_SCATTER_SIZE,
        color=color,
        edgecolor=None,
    )
    g.plot_marginals(sns.histplot, color=color, bins=DEFAULT_N_BINS)

    g.ax_joint.set_xlabel("Previous speech duration (s)")
    g.ax_joint.set_ylabel("Response duration (s)")
    g.ax_joint.set_title(title)

    return g.figure


def _filter_condition(df: pd.DataFrame, condition_value: str) -> pd.DataFrame:
    cols = BehaviorColumns()
    _require_columns(df, [cols.condition])
    return df[df[cols.condition] == condition_value].copy()


# ======================================================================================================================
# Pipeline entrypoint: produce all behavior figures
# ======================================================================================================================

def make_behavior_figures(
    duration_offsets_csv: Path,
    latency_offsets_csv: Path,
    out_main: Path,
    out_s1: Path,
    out_s2: Path,
    out_s3_long: Path,
    out_s3_short: Path,
    out_s3_fast: Path,
    out_s3_slow: Path,
    n_bins: int = DEFAULT_N_BINS,
    figure_profile: str = "jneuro_2col",
) -> None:
    """
    Generate all behavior figures (Fig1 + S1–S3) from offsets CSVs.

    Parameters
    ----------
    duration_offsets_csv
        CSV path for duration contrast offsets (long/short).
    latency_offsets_csv
        CSV path for latency contrast offsets (fast/slow).
    out_main
        Output path for main behavior figure (Fig1).
    out_s1
        Output path for S1 response duration histogram.
    out_s2
        Output path for S2 previous speech duration histogram.
    out_s3_long
        Output path for S3 (long) joint plot.
    out_s3_short
        Output path for S3 (short) joint plot.
    out_s3_fast
        Output path for S3 (fast) joint plot.
    out_s3_slow
        Output path for S3 (slow) joint plot.
    n_bins
        Histogram bins.
    figure_profile
        Figure style profile routed through save_figure().

    Returns
    -------
    None
    """
    duration_df = pd.read_csv(duration_offsets_csv)
    latency_df = pd.read_csv(latency_offsets_csv)

    # Fig1: use duration_df as the canonical “overall distribution” source (avoids duplicating turns).
    fig1 = plot_behavior(duration_df, n_bins=n_bins)
    save_figure(fig1, out_main, profile_name=figure_profile)
    plt.close(fig1)

    # S1: response duration hist by condition for both contrasts
    fig_s1 = plot_response_duration_hist(duration_df=duration_df, latency_df=latency_df, n_bins=n_bins)
    save_figure(fig_s1, out_s1, profile_name=figure_profile)
    plt.close(fig_s1)

    # S2: previous speech duration hist
    fig_s2 = plot_other_duration(duration_df=duration_df, latency_df=latency_df)
    save_figure(fig_s2, out_s2, profile_name=figure_profile)
    plt.close(fig_s2)

    # S3: four joint plots
    long_df = _filter_condition(duration_df, "long")
    short_df = _filter_condition(duration_df, "short")
    fast_df = _filter_condition(latency_df, "fast")
    slow_df = _filter_condition(latency_df, "slow")

    fig_long = plot_joint(long_df, title="Long", color=DURATION_COLOR_1)
    save_figure(fig_long, out_s3_long, profile_name=figure_profile)
    plt.close(fig_long)

    fig_short = plot_joint(short_df, title="Short", color=DURATION_COLOR_2)
    save_figure(fig_short, out_s3_short, profile_name=figure_profile)
    plt.close(fig_short)

    fig_fast = plot_joint(fast_df, title="Fast", color=LATENCY_COLOR_1)
    save_figure(fig_fast, out_s3_fast, profile_name=figure_profile)
    plt.close(fig_fast)

    fig_slow = plot_joint(slow_df, title="Slow", color=LATENCY_COLOR_2)
    save_figure(fig_slow, out_s3_slow, profile_name=figure_profile)
    plt.close(fig_slow)
