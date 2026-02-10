# src/turntaking/stats/cluster_test.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import mne
from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_1samp_test

try:
    # Available in modern MNE
    from mne.stats import combine_adjacency
except Exception:  # pragma: no cover
    combine_adjacency = None  # type: ignore[assignment]


ClusterKind = Literal["erp", "tfr"]


@dataclass(frozen=True)
class ClusterTestParams:
    """Parameters controlling the cluster permutation test."""
    n_permutations: int = 1024
    threshold: float | None = None
    tail: int = 0  # 0: two-sided, 1: upper tail, -1: lower tail
    alpha: float = 0.05
    seed: int = 0
    n_jobs: int = 1
    ch_type: Literal["eeg", "mag", "grad"] = "eeg"


@dataclass(frozen=True)
class ClusterTestResult:
    """Outputs of a spatio-temporal (or spatio-spectro-temporal) cluster test."""
    t_values: np.ndarray
    clusters: list[tuple[np.ndarray, ...]]
    p_values: np.ndarray
    h0: np.ndarray
    metadata: dict[str, Any]


def _channel_adjacency(info: mne.Info, ch_type: str) -> np.ndarray:
    adj, _ = find_ch_adjacency(info, ch_type=ch_type)
    return adj


def run_cluster_1samp_erp(
    X: np.ndarray,
    *,
    info: mne.Info,
    params: ClusterTestParams,
) -> ClusterTestResult:
    """
    ERP cluster test.

    Parameters
    ----------
    X
        Shape (n_subjects, n_times, n_channels). This matches your old stats layout.
    info
        MNE Info providing channel layout for adjacency.
    """
    if X.ndim != 3:
        raise ValueError(f"ERP X must be (N,T,C). Got {X.shape}")

    ch_adj = _channel_adjacency(info, params.ch_type)

    t_vals, clusters, p_vals, h0 = spatio_temporal_cluster_1samp_test(
        X,
        adjacency=ch_adj,
        n_permutations=int(params.n_permutations),
        threshold=params.threshold,
        tail=int(params.tail),
        n_jobs=int(params.n_jobs),
        seed=int(params.seed),
        check_disjoint=True,
    )

    meta = {
        "kind": "erp",
        "n_subjects": int(X.shape[0]),
        "n_times": int(X.shape[1]),
        "n_channels": int(X.shape[2]),
        "n_permutations": int(params.n_permutations),
        "threshold": None if params.threshold is None else float(params.threshold),
        "tail": int(params.tail),
        "alpha": float(params.alpha),
        "ch_type": params.ch_type,
    }

    return ClusterTestResult(
        t_values=t_vals,
        clusters=list(clusters),
        p_values=p_vals,
        h0=h0,
        metadata=meta,
    )


def run_cluster_1samp_tfr(
    X: np.ndarray,
    *,
    info: mne.Info,
    n_freqs: int,
    params: ClusterTestParams,
) -> ClusterTestResult:
    """
    TFR cluster test.

    Parameters
    ----------
    X
        Shape (n_subjects, n_times, n_channels * n_freqs)
        i.e. time is the "temporal" axis, and the last axis is "space-like"
        (channels×freq). We then build adjacency over (channels, freqs).

    n_freqs
        Number of frequency bins used in the flattened last dimension.
        Requires last_dim % n_freqs == 0.
    """
    if X.ndim != 3:
        raise ValueError(f"TFR X must be (N,T,CF). Got {X.shape}")
    if combine_adjacency is None:
        raise RuntimeError("mne.stats.combine_adjacency is required for TFR cluster tests.")
    if n_freqs <= 1:
        raise ValueError("n_freqs must be >= 2 for TFR adjacency to be meaningful.")
    if X.shape[2] % n_freqs != 0:
        raise ValueError(f"Last dim {X.shape[2]} is not divisible by n_freqs={n_freqs}.")

    n_channels = X.shape[2] // n_freqs

    ch_adj = _channel_adjacency(info, params.ch_type)
    freq_adj = _line_adjacency(n_freqs)

    # adjacency over (channels, freqs) -> combined "space" adjacency
    space_adj = combine_adjacency(ch_adj, freq_adj)

    t_vals, clusters, p_vals, h0 = spatio_temporal_cluster_1samp_test(
        X,
        adjacency=space_adj,
        n_permutations=int(params.n_permutations),
        threshold=params.threshold,
        tail=int(params.tail),
        n_jobs=int(params.n_jobs),
        seed=int(params.seed),
        check_disjoint=True,
    )

    meta = {
        "kind": "tfr",
        "n_subjects": int(X.shape[0]),
        "n_times": int(X.shape[1]),
        "n_channels": int(n_channels),
        "n_freqs": int(n_freqs),
        "n_permutations": int(params.n_permutations),
        "threshold": None if params.threshold is None else float(params.threshold),
        "tail": int(params.tail),
        "alpha": float(params.alpha),
        "ch_type": params.ch_type,
    }

    return ClusterTestResult(
        t_values=t_vals,
        clusters=list(clusters),
        p_values=p_vals,
        h0=h0,
        metadata=meta,
    )


def _line_adjacency(n: int) -> np.ndarray:
    """Simple 1D adjacency for a line graph of length n."""
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        if i - 1 >= 0:
            adj[i, i - 1] = True
        if i + 1 < n:
            adj[i, i + 1] = True
    return adj
