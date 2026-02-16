# src/turntaking/stats/cluster_test.py

from dataclasses import dataclass
from typing import Any, Literal

import mne
import numpy as np
from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_1samp_test


ClusterKind = Literal["erp", "tfr"]


@dataclass(frozen=True)
class ClusterTestParams:
    """Parameters controlling the cluster permutation test."""
    n_permutations: int = 1024
    threshold: float | dict[str, Any] | None = None
    tail: int = 0  # 0: two-sided, 1: upper tail, -1: lower tail
    alpha: float = 0.05
    seed: int = 0
    n_jobs: int = 1
    ch_type: Literal["eeg", "mag", "grad"] = "eeg"


@dataclass(frozen=True)
class ClusterTestResult:
    """Outputs of a spatio-temporal cluster permutation test."""
    t_values: np.ndarray
    clusters: list[tuple[np.ndarray, ...]]
    p_values: np.ndarray
    h0: np.ndarray
    metadata: dict[str, Any]


def _channel_adjacency(info: mne.Info, ch_type: str) -> np.ndarray:
    adj, _ = find_ch_adjacency(info, ch_type=ch_type)
    return adj


def _serialize_threshold(threshold: float | dict[str, Any] | None) -> float | dict[str, Any] | None:
    if threshold is None:
        return None
    if isinstance(threshold, dict):
        out: dict[str, Any] = {}
        for key, value in threshold.items():
            if isinstance(value, (np.integer, np.floating)):
                out[str(key)] = float(value)
            elif isinstance(value, (int, float)):
                out[str(key)] = float(value)
            else:
                out[str(key)] = value
        return out
    return float(threshold)


def run_cluster_1samp_spatiotemporal(
    X: np.ndarray,
    *,
    info: mne.Info,
    params: ClusterTestParams,
    kind: ClusterKind,
    data_tmin_s: float,
    sfreq_hz: float,
) -> ClusterTestResult:
    """
    Spatio-temporal cluster test for ERP-like data.

    This function is intentionally used for BOTH:
    - ERP (evoked difference waves)
    - TFR (band-averaged induced envelope; frequency already collapsed)

    Parameters
    ----------
    X
        Shape (n_subjects, n_times, n_channels). Matches the legacy stats layout.
    info
        MNE Info providing channel layout for adjacency.
    params
        Cluster test parameters.
    kind
        "erp" or "tfr" (stored in metadata only; test is identical).

    Returns
    -------
    ClusterTestResult
        Cluster test outputs.

    Usage example
    -------------
        result = run_cluster_1samp_spatiotemporal(
            X,
            info=evoked.info,
            params=ClusterTestParams(n_permutations=1000, n_jobs=8),
            kind="erp",
        )
    """
    if X.ndim != 3:
        raise ValueError(f"X must be (N,T,C). Got shape={X.shape}")
    if kind not in ("erp", "tfr"):
        raise ValueError(f"kind must be 'erp' or 'tfr'. Got {kind!r}")

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
        "kind": str(kind),
        "n_subjects": int(X.shape[0]),
        "n_times": int(X.shape[1]),
        "n_channels": int(X.shape[2]),
        "n_permutations": int(params.n_permutations),
        "threshold": _serialize_threshold(params.threshold),
        "tail": int(params.tail),
        "alpha": float(params.alpha),
        "seed": int(params.seed),
        "n_jobs": int(params.n_jobs),
        "ch_type": str(params.ch_type),
        "data_tmin": float(data_tmin_s),
        "sfreq_hz": float(sfreq_hz),
    }

    return ClusterTestResult(
        t_values=t_vals,
        clusters=list(clusters),
        p_values=p_vals,
        h0=h0,
        metadata=meta,
    )
