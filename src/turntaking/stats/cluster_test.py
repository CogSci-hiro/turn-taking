# src/turntaking/stats/cluster_test.py


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
    return
