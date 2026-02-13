from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class DecodingClusterResults:
    """
    Cluster test results for temporal generalization decoding.

    Attributes
    ----------
    t_values
        (n_times, n_times) t-statistics.
    clusters
        List of (train_idx, test_idx) arrays (MNE out_type="indices" style).
    p_values
        (n_clusters,) p-values.
    h0
        (n_permutations,) max cluster statistics under H0.
    """

    t_values: np.ndarray
    clusters: list[tuple[np.ndarray, np.ndarray]]
    p_values: np.ndarray
    h0: np.ndarray


def load_decoding_cluster_results_hdf5(path: Path) -> DecodingClusterResults:
    """
    Load cluster results saved by decoding_cluster CLI.

    Usage example
    -------------
        res = load_decoding_cluster_results_hdf5(Path(".../cluster_results.hdf5"))
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing cluster results: {path}")

    with h5py.File(path, "r") as f:
        t_values = np.asarray(f["t-values"])
        p_values = np.asarray(f["p-values"])
        h0 = np.asarray(f["h0"])

        clusters: list[tuple[np.ndarray, np.ndarray]] = []
        i = 0
        while True:
            k_train = f"clusters/train-{i}"
            k_test = f"clusters/test-{i}"
            if k_train not in f or k_test not in f:
                break
            train_idx = np.asarray(f[k_train], dtype=int)
            test_idx = np.asarray(f[k_test], dtype=int)
            clusters.append((train_idx, test_idx))
            i += 1

    return DecodingClusterResults(
        t_values=t_values,
        clusters=clusters,
        p_values=p_values,
        h0=h0,
    )
