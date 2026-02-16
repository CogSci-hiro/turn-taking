
from dataclasses import dataclass
from typing import Final, Iterable

import numpy as np
import pandas as pd
from mne.stats import spatio_temporal_cluster_1samp_test


# ##################################################################################################
# Decoding cluster test
# ##################################################################################################

_DEFAULT_CHANCE_LEVEL: Final[float] = 0.5


@dataclass(frozen=True)
class DecodingClusterTestParams:
    """
    Parameters for cluster-based permutation testing of temporal generalization AUC scores.

    Notes
    -----
    We assume AUC scores with chance level 0.5. We test (scores - 0.5) against 0 using a one-tailed
    cluster permutation test (tail=1), mirroring the legacy implementation.

    Usage example
    -------------
        params = DecodingClusterTestParams(
            threshold=None,
            n_permutations=1000,
            tail=1,
            n_jobs=10,
            chance_level=0.5,
        )
    """

    threshold: float | None
    n_permutations: int
    tail: int
    n_jobs: int
    chance_level: float = _DEFAULT_CHANCE_LEVEL


def run_decoding_cluster_test(
    *,
    scores: np.ndarray,
    params: DecodingClusterTestParams,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """
    Run a one-sample cluster permutation test on temporal generalization decoding scores vs chance.

    Parameters
    ----------
    scores
        Temporal generalization AUC scores shaped:
            (n_subjects, n_splits, n_times, n_times)
        or already-averaged across CV splits:
            (n_subjects, n_times, n_times)

    params
        Cluster test parameters.

    Returns
    -------
    t_values
        T-statistics array shaped (n_times, n_times).

    clusters
        List of clusters as index tuples (train_time_indices, test_time_indices).

    p_values
        P-values per cluster shaped (n_clusters,).

    h0
        Max cluster-level stats under H0 shaped (n_permutations,).

    Usage example
    -------------
        t_vals, clusters, p_vals, h0 = run_decoding_cluster_test(scores=scores, params=params)
    """
    if scores.ndim not in (3, 4):
        raise ValueError(
            "Expected scores with ndim 3 or 4: "
            "(n_subjects,n_times,n_times) or (n_subjects,n_splits,n_times,n_times). "
            f"Got shape={scores.shape}."
        )

    if scores.ndim == 4:
        # (n_subjects, n_splits, n_times, n_times) -> (n_subjects, n_times, n_times)
        scores_ = scores.mean(axis=1)
    else:
        scores_ = scores

    if scores_.shape[0] < 2:
        raise ValueError(f"Need >=2 subjects for a group test; got n_subjects={scores_.shape[0]}.")

    if scores_.shape[1] != scores_.shape[2]:
        raise ValueError(
            "Expected square temporal generalization matrices (n_times,n_times). "
            f"Got shape={scores_.shape}."
        )

    t_values, clusters, p_values, h0 = spatio_temporal_cluster_1samp_test(
        scores_ - params.chance_level,
        out_type="indices",
        tail=params.tail,
        threshold=params.threshold,
        n_permutations=params.n_permutations,
        n_jobs=params.n_jobs,
        verbose=False,
    )
    return t_values, list(clusters), p_values, h0


def make_decoding_cluster_summary(
    *,
    clusters: Iterable[tuple[np.ndarray, np.ndarray]],
    p_values: np.ndarray,
    times_s: np.ndarray,
) -> pd.DataFrame:
    """
    Create a human-readable per-cluster summary table.

    Parameters
    ----------
    clusters
        Cluster index tuples (train_time_indices, test_time_indices).

    p_values
        P-values for each cluster (n_clusters,).

    times_s
        Time axis in seconds shaped (n_times,).

    Returns
    -------
    summary_df
        DataFrame with one row per cluster.

    DataFrame format example
    ------------------------
    | cluster_id | p_value | n_points | train_tmin_s | train_tmax_s | test_tmin_s | test_tmax_s |
    |-----------:|--------:|---------:|-------------:|-------------:|------------:|------------:|
    |          0 |   0.004 |     1820 |       -1.250 |       -0.062 |      -1.125 |       0.000 |
    """
    times_s = np.asarray(times_s, dtype=float)
    if times_s.ndim != 1:
        raise ValueError(f"Expected times_s as (n_times,), got shape={times_s.shape}.")

    p_values = np.asarray(p_values, dtype=float)
    rows: list[dict[str, float | int]] = []

    for idx, (train_idx, test_idx) in enumerate(clusters):
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        if train_idx.size == 0 or test_idx.size == 0:
            # Should not happen, but guard anyway
            train_tmin = np.nan
            train_tmax = np.nan
            test_tmin = np.nan
            test_tmax = np.nan
            n_points = 0
        else:
            train_tmin = float(times_s[train_idx].min())
            train_tmax = float(times_s[train_idx].max())
            test_tmin = float(times_s[test_idx].min())
            test_tmax = float(times_s[test_idx].max())
            n_points = int(train_idx.size)  # indices are paired; size is a decent “mass” proxy

        rows.append(
            {
                "cluster_id": idx,
                "p_value": float(p_values[idx]) if idx < p_values.size else np.nan,
                "n_points": n_points,
                "train_tmin_s": train_tmin,
                "train_tmax_s": train_tmax,
                "test_tmin_s": test_tmin,
                "test_tmax_s": test_tmax,
            }
        )

    return pd.DataFrame(rows)
