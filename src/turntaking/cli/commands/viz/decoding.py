
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np

from turntaking.analysis.io.decoding import (
    ContrastName,
    DecodingScorePaths,
    get_decoding_out_dir,
    load_decoding_scores,
)
from turntaking.viz.figures.decoding import plot_decoding


# ##################################################################################################
# CLI REGISTRATION
# ##################################################################################################

def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Register the `viz-decoding` command.

    Usage example
    -------------
        python -m turntaking.cli.main viz-decoding --config workflow/config.yaml --out out/figures/fig5_decoding.png
    """
    p = subparsers.add_parser("viz-decoding", help="Make decoding figure (diag + temporal generalization).")
    p.add_argument("--config", type=Path, required=True, help="Path to workflow/config.yaml")
    p.add_argument("--out", type=Path, required=True, help="Output image path (e.g., fig5_decoding.png)")
    p.add_argument("--p-threshold", type=float, default=0.05, help="Cluster p-value threshold")
    p.add_argument("--ymax", type=float, default=0.65, help="Diagonal panel y-axis max")
    p.set_defaults(func=run)


# ##################################################################################################
# Implementation (CLI only: load + call plotting)
# ##################################################################################################

@dataclass(frozen=True)
class _ClusterResults:
    clusters: List[Tuple[np.ndarray, np.ndarray]]
    p_values: np.ndarray


def _load_decoding_cluster_results_for_contrast(
    *,
    out_dir: Path,
    contrast: ContrastName,
) -> _ClusterResults:
    """
    Load decoding cluster results saved by `decoding-cluster`.

    Expected layout:
        {out_dir}/stats/decoding/erp/{contrast}/cluster_results.hdf5

    Expected datasets:
        p-values
        clusters/train-{i}
        clusters/test-{i}
    """
    hdf5_path = out_dir / "stats" / "decoding" / "erp" / contrast / "cluster_results.hdf5"
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Missing decoding cluster results: {hdf5_path}")

    clusters: List[Tuple[np.ndarray, np.ndarray]] = []

    with h5py.File(hdf5_path, "r") as f:
        p_values = np.asarray(f["p-values"], dtype=float)

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

    return _ClusterResults(clusters=clusters, p_values=p_values)


def _load_scores_for_contrast(
    *,
    out_dir: Path,
    contrast: ContrastName,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Load scores + times for a given contrast and compute (tmin, tmax).
    """
    decoding_dir = get_decoding_out_dir(out_dir, contrast)
    scores, times_s = load_decoding_scores(DecodingScorePaths.from_dir(decoding_dir))

    tmin = float(times_s.min())
    tmax = float(times_s.max())
    return scores, times_s, tmin, tmax


def run(args: argparse.Namespace, cfg) -> None:
    """
    CLI entrypoint. `cfg` is a TurntakingConfig object, passed by turntaking.cli.main.
    """
    out_dir: Path = cfg.io.out_dir

    # Load both contrasts
    duration_scores, duration_times_s, tmin_d, tmax_d = _load_scores_for_contrast(out_dir=out_dir, contrast="duration")
    latency_scores, latency_times_s, tmin_l, tmax_l = _load_scores_for_contrast(out_dir=out_dir, contrast="latency")

    # Sanity: must share the same time axis for the shared plot window
    if duration_times_s.shape != latency_times_s.shape or not np.allclose(duration_times_s, latency_times_s):
        raise ValueError("Duration and latency decoding times do not match; cannot plot in one figure.")

    tmin = min(tmin_d, tmin_l)
    tmax = max(tmax_d, tmax_l)

    duration_stats = _load_decoding_cluster_results_for_contrast(out_dir=out_dir, contrast="duration")
    latency_stats = _load_decoding_cluster_results_for_contrast(out_dir=out_dir, contrast="latency")

    fig = plot_decoding(
        tmin=tmin,
        tmax=tmax,
        duration_scores=duration_scores,
        latency_scores=latency_scores,
        duration_clusters=duration_stats.clusters,
        latency_clusters=latency_stats.clusters,
        duration_p=duration_stats.p_values,
        latency_p=latency_stats.p_values,
        p_threshold=float(args.p_threshold),
        ymax=float(args.ymax),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
