from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from turntaking.analysis.io.decoding import DecodingScorePaths, load_decoding_scores
from turntaking.stats.decoding_cluster_test import (
    DecodingClusterTestParams,
    make_decoding_cluster_summary,
    run_decoding_cluster_test,
)
from turntaking.config.analysis_schema import TurntakingConfig
from turntaking.analysis.io.decoding import (
    DecodingScorePaths,
    get_decoding_out_dir,
    load_decoding_scores,
)


# ##################################################################################################
# CLI REGISTRATION
# ##################################################################################################

def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Register the `decoding-cluster` command.

    Usage example
    -------------
        python -m turntaking.cli.main decoding-cluster \
          --config workflow/config.yaml \
          --feature erp \
          --contrast duration
    """
    p = subparsers.add_parser("decoding-cluster", help="Cluster test for decoding temporal generalization scores.")
    p.add_argument("--config", type=Path, required=True, help="Path to workflow/config.yaml")
    p.add_argument("--feature", type=str, default="erp", choices=["erp"], help="Decoding feature family.")
    p.add_argument("--contrast", type=str, required=True, choices=["duration", "latency"], help="Contrast to test.")
    p.set_defaults(func=run)


# ##################################################################################################
# Implementation
# ##################################################################################################

@dataclass(frozen=True)
class _CliCfg:
    config: Path
    feature: str
    contrast: str


def _save_cluster_results_hdf5(
    *,
    out_hdf5: Path,
    t_values: np.ndarray,
    clusters: list[tuple[np.ndarray, np.ndarray]],
    p_values: np.ndarray,
    h0: np.ndarray,
) -> None:
    out_hdf5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_hdf5, "w") as f:
        f.create_dataset("t-values", data=t_values, dtype=float)
        f.create_dataset("p-values", data=p_values, dtype=float)
        f.create_dataset("h0", data=h0, dtype=float)

        for idx, (train_idx, test_idx) in enumerate(clusters):
            f.create_dataset(f"clusters/train-{idx}", data=np.asarray(train_idx, dtype=int), dtype=int)
            f.create_dataset(f"clusters/test-{idx}", data=np.asarray(test_idx, dtype=int), dtype=int)


def run(args: argparse.Namespace, cfg: TurntakingConfig) -> None:
    out_dir = cfg.io.out_dir  # <- correct section name from your dataclass

    # Align with your decoding I/O helpers (currently ERP-only)
    if args.feature != "erp":
        raise ValueError(f"Only feature='erp' is supported right now; got feature={args.feature!r}.")

    decoding_dir = get_decoding_out_dir(out_dir, args.contrast)
    scores, times_s = load_decoding_scores(DecodingScorePaths.from_dir(decoding_dir))

    # Pull params from typed config
    decoding_cfg = cfg.analysis.decoding

    params = DecodingClusterTestParams(
        threshold=decoding_cfg.threshold,
        n_permutations=int(decoding_cfg.n_permutations),
        tail=int(getattr(decoding_cfg, "tail", 1)),
        n_jobs=int(getattr(decoding_cfg, "n_jobs", 1)),
        chance_level=float(getattr(decoding_cfg, "chance_level", 0.5)),
    )

    t_values, clusters, p_values, h0 = run_decoding_cluster_test(scores=scores, params=params)

    stats_dir = out_dir / "stats" / "decoding" / "erp" / args.contrast
    out_hdf5 = stats_dir / "cluster_results.hdf5"
    out_csv = stats_dir / "cluster_summary.csv"

    _save_cluster_results_hdf5(
        out_hdf5=out_hdf5,
        t_values=t_values,
        clusters=clusters,
        p_values=p_values,
        h0=h0,
    )

    summary_df = make_decoding_cluster_summary(clusters=clusters, p_values=p_values, times_s=times_s)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)

