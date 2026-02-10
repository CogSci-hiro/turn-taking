# src/turntaking/cli/commands/cluster.py

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import mne
import numpy as np

from turntaking.analysis.io.cluster import write_cluster_outputs
from turntaking.stats.cluster_test import (
    ClusterKind,
    ClusterTestParams,
    run_cluster_1samp_erp,
    run_cluster_1samp_tfr,
)


@dataclass(frozen=True)
class ClusterCommandConfig:
    kind: ClusterKind
    contrast: str
    out_dir: Path
    evoked_data_npy: Path
    reference_evoked_fif: Path  # for Info / channel layout
    n_freqs: int | None = None  # required for TFR


def _getattr_default(obj: Any, name: str, default: Any) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def _load_params(cfg: Any, args: Any) -> ClusterTestParams:
    """
    Read cluster-test params from TurntakingConfig (attribute-based),
    then allow CLI flags to override if provided.
    """
    # Expect something like cfg.stats.cluster.{...}
    stats = _getattr_default(cfg, "stats", None)
    cluster = _getattr_default(stats, "cluster", None)

    params = ClusterTestParams(
        n_permutations=int(_getattr_default(cluster, "n_permutations", 1024)),
        threshold=_getattr_default(cluster, "threshold", None),
        tail=int(_getattr_default(cluster, "tail", 0)),
        alpha=float(_getattr_default(cluster, "alpha", 0.05)),
        seed=int(_getattr_default(cluster, "seed", 0)),
        n_jobs=int(_getattr_default(cluster, "n_jobs", 1)),
        ch_type=str(_getattr_default(cluster, "ch_type", "eeg")),
    )

    # Optional CLI overrides
    if getattr(args, "n_permutations", None) is not None:
        params = replace(params, n_permutations=int(args.n_permutations))
    if getattr(args, "threshold", None) is not None:
        params = replace(params, threshold=float(args.threshold))
    if getattr(args, "tail", None) is not None:
        params = replace(params, tail=int(args.tail))
    if getattr(args, "n_jobs", None) is not None:
        params = replace(params, n_jobs=int(args.n_jobs))
    if getattr(args, "seed", None) is not None:
        params = replace(params, seed=int(args.seed))
    if getattr(args, "ch_type", None) is not None:
        params = replace(params, ch_type=str(args.ch_type))

    return params


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "cluster",
        help="Run cluster permutation tests (ERP/TFR).",
    )

    # REQUIRED because your main CLI expects each command to accept --config
    p.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )

    p.add_argument(
        "--kind",
        required=True,
        choices=["erp", "tfr"],
        help="Which analysis kind to test.",
    )
    p.add_argument(
        "--contrast",
        required=True,
        help="Contrast name (e.g. latency, duration).",
    )

    # Optional overrides (safe to keep even if run() ignores them for now)
    p.add_argument("--n-permutations", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--tail", type=int, choices=[-1, 0, 1], default=None)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--ch-type", choices=["eeg", "mag", "grad"], default=None)


def run(args: Any, cfg: Any) -> None:
    params = _load_params(cfg, args)

    # Expect cfg.io.out_dir
    io_cfg = getattr(cfg, "io")
    out_root = Path(getattr(io_cfg, "out_dir"))

    out_dir = out_root / "stats" / args.kind / args.contrast
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.kind == "erp":
        evoked_data_path = out_root / "erp" / args.contrast / "evoked-data.npy"
        diff_ave_path = out_root / "erp" / args.contrast / "difference_ave.fif"

        arr = np.load(evoked_data_path)
        # (N,3,C,T) -> take diff slot -> (N,T,C)
        X = np.transpose(arr[:, 2, :, :], (0, 2, 1)).astype(float)

        ref = mne.read_evokeds(diff_ave_path, condition=0)
        result = run_cluster_1samp_erp(X, info=ref.info, params=params)

        write_cluster_outputs(out_dir, result)
        return

    if args.kind == "tfr":
        raise NotImplementedError("TFR cluster test not wired yet.")

    raise ValueError(f"Unknown kind: {args.kind!r}")
