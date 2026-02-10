# src/turntaking/cli/commands/cluster.py

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import mne
import numpy as np

from turntaking.analysis.io.cluster import write_cluster_outputs
from turntaking.stats.cluster_test import (
    ClusterTestParams,
    run_cluster_1samp_erp,
    run_cluster_1samp_tfr,
)
from turntaking.stats.cropping import crop_time_margins_samples

Kind = Literal["erp", "tfr"]


# =============================================================================
#                     ########################################
#                     #            CLI REGISTRATION          #
#                     ########################################
# =============================================================================
def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Register the `cluster` command.

    Usage example
    -------------
        python -m turntaking.cli.main cluster --config workflow/config.yaml --kind erp --contrast latency
    """
    p = subparsers.add_parser(
        "cluster",
        help="Run cluster permutation tests (ERP/TFR).",
    )

    # Your CLI framework expects each command to accept --config.
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

    # For ERP this is required (latency/duration). For TFR you can reuse or ignore later.
    p.add_argument(
        "--contrast",
        required=True,
        help="Contrast name (e.g. latency, duration).",
    )

    # Optional overrides (if omitted, values are read from config.yaml)
    p.add_argument("--n-permutations", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--tail", type=int, choices=[-1, 0, 1], default=None)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--ch-type", choices=["eeg", "mag", "grad"], default=None)

    # If you later want CLI cropping overrides, add:
    # p.add_argument("--left-margin", type=float, default=None)
    # p.add_argument("--right-margin", type=float, default=None)
    # p.add_argument("--sfreq", type=float, default=None)


# =============================================================================
#                     ########################################
#                     #              CONFIG READ             #
#                     ########################################
# =============================================================================
def _load_params(cfg: Any, args: Any) -> ClusterTestParams:
    """
    Load cluster-test params from TurntakingConfig (attribute-based).

    Reads from:
      - cfg.analysis.erp.*  when args.kind == "erp"
      - cfg.analysis.tfr.*  when args.kind == "tfr"

    Expected YAML fields (matching your config.yaml)
    -----------------------------------------------
    analysis:
      erp:
        n_permutations: 1000
        threshold: null
      tfr:
        n_permutations: 1000
        threshold: null
    """
    analysis = getattr(cfg, "analysis", None)
    if analysis is None:
        raise ValueError("Config missing 'analysis' section (cfg.analysis).")

    if args.kind == "erp":
        section = getattr(analysis, "erp", None)
    elif args.kind == "tfr":
        section = getattr(analysis, "tfr", None)
    else:
        raise ValueError(f"Invalid kind: {args.kind!r}")

    if section is None:
        raise ValueError(f"Config missing analysis.{args.kind} section.")

    params = ClusterTestParams(
        n_permutations=int(getattr(section, "n_permutations", 1024)),
        threshold=getattr(section, "threshold", None),
        tail=int(getattr(section, "tail", 0)),          # optional (not in your YAML)
        alpha=float(getattr(section, "alpha", 0.05)),   # optional
        seed=int(getattr(section, "seed", 0)),          # optional
        n_jobs=int(getattr(section, "n_jobs", 1)),      # optional
        ch_type=str(getattr(section, "ch_type", "eeg")),# optional
    )

    # CLI overrides (optional)
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


def _load_crop_settings(cfg: Any, kind: Kind) -> tuple[float, float, float]:
    """
    Load (left_margin, right_margin, sfreq) from cfg.analysis.{kind}.
    """
    analysis = getattr(cfg, "analysis", None)
    if analysis is None:
        raise ValueError("Config missing 'analysis' section (cfg.analysis).")

    section = getattr(analysis, kind, None)
    if section is None:
        raise ValueError(f"Config missing analysis.{kind} section.")

    left_margin = float(getattr(section, "left_margin"))
    right_margin = float(getattr(section, "right_margin"))
    sfreq = float(getattr(section, "sfreq"))

    return left_margin, right_margin, sfreq


# =============================================================================
#                     ########################################
#                     #                 RUN                 #
#                     ########################################
# =============================================================================
def run(args: Any, cfg: Any) -> None:
    """
    Run cluster permutation tests and write stats artifacts.

    Notes
    -----
    ERP expects:
      {out_dir}/erp/{contrast}/evoked-data.npy   shape (N,3,C,T)
      {out_dir}/erp/{contrast}/difference_ave.fif

    Outputs:
      {out_dir}/stats/{kind}/{contrast}/cluster_results.hdf5
      {out_dir}/stats/{kind}/{contrast}/cluster_summary.csv

    Usage example
    -------------
        python -m turntaking.cli.main cluster --config workflow/config.yaml --kind erp --contrast latency
    """
    kind: Kind = args.kind
    contrast: str = args.contrast

    # Root output directory comes from cfg.io.out_dir (TurntakingConfig attribute)
    io_cfg = getattr(cfg, "io", None)
    if io_cfg is None:
        raise ValueError("Config missing 'io' section (cfg.io).")

    out_root = Path(getattr(io_cfg, "out_dir"))
    stats_out_dir = out_root / "stats" / kind / contrast

    params = _load_params(cfg, args)
    left_margin, right_margin, sfreq = _load_crop_settings(cfg, kind)

    print(
        f"[cluster] kind={kind} contrast={contrast} "
        f"n_permutations={params.n_permutations} threshold={params.threshold} "
        f"left_margin={left_margin} right_margin={right_margin} sfreq={sfreq}"
    )

    if kind == "erp":
        evoked_data_path = out_root / "erp" / contrast / "evoked-data.npy"
        diff_ave_path = out_root / "erp" / contrast / "difference_ave.fif"

        arr = np.load(evoked_data_path)  # (N,3,C,T)
        if arr.ndim != 4 or arr.shape[1] != 3:
            raise ValueError(
                f"Unexpected ERP evoked-data.npy shape: {arr.shape} "
                "(expected (N,3,C,T))."
            )

        # Take diff slot and convert to (N,T,C)
        diff = arr[:, 2, :, :]  # (N,C,T)
        X = np.transpose(diff, (0, 2, 1)).astype(float)  # (N,T,C)

        # Crop margins (same behavior as legacy script)
        X, start_idx, end_idx = crop_time_margins_samples(
            X, sfreq=sfreq, left_margin=left_margin, right_margin=right_margin
        )

        # Reference evoked for channel adjacency (and for sanity)
        evoked = mne.read_evokeds(diff_ave_path, condition=0)

        result = run_cluster_1samp_erp(X, info=evoked.info, params=params)

        # Record crop metadata (helps reproducibility/audit)
        result.metadata["crop_left_margin"] = float(left_margin)
        result.metadata["crop_right_margin"] = float(right_margin)
        result.metadata["crop_sfreq_used"] = float(sfreq)
        result.metadata["crop_start_idx"] = int(start_idx)
        result.metadata["crop_end_idx"] = int(end_idx)

        write_cluster_outputs(stats_out_dir, result)
        return

    if kind == "tfr":
        # You said cropping is shared; this branch is the same idea:
        # build X with shape (N,T,S) where S is "space-like" (e.g. channels×freq),
        # crop with crop_time_margins_samples(), then call run_cluster_1samp_tfr().
        #
        # This needs your TFR artifact contract (where the per-subject arrays live and
        # how to infer n_freqs / space shape). Wire once TFR outputs are finalized.
        raise NotImplementedError(
            "TFR cluster test not wired yet (needs a TFR dataset artifact contract)."
        )

    raise ValueError(f"Unknown kind: {kind!r}")
