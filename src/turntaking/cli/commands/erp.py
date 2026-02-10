from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

from turntaking.analysis.datasets.evoked_dataset import build_evoked_dataset
from turntaking.analysis.io import save_evokeds, save_table
from turntaking.analysis.selection import SelectionParams


def add_subparser(subparsers: Any) -> None:
    """Register the `erp` subcommand."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "erp",
        help="Build ERP evoked datasets (thin wrapper).",
    )

    # IMPORTANT: this must exist because cli/main.py requires args.config
    parser.add_argument("--config", required=True, help="Path to YAML config.")

    # Keep these for now since you don’t have config-driven epoch discovery wired yet
    parser.add_argument(
        "--epochs-glob",
        required=True,
        help="Glob pattern for epochs FIF files.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--contrast", choices=["latency", "duration"], required=True)

    parser.add_argument("--min-latency", type=float, default=0.0)
    parser.add_argument("--max-latency", type=float, default=999.0)
    parser.add_argument("--min-self-duration", type=float, default=0.0)


def run(args: argparse.Namespace, cfg: Any) -> None:
    """Run ERP generation."""
    epoch_paths = [Path(p) for p in sorted(glob.glob(str(args.epochs_glob))) if Path(p).is_file()]
    if len(epoch_paths) == 0:
        raise RuntimeError(f"No files matched --epochs-glob: {args.epochs_glob}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selection_params = SelectionParams(
        min_latency=float(args.min_latency),
        max_latency=float(args.max_latency),
        min_self_duration=float(args.min_self_duration),
    )

    result = build_evoked_dataset(
        epoch_paths=epoch_paths,
        kind="erp",
        contrast=str(args.contrast),
        selection_params=selection_params,
    )

    save_evokeds({"grand_diff": result.difference}, out_dir=out_dir)
    save_table(result.metadata, out_dir / "metadata.parquet")
    save_table(result.n_trials, out_dir / "n_trials.csv")
