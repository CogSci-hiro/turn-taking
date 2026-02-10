from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from turntaking.analysis.datasets.evoked_dataset import build_evoked_dataset
from turntaking.analysis.io import save_evokeds, save_table
from turntaking.analysis.selection import SelectionParams


def add_subparser(subparsers: Any) -> None:
    """Register the `erp-generate` subcommand."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "erp-generate",
        help="Generate ERP evokeds from already-preprocessed epochs.",
    )
    parser.add_argument("--config", required=True, help="Path to project YAML config.")
    parser.add_argument("--epochs", nargs="+", required=True, help="One or more epoch FIF files.")
    parser.add_argument("--out-dir", required=True, help="Output directory for ERP artifacts.")
    parser.add_argument("--contrast", choices=["latency", "duration"], required=True)

    parser.add_argument("--min-latency", type=float, default=0.0)
    parser.add_argument("--max-latency", type=float, default=10.0)
    parser.add_argument("--min-self-duration", type=float, default=0.0)

    parser.set_defaults(command="erp-generate")


def run(args: argparse.Namespace, cfg: Any) -> None:
    """Run ERP generation."""
    epoch_paths = [Path(p) for p in args.epochs]
    selection = SelectionParams(
        min_latency=float(args.min_latency),
        max_latency=float(args.max_latency),
        min_self_duration=float(args.min_self_duration),
    )

    result = build_evoked_dataset(
        epoch_paths,
        kind="erp",
        contrast=str(args.contrast),
        selection_params=selection,
    )

    out_dir = Path(args.out_dir)
    save_evokeds({"grand_diff": result.difference}, out_dir)
    save_table(result.metadata, out_dir / "metadata.parquet")
    save_table(result.n_trials, out_dir / "n_trials.csv")
