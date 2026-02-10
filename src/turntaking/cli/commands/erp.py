"""turntaking.cli.commands.erp

CLI wrapper for building ERP evoked datasets.

This is intentionally thin:
- parse args
- resolve file paths
- call core analysis functions
- save outputs

"""

from __future__ import annotations

import argparse
from pathlib import Path

from turntaking.analysis.datasets.evoked_dataset import build_evoked_dataset
from turntaking.analysis.io import save_evokeds, save_table
from turntaking.analysis.selection import SelectionParams


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="turntaking erp", description="Build ERP evoked datasets.")
    parser.add_argument(
        "--epochs-glob",
        type=str,
        required=True,
        help="Glob pattern for epochs files (e.g., '/path/to/epochs/sub-*_task-*_run-*_epochs-epo.fif').",
    )
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--contrast",
        type=str,
        choices=["latency", "duration"],
        required=True,
        help="Median-split contrast variable.",
    )

    parser.add_argument("--min-latency", type=float, default=0.0, help="Selection: latency > min (s).")
    parser.add_argument("--max-latency", type=float, default=999.0, help="Selection: latency < max (s).")
    parser.add_argument("--min-self-duration", type=float, default=0.0, help="Selection: self_duration > min (s).")

    return parser


def main() -> None:
    """Entry point.

    Usage example
    -------------
        python -m turntaking.cli.commands.erp \
            --epochs-glob "/Volumes/work-4T/hyperscanning/derived/eeg/epochs/sub-*_task-conversation_run-*_epochs-epo.fif" \
            --out-dir "/tmp/turntaking_erp" \
            --contrast latency \
            --min-latency 0.05 --max-latency 1.0 --min-self-duration 0.2
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    import glob

    epoch_paths = [Path(p) for p in sorted(glob.glob(args.epochs_glob))]
    epoch_paths = [p for p in epoch_paths if p.is_file()]
    if len(epoch_paths) == 0:
        raise SystemExit(f"No files matched --epochs-glob: {args.epochs_glob}")

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
        contrast=args.contrast,
        selection_params=selection_params,
    )

    evokeds_to_save = {
        "grand_diff": result.difference,
    }
    save_evokeds(evokeds_to_save, out_dir=out_dir)

    save_table(result.metadata, out_dir / "metadata.parquet")
    save_table(result.n_trials, out_dir / "n_trials.csv")
