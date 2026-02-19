"""CLI wiring for mixed-effect table generation."""


import argparse
from pathlib import Path
from typing import Any

def add_subparser(subparsers) -> None:
    parser = subparsers.add_parser(
        "mixed-effect",
        help="Export trial-level CSV table for R mixed-effects models.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to workflow config YAML.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional override for output CSV path.")


def run(args: argparse.Namespace, cfg: Any) -> None:
    from turntaking.analysis.mixed_effect.entry import run_mixed_effect

    run_mixed_effect(args, cfg)
