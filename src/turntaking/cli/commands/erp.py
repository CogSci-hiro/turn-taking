"""CLI wiring for ERP generation."""

from __future__ import annotations

import argparse
from typing import Any

from turntaking.analysis.erp.entry import _expand_epoch_paths_from_config, run_erp


def add_subparser(subparsers: Any) -> None:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "erp",
        help="Run ERP data generation (config-driven by default).",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--epochs-glob", default=None, help="Override epoch file glob.")
    parser.add_argument("--out-dir", default=None, help="Override output directory.")
    parser.add_argument(
        "--contrast",
        choices=["latency", "duration"],
        default=None,
        help="Override: run only one contrast (otherwise run all from config).",
    )
    parser.add_argument("--min-latency", type=float, default=None)
    parser.add_argument("--max-latency", type=float, default=None)
    parser.add_argument("--min-response-duration", type=float, default=None)


def run(args: argparse.Namespace, cfg: Any) -> None:
    run_erp(args, cfg)


__all__ = ["add_subparser", "run", "_expand_epoch_paths_from_config"]
