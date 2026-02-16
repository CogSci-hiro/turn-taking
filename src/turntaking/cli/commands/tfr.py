"""CLI wiring for TFR generation."""

from __future__ import annotations

import argparse
from typing import Any

from turntaking.analysis.tfr.entry import run_tfr


def add_subparser(subparsers: Any) -> None:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "tfr",
        help="Run band-limited induced (Hilbert) TFR data generation (ERP-like contract).",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--epochs-glob", default=None, help="Override epoch file glob.")
    parser.add_argument("--out-dir", default=None, help="Override output directory (base; 'tfr' appended).")
    parser.add_argument(
        "--contrast",
        choices=["latency", "duration"],
        default=None,
        help="Override: run only one contrast (otherwise run all from config).",
    )
    parser.add_argument(
        "--band",
        default=None,
        help="Override: run only one band (otherwise run all from config.analysis.bands).",
    )
    parser.add_argument("--min-latency", type=float, default=None)
    parser.add_argument("--max-latency", type=float, default=None)
    parser.add_argument("--min-response-duration", type=float, default=None)
    parser.add_argument("--sfreq", type=float, default=None)


def run(args: argparse.Namespace, cfg: Any) -> None:
    run_tfr(args, cfg)
