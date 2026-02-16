"""CLI wiring for decoding generation."""


import argparse
from typing import Any

def add_subparser(subparsers: Any) -> None:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "decoding",
        help="Run ERP temporal-generalization decoding and write scores.npy/times.npy.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--epochs-glob", default=None, help="Override epoch file glob (validation only).")
    parser.add_argument("--out-dir", default=None, help="Override base output directory (writes into decoding/erp/*).")
    parser.add_argument(
        "--contrast",
        choices=["latency", "duration"],
        required=True,
        help="Which contrast to decode.",
    )
    parser.add_argument("--sfreq", type=float, default=None, help="Override decoding sfreq (Hz).")
    parser.add_argument("--n-splits", type=int, default=None, help="Override number of CV folds.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed for deterministic CV.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Override parallel jobs for decoding.")
    parser.add_argument(
        "--cache-features",
        action="store_true",
        help="Cache per-subject (X,y,times) features to HDF5 for faster iteration.",
    )
    parser.add_argument("--min-latency", type=float, default=None)
    parser.add_argument("--max-latency", type=float, default=None)
    parser.add_argument("--min-response-duration", type=float, default=None)


def run(args: argparse.Namespace, cfg: Any) -> None:
    from turntaking.analysis.decoding.entry import run_decoding

    run_decoding(args, cfg)
