"""CLI wiring for ERP/TFR cluster permutation testing."""

from __future__ import annotations

import argparse
from typing import Any

from turntaking.analysis.cluster.entry import run_cluster


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("cluster", help="Run cluster permutation tests (ERP/TFR).")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--kind", required=True, choices=["erp", "tfr"], help="Which analysis kind to test.")
    parser.add_argument("--contrast", required=True, help="Contrast name (e.g. latency, duration).")
    parser.add_argument("--band", default=None, help="Band name (required when --kind tfr), e.g. alpha, beta.")
    parser.add_argument("--n-permutations", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tail", type=int, choices=[-1, 0, 1], default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ch-type", choices=["eeg", "mag", "grad"], default=None)


def run(args: Any, cfg: Any) -> None:
    run_cluster(args, cfg)
