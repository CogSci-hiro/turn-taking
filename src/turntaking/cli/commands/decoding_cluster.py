"""CLI wiring for decoding cluster statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("decoding-cluster", help="Cluster test for decoding temporal generalization scores.")
    parser.add_argument("--config", type=Path, required=True, help="Path to workflow/config.yaml")
    parser.add_argument("--feature", type=str, default="erp", choices=["erp"], help="Decoding feature family.")
    parser.add_argument("--contrast", type=str, required=True, choices=["duration", "latency"], help="Contrast to test.")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace, cfg: Any) -> None:
    from turntaking.analysis.decoding.cluster_entry import run_decoding_cluster

    run_decoding_cluster(args, cfg)
