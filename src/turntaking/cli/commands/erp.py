"""CLI wiring for ERP generation."""


import argparse
from typing import Any


def add_subparser(subparsers: Any) -> None:
    """
    Register the ``turntaking analyze erp`` subcommand.

    This module is intentionally thin: it defines CLI flags and delegates all
    scientific work to ``turntaking.analysis.erp.entry.run_erp``.
    """
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
    """
    Execute the ERP analysis pipeline for the configured contrasts.

    Parameters
    ----------
    args
        Parsed CLI arguments from ``argparse``.
    cfg
        Loaded configuration (typically ``TurntakingConfig``).
    """
    from turntaking.analysis.erp.entry import run_erp

    run_erp(args, cfg)


def _expand_epoch_paths_from_config(cfg: Any):
    """
    Backward-compatible helper used by a few workflow entrypoints/tests.

    Prefer calling the library entrypoint (``run_erp``) rather than reaching
    into config expansion directly, unless you are validating discovery logic.
    """
    from turntaking.analysis.erp.entry import _expand_epoch_paths_from_config as _impl

    return _impl(cfg)


__all__ = ["add_subparser", "run", "_expand_epoch_paths_from_config"]
