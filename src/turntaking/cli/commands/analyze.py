"""Grouped analysis command: `turntaking analyze <target>`."""

from __future__ import annotations

import argparse
from typing import Any

def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("analyze", help="Run analysis pipelines.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    analyze_subparsers = parser.add_subparsers(dest="analysis_target", required=True)

    erp = analyze_subparsers.add_parser("erp", help="Run ERP analysis.")
    _add_erp_like_args(erp)

    tfr = analyze_subparsers.add_parser("tfr", help="Run TFR analysis.")
    _add_tfr_like_args(tfr)

    decoding = analyze_subparsers.add_parser("decoding", help="Run decoding analysis.")
    _add_decoding_args(decoding)

    mixed = analyze_subparsers.add_parser("mixed", help="Run mixed-effect table generation.")
    mixed.add_argument("--out-csv", default=None)

    all_parser = analyze_subparsers.add_parser("all", help="Run ERP, TFR, decoding, and mixed-effect.")
    all_parser.add_argument("--out-dir", default=None)


def run(args: argparse.Namespace, cfg: Any) -> None:
    if args.analysis_target == "erp":
        from turntaking.analysis.erp.entry import run_erp

        run_erp(args, cfg)
        return
    if args.analysis_target == "tfr":
        from turntaking.analysis.tfr.entry import run_tfr

        run_tfr(args, cfg)
        return
    if args.analysis_target == "decoding":
        from turntaking.analysis.decoding.entry import run_decoding

        run_decoding(args, cfg)
        return
    if args.analysis_target == "mixed":
        from turntaking.analysis.mixed_effect.entry import run_mixed_effect

        run_mixed_effect(args, cfg)
        return
    if args.analysis_target == "all":
        _run_all(cfg, config_path=args.config, out_dir_override=args.out_dir)
        return
    raise ValueError(f"Unknown analyze target: {args.analysis_target!r}")


def _add_erp_like_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs-glob", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--contrast", choices=["duration", "latency"], default=None)
    parser.add_argument("--min-latency", type=float, default=None)
    parser.add_argument("--max-latency", type=float, default=None)
    parser.add_argument("--min-response-duration", type=float, default=None)


def _add_tfr_like_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs-glob", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--contrast", choices=["duration", "latency"], default=None)
    parser.add_argument("--band", default=None)
    parser.add_argument("--min-latency", type=float, default=None)
    parser.add_argument("--max-latency", type=float, default=None)
    parser.add_argument("--min-response-duration", type=float, default=None)
    parser.add_argument("--sfreq", type=float, default=None)


def _add_decoding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs-glob", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--contrast", choices=["duration", "latency"], required=True)
    parser.add_argument("--sfreq", type=float, default=None)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--min-latency", type=float, default=None)
    parser.add_argument("--max-latency", type=float, default=None)
    parser.add_argument("--min-response-duration", type=float, default=None)


def _run_all(cfg: Any, *, config_path: str, out_dir_override: str | None) -> None:
    from turntaking.analysis.decoding.entry import run_decoding
    from turntaking.analysis.erp.entry import run_erp
    from turntaking.analysis.mixed_effect.entry import run_mixed_effect
    from turntaking.analysis.tfr.entry import run_tfr

    erp_args = argparse.Namespace(
        config=config_path,
        epochs_glob=None,
        out_dir=out_dir_override,
        contrast=None,
        min_latency=None,
        max_latency=None,
        min_response_duration=None,
    )
    run_erp(erp_args, cfg)

    tfr_args = argparse.Namespace(
        config=config_path,
        epochs_glob=None,
        out_dir=out_dir_override,
        contrast=None,
        band=None,
        min_latency=None,
        max_latency=None,
        min_response_duration=None,
        sfreq=None,
    )
    run_tfr(tfr_args, cfg)

    for contrast in list(cfg.analysis.contrasts):
        decoding_args = argparse.Namespace(
            config=config_path,
            epochs_glob=None,
            out_dir=out_dir_override,
            contrast=str(contrast),
            sfreq=None,
            n_splits=None,
            seed=None,
            n_jobs=None,
            cache_features=False,
            min_latency=None,
            max_latency=None,
            min_response_duration=None,
        )
        run_decoding(decoding_args, cfg)

    mixed_args = argparse.Namespace(config=config_path, out_csv=None)
    run_mixed_effect(mixed_args, cfg)
