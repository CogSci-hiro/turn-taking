"""Thin CLI wrapper for visualization entrypoints."""


import argparse
from pathlib import Path
from typing import Any

from turntaking.analysis.viz.behavior_entry import BehaviorVizConfig, run_behavior_viz
from turntaking.viz.decoding.entry import run as run_decoding_viz
from turntaking.viz.erp.entry import run as run_erp_viz
from turntaking.viz.tfr.entry import run as run_tfr_viz
from turntaking.viz.utils import resolve_from_out_dir


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("viz", help="Run visualization entrypoints.")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config.")
    viz_subparsers = parser.add_subparsers(dest="viz_family", required=True)

    erp = viz_subparsers.add_parser("erp", help="Run ERP visualization.")
    erp.add_argument("--mode", choices=["timecourse", "hist", "topomap"], default=None)
    erp.add_argument("--format", choices=["static", "svg"], default=None)

    tfr = viz_subparsers.add_parser("tfr", help="Run TFR visualization.")
    tfr.add_argument("--mode", choices=["topomap"], default=None)
    tfr.add_argument("--format", choices=["static", "svg"], default=None)

    decoding = viz_subparsers.add_parser("decoding", help="Run decoding visualization.")
    decoding.add_argument("--mode", choices=["figure"], default=None)

    viz_subparsers.add_parser("behavior", help="Run behavior visualization.")
    viz_subparsers.add_parser("main", help="Run all main visualization figures.")
    viz_subparsers.add_parser("supp", help="Run supplementary visualization figures.")


def _run_behavior(cfg: Any) -> None:
    section = cfg.viz.behavior
    viz_cfg = BehaviorVizConfig(
        duration_offsets_csv=resolve_from_out_dir(cfg, "erp/duration/offsets.csv"),
        latency_offsets_csv=resolve_from_out_dir(cfg, "erp/latency/offsets.csv"),
        turn_table_csv=resolve_from_out_dir(cfg, "beh/turn_table.csv"),
        out_main=resolve_from_out_dir(cfg, "figures/main/F_behavior.tif"),
        out_s1=resolve_from_out_dir(cfg, "figures/supp/S1_response_duration_hist.tif"),
        out_s2=resolve_from_out_dir(cfg, "figures/supp/S2_previous_speech_duration_hist.tif"),
        out_s3_long=resolve_from_out_dir(cfg, "figures/supp/S3_long_joint.tif"),
        out_s3_short=resolve_from_out_dir(cfg, "figures/supp/S3_short_joint.tif"),
        out_s3_fast=resolve_from_out_dir(cfg, "figures/supp/S3_fast_joint.tif"),
        out_s3_slow=resolve_from_out_dir(cfg, "figures/supp/S3_slow_joint.tif"),
        n_bins=int(getattr(section, "n_bins", 100)),
        figure_profile=str(getattr(section, "figure_profile", "jneuro_2col")),
    )
    run_behavior_viz(viz_cfg)


def run(args: argparse.Namespace, cfg: Any) -> None:
    if args.viz_family == "erp":
        run_erp_viz(cfg, mode=args.mode, topomap_format=args.format)
        return
    if args.viz_family == "tfr":
        run_tfr_viz(cfg, mode=args.mode, topomap_format=args.format)
        return
    if args.viz_family == "decoding":
        run_decoding_viz(cfg, mode=args.mode)
        return
    if args.viz_family == "behavior":
        _run_behavior(cfg)
        return
    if args.viz_family == "main":
        _run_behavior(cfg)
        run_erp_viz(cfg, mode="timecourse")
        run_erp_viz(cfg, mode="topomap", topomap_format="svg")
        run_tfr_viz(cfg, mode="topomap", topomap_format="svg")
        run_decoding_viz(cfg, mode="figure")
        return
    if args.viz_family == "supp":
        run_erp_viz(cfg, mode="hist")
        run_erp_viz(cfg, mode="topomap", topomap_format="static")
        run_tfr_viz(cfg, mode="topomap", topomap_format="static")
        return
    raise ValueError(f"Unsupported viz family: {args.viz_family!r}")
