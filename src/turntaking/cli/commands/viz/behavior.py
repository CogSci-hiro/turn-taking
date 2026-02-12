import argparse
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt


# ##################################################################################################
# Behavior viz command
# ##################################################################################################

@dataclass(frozen=True)
class BehaviorVizConfig:
    duration_offsets_csv: Path
    latency_offsets_csv: Path
    out_main: Path
    out_s1: Path
    out_s2: Path
    out_s3_long: Path
    out_s3_short: Path
    out_s3_fast: Path
    out_s3_slow: Path
    n_bins: int
    figure_profile: str


def _run_impl(cfg: BehaviorVizConfig) -> None:
    for p in [
        cfg.out_main,
        cfg.out_s1,
        cfg.out_s2,
        cfg.out_s3_long,
        cfg.out_s3_short,
        cfg.out_s3_fast,
        cfg.out_s3_slow,
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)

    from turntaking.viz.figures.behavior import make_behavior_figures

    make_behavior_figures(
        duration_offsets_csv=cfg.duration_offsets_csv,
        latency_offsets_csv=cfg.latency_offsets_csv,
        out_main=cfg.out_main,
        out_s1=cfg.out_s1,
        out_s2=cfg.out_s2,
        out_s3_long=cfg.out_s3_long,
        out_s3_short=cfg.out_s3_short,
        out_s3_fast=cfg.out_s3_fast,
        out_s3_slow=cfg.out_s3_slow,
        n_bins=cfg.n_bins,
        figure_profile=cfg.figure_profile,
    )

    # make_behavior_figures already closes figures, but keep CLI clean anyway
    plt.close("all")


def run(args: argparse.Namespace, cfg) -> None:
    """
    Expected config keys (suggested):

    cfg.io.out_dir
    cfg.viz.behavior.duration_offsets_csv
    cfg.viz.behavior.latency_offsets_csv
    """
    out_dir = Path(cfg.io.out_dir)

    section = cfg.viz.behavior

    viz_cfg = BehaviorVizConfig(
        duration_offsets_csv=Path(section.duration_offsets_csv),
        latency_offsets_csv=Path(section.latency_offsets_csv),
        out_main=out_dir / "figures" / "main" / "F_behavior.tif",
        out_s1=out_dir / "figures" / "supp" / "S1_response_duration_hist.tif",
        out_s2=out_dir / "figures" / "supp" / "S2_previous_speech_duration_hist.tif",
        out_s3_long=out_dir / "figures" / "supp" / "S3_long_joint.tif",
        out_s3_short=out_dir / "figures" / "supp" / "S3_short_joint.tif",
        out_s3_fast=out_dir / "figures" / "supp" / "S3_fast_joint.tif",
        out_s3_slow=out_dir / "figures" / "supp" / "S3_slow_joint.tif",
        n_bins=int(getattr(section, "n_bins", 100)),
        figure_profile=str(getattr(section, "figure_profile", "jneuro_2col")),
    )

    _run_impl(viz_cfg)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "viz-behavior",
        help="Build behavior figures (Fig1 + S1–S3) from offsets.csv files.",
    )
    parser.add_argument("--config", required=True)
