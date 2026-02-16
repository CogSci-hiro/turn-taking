
"""Service entrypoint for behavior figure generation."""

from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt


@dataclass(frozen=True)
class BehaviorVizConfig:
    """Resolved input/output paths and render parameters for behavior figures."""

    duration_offsets_csv: Path
    latency_offsets_csv: Path
    turn_table_csv: Path
    out_main: Path
    out_s1: Path
    out_s2: Path
    out_s3_long: Path
    out_s3_short: Path
    out_s3_fast: Path
    out_s3_slow: Path
    n_bins: int
    figure_profile: str


def run_behavior_viz(cfg: BehaviorVizConfig) -> None:
    """Render the behavior figure set and close Matplotlib state."""
    for out_path in (
        cfg.out_main,
        cfg.out_s1,
        cfg.out_s2,
        cfg.out_s3_long,
        cfg.out_s3_short,
        cfg.out_s3_fast,
        cfg.out_s3_slow,
    ):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    from turntaking.viz.figures.behavior import make_behavior_figures

    make_behavior_figures(
        duration_offsets_csv=cfg.duration_offsets_csv,
        latency_offsets_csv=cfg.latency_offsets_csv,
        turn_table_csv=cfg.turn_table_csv,
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
    plt.close("all")
