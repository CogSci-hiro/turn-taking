import argparse
from dataclasses import dataclass
import glob
from pathlib import Path
from typing import List, Sequence

import mne


# ##################################################################################################
# ERP time-course viz command
# ##################################################################################################

@dataclass(frozen=True)
class ErpTimecourseVizConfig:
    duration_long_fif: Path
    duration_short_fif: Path
    latency_fast_fif: Path
    latency_slow_fif: Path
    out_base: Path
    xlim_ms: tuple[float, float]
    ylim_uv: tuple[float, float]


def _glob_sorted(pattern: str) -> List[Path]:
    matches = glob.glob(pattern)
    paths = sorted(Path(p) for p in matches)
    if len(paths) == 0:
        raise FileNotFoundError(f"No files matched glob pattern: {pattern!r}")
    return paths


def _load_evoked_list(path: Path) -> list[mne.Evoked]:
    if not path.exists():
        raise FileNotFoundError(f"Evoked file not found: {path}")

    evokeds = mne.read_evokeds(path, condition=None, verbose="ERROR")
    if len(evokeds) == 0:
        raise ValueError(f"No Evoked objects found in {path}")

    return evokeds


def _run_impl(cfg: ErpTimecourseVizConfig) -> None:
    long_path = cfg.duration_long_fif
    short_path = cfg.duration_short_fif
    fast_path = cfg.latency_fast_fif
    slow_path = cfg.latency_slow_fif

    long_list = _load_evoked_list(cfg.duration_long_fif)
    short_list = _load_evoked_list(cfg.duration_short_fif)
    fast_list = _load_evoked_list(cfg.latency_fast_fif)
    slow_list = _load_evoked_list(cfg.latency_slow_fif)

    # Optional sanity check: same number of subjects per contrast
    if len(long_list) != len(short_list):
        raise ValueError(f"duration long ({len(long_list)}) != short ({len(short_list)}) evoked counts")
    if len(fast_list) != len(slow_list):
        raise ValueError(f"latency fast ({len(fast_list)}) != slow ({len(slow_list)}) evoked counts")

    from turntaking.viz.figures.erp import plot_electrode_time_course
    from turntaking.viz._style import save_figure  # noqa
    from matplotlib import pyplot as plt

    fig = plot_electrode_time_course(
        long_list=long_list,
        short_list=short_list,
        fast_list=fast_list,
        slow_list=slow_list,
        xmin=cfg.xlim_ms[0],
        xmax=cfg.xlim_ms[1],
        ymin=cfg.ylim_uv[0],
        ymax=cfg.ylim_uv[1],
        figure_profile="jneuro_2col",
        save_basepath=cfg.out_base,  # let the function call save_figure()
    )

    plt.close(fig)


def run(args: argparse.Namespace, cfg) -> None:
    section = cfg.viz.erp_timecourse

    viz_cfg = ErpTimecourseVizConfig(
        duration_long_fif=Path(section.duration_long_fif),
        duration_short_fif=Path(section.duration_short_fif),
        latency_fast_fif=Path(section.latency_fast_fif),
        latency_slow_fif=Path(section.latency_slow_fif),
        out_base=Path(section.out_base),
        xlim_ms=(section.xlim_ms[0], section.xlim_ms[1]),
        ylim_uv=(section.ylim_uv[0], section.ylim_uv[1]),
    )

    _run_impl(viz_cfg)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "viz-erp-timecourse",
        help="Plot ERP time-course (Fz/Pz) for long/short and fast/slow.",
    )
    parser.add_argument("--config", required=True)
