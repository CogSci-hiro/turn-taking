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
    long_glob: str
    short_glob: str
    fast_glob: str
    slow_glob: str
    out_pdf: Path
    xlim_ms: tuple[float, float]
    ylim_uv: tuple[float, float]


def _glob_sorted(pattern: str) -> List[Path]:
    matches = glob.glob(pattern)
    paths = sorted(Path(p) for p in matches)
    if len(paths) == 0:
        raise FileNotFoundError(f"No files matched glob pattern: {pattern!r}")
    return paths


def _load_single_evoked(path: Path) -> mne.Evoked:
    evokeds = mne.read_evokeds(path, condition=None, verbose="ERROR")
    if len(evokeds) != 1:
        raise ValueError(f"Expected exactly 1 Evoked in {path}, found {len(evokeds)}")
    return evokeds[0]


def _load_evoked_list(paths: Sequence[Path]) -> List[mne.Evoked]:
    return [_load_single_evoked(p) for p in paths]


def _run_impl(cfg: ErpTimecourseVizConfig) -> None:
    long_paths = _glob_sorted(cfg.long_glob)
    short_paths = _glob_sorted(cfg.short_glob)
    fast_paths = _glob_sorted(cfg.fast_glob)
    slow_paths = _glob_sorted(cfg.slow_glob)

    if len(long_paths) != len(short_paths):
        raise ValueError(f"ERP long ({len(long_paths)}) != short ({len(short_paths)}) counts")
    if len(fast_paths) != len(slow_paths):
        raise ValueError(f"ERP fast ({len(fast_paths)}) != slow ({len(slow_paths)}) counts")

    long_list = _load_evoked_list(long_paths)
    short_list = _load_evoked_list(short_paths)
    fast_list = _load_evoked_list(fast_paths)
    slow_list = _load_evoked_list(slow_paths)

    from turntaking.viz.erp import plot_electrode_time_course

    fig = plot_electrode_time_course(
        long_list=long_list,
        short_list=short_list,
        fast_list=fast_list,
        slow_list=slow_list,
        xmin=cfg.xlim_ms[0],
        xmax=cfg.xlim_ms[1],
        ymin=cfg.ylim_uv[0],
        ymax=cfg.ylim_uv[1],
    )

    cfg.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.out_pdf, bbox_inches="tight", dpi=300)


def run(args: argparse.Namespace, cfg) -> None:
    section = cfg.viz.erp_timecourse

    viz_cfg = ErpTimecourseVizConfig(
        duration_long_fif=Path(section.duration_long_fif),
        duration_short_fif=Path(section.duration_short_fif),
        latency_fast_fif=Path(section.latency_fast_fif),
        latency_slow_fif=Path(section.latency_slow_fif),
        out_pdf=Path(section.out_pdf),
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
