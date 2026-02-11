import argparse
from dataclasses import dataclass
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
    paths = sorted(Path().glob(pattern))
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


def run(cfg: ErpTimecourseVizConfig) -> None:
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


def _handler(args: argparse.Namespace) -> None:
    # If you want to read from YAML/schema, replace this with:
    #   cfg = load_config(args.config).viz.erp_timecourse
    # and ignore CLI args except --config.
    cfg = ErpTimecourseVizConfig(
        long_glob=args.long_glob,
        short_glob=args.short_glob,
        fast_glob=args.fast_glob,
        slow_glob=args.slow_glob,
        out_pdf=Path(args.out),
        xlim_ms=(args.xmin, args.xmax),
        ylim_uv=(args.ymin, args.ymax),
    )
    run(cfg)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Required hook for turntaking.cli.main's dynamic command registration.
    """
    parser = subparsers.add_parser(
        "viz-erp-timecourse",
        help="Plot ERP time-course (Fz/Pz) for long/short and fast/slow.",
    )

    # If you want YAML-only, keep just --config and remove the others.
    parser.add_argument("--long_glob", required=True)
    parser.add_argument("--short_glob", required=True)
    parser.add_argument("--fast_glob", required=True)
    parser.add_argument("--slow_glob", required=True)
    parser.add_argument("--out", required=True, help="Output PDF path.")

    parser.add_argument("--xmin", type=float, default=-1500.0)
    parser.add_argument("--xmax", type=float, default=500.0)
    parser.add_argument("--ymin", type=float, default=-2.8)
    parser.add_argument("--ymax", type=float, default=1.9)

    # This is the key line: the dispatcher likely calls args.func(args)
    parser.set_defaults(func=_handler)
