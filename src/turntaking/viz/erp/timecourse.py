"""ERP time-course visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
from matplotlib import pyplot as plt

from turntaking.viz._style import apply_style
from turntaking.viz.figures.erp import plot_electrode_time_course
from turntaking.viz.utils import cfg_get_optional, resolve_from_out_dir


def _erp_artifact(cfg: Any, contrast: str, condition: str, fallback: str) -> Path:
    relative = cfg_get_optional(
        cfg,
        "analysis",
        "erp",
        "artifacts",
        contrast,
        condition,
        default=fallback,
    )
    return resolve_from_out_dir(cfg, relative)


def _viz_range(cfg: Any, key: str, fallback: tuple[float, float]) -> tuple[float, float]:
    value = cfg_get_optional(cfg, "viz", "erp", "timecourse", key, default=None)
    if value is None:
        value = cfg_get_optional(cfg, "viz", "erp_timecourse", key, default=fallback)
    return float(value[0]), float(value[1])


def _load_evokeds(path: Path) -> list[mne.Evoked]:
    if not path.exists():
        raise FileNotFoundError(f"Evoked file not found: {path}")
    evokeds = mne.read_evokeds(path, condition=None, verbose="ERROR")
    if not evokeds:
        raise ValueError(f"No evoked entries in {path}")
    return evokeds


def render(cfg: Any) -> None:
    apply_style("jneuro_2col")
    long_list = _load_evokeds(_erp_artifact(cfg, "duration", "long", "erp/duration/long_ave.fif"))
    short_list = _load_evokeds(_erp_artifact(cfg, "duration", "short", "erp/duration/short_ave.fif"))
    fast_list = _load_evokeds(_erp_artifact(cfg, "latency", "fast", "erp/latency/fast_ave.fif"))
    slow_list = _load_evokeds(_erp_artifact(cfg, "latency", "slow", "erp/latency/slow_ave.fif"))
    xlim = _viz_range(cfg, "xlim_ms", (-1500.0, 500.0))
    ylim = _viz_range(cfg, "ylim_uv", (-2.8, 1.9))
    out_path = resolve_from_out_dir(cfg, "figures/main/F_erp_timecourse")

    fig = plot_electrode_time_course(
        long_list=long_list,
        short_list=short_list,
        fast_list=fast_list,
        slow_list=slow_list,
        xmin=xlim[0],
        xmax=xlim[1],
        ymin=ylim[0],
        ymax=ylim[1],
        figure_profile="jneuro_2col",
        save_basepath=out_path,
    )
    plt.close(fig)
