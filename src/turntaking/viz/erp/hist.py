"""ERP latency histogram visualization."""


from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from turntaking.viz._style import apply_style
from turntaking.viz.figures.erp import plot_latency_erp_with_histograms
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


def _latency_hist_table(mixed_table: pd.DataFrame) -> pd.DataFrame:
    required = {"subject", "latency"}
    missing = sorted(required - set(mixed_table.columns))
    if missing:
        raise KeyError(f"mixed_effect table missing columns: {missing}.")

    data = mixed_table[["subject", "latency"]].copy()
    data["subject"] = data["subject"].astype(str)
    data["latency"] = pd.to_numeric(data["latency"], errors="coerce").astype(float)
    data = data[np.isfinite(data["latency"])]
    if data.empty:
        raise ValueError("No finite latency values available.")

    subject_median = data.groupby("subject", sort=False)["latency"].transform("median")
    data["condition"] = np.where(data["latency"] <= subject_median, "fast", "slow")
    return data[["latency", "condition"]].reset_index(drop=True)


def render(cfg: Any) -> None:
    apply_style("jneuro_2col")
    fast_path = _erp_artifact(cfg, "latency", "fast", "erp/latency/fast_ave.fif")
    slow_path = _erp_artifact(cfg, "latency", "slow", "erp/latency/slow_ave.fif")
    table_path = resolve_from_out_dir(cfg, "mixed_effect/table.csv")
    out_path = resolve_from_out_dir(cfg, "figures/main/F_erp_timecourse_hist")

    fast_list = mne.read_evokeds(fast_path, condition=None, verbose="ERROR")
    slow_list = mne.read_evokeds(slow_path, condition=None, verbose="ERROR")
    mixed_table = pd.read_csv(table_path)
    hist_df = _latency_hist_table(mixed_table)
    plot_latency_erp_with_histograms(
        fast_list=fast_list,
        slow_list=slow_list,
        df=hist_df,
        save_basepath=out_path,
    )
