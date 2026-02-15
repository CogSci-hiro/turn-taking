
# src/turntaking/cli/commands/viz/erp_latency_hist.py

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import pandas as pd

from turntaking.viz.figures.erp import plot_latency_erp_with_histograms

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, required: list[str], where: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {where}: {missing}. Available: {list(df.columns)}")


def _make_latency_hist_df_subject_median(mixed_table: pd.DataFrame) -> pd.DataFrame:
    required = {"subject", "latency"}
    missing = sorted(required - set(mixed_table.columns))
    if missing:
        raise KeyError(f"mixed_effect table is missing columns: {missing}. Available: {list(mixed_table.columns)}")

    df = mixed_table[["subject", "latency"]].copy()
    df["subject"] = df["subject"].astype(str)
    df["latency"] = pd.to_numeric(df["latency"], errors="coerce").astype(float)
    df = df[np.isfinite(df["latency"])]

    if df.empty:
        raise ValueError("No finite latency values in mixed_effect/table.csv after cleaning.")

    subj_med = df.groupby("subject", sort=False)["latency"].transform("median")
    df["condition"] = np.where(df["latency"] <= subj_med, "fast", "slow")

    return df[["latency", "condition"]].reset_index(drop=True)


# =============================================================================
# CLI REGISTRATION
# =============================================================================

def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Register `viz-erp-latency-hist`.

    Usage example
    -------------
        python -m turntaking.cli.main viz-erp-latency-hist \
          --config workflow/config.yaml \
          --out out/figures/supp/figS4_erp_timecourse_with_hist.png
    """
    p = subparsers.add_parser(
        "viz-erp-latency-hist",
        help="Plot latency ERP (fast vs slow) with latency histograms (Fz/Pz).",
    )
    p.add_argument("--config", type=Path, required=True, help="Path to workflow/config.yaml")
    p.add_argument("--out", type=Path, required=False, default=None,
                   help="Optional override for output path. If omitted, uses cfg.erp_hist.out_base.")
    p.add_argument("--ymax", type=float, default=2000, help="Histogram y-axis max.")
    p.set_defaults(func=run)


# =============================================================================
# Implementation
# =============================================================================

def _cfg_get(cfg, *keys):
    cur = cfg
    for k in keys:
        if isinstance(cur, dict):
            cur = cur[k]
        else:
            cur = getattr(cur, k)
    return cur

def run(args: argparse.Namespace, cfg) -> None:
    sec = cfg.viz.erp_hist

    out_path = Path(args.out) if getattr(args, "out", None) else sec.out_base

    fast_list = mne.read_evokeds(sec.latency_fast_fif, condition=None)
    slow_list = mne.read_evokeds(sec.latency_slow_fif, condition=None)

    mixed_table = pd.read_csv(cfg.viz.erp_hist.hist_table_csv)
    hist_df = _make_latency_hist_df_subject_median(mixed_table)

    plot_latency_erp_with_histograms(
        fast_list=fast_list,
        slow_list=slow_list,
        df=hist_df,
        save_basepath=out_path,
    )
