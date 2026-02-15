
# src/turntaking/cli/commands/viz/erp_latency_hist.py

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import pandas as pd

from turntaking.viz.figures.erp import plot_latency_erp_with_histograms

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, required: list[str], where: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {where}: {missing}. Available: {list(df.columns)}")


def _make_latency_hist_df(turn_table: pd.DataFrame) -> pd.DataFrame:
    """
    Create histogram dataframe with columns: latency, condition.

    Expects a 'latency' column in seconds (float).
    Uses median split into fast/slow.
    """
    _require_columns(turn_table, ["latency"], where="turn_table_csv")

    latency = pd.to_numeric(turn_table["latency"], errors="coerce").astype(float)
    latency = latency[np.isfinite(latency)]

    if latency.empty:
        raise ValueError("No finite latency values found in turn_table_csv.")

    cutoff = float(latency.median())
    condition = np.where(latency <= cutoff, "fast", "slow")

    return pd.DataFrame({"latency": latency.to_numpy(dtype=float), "condition": condition})



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

    turn_table = pd.read_csv(cfg.viz.behavior.turn_table_csv)
    hist_df = _make_latency_hist_df(turn_table)

    plot_latency_erp_with_histograms(
        fast_list=fast_list,
        slow_list=slow_list,
        df=hist_df,
        save_basepath=out_path,
    )
