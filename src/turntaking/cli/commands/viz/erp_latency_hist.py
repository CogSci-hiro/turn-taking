
# src/turntaking/cli/commands/viz/erp_latency_hist.py

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import pandas as pd

from turntaking.viz.figures.erp import plot_latency_erp_with_histograms


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

def run(args: argparse.Namespace, cfg) -> None:
    out_dir: Path = cfg.io.out_dir

    # 1) Load the evokeds you want to compare (fast vs slow)
    #
    # You need to match *your* actual file naming/layout.
    # Common pattern: store condition-specific evokeds under out_dir/erp/...
    fast_fif = out_dir / "erp" / "evoked_fast-ave.fif"
    slow_fif = out_dir / "erp" / "evoked_slow-ave.fif"

    if not fast_fif.exists():
        raise FileNotFoundError(f"Missing: {fast_fif}")
    if not slow_fif.exists():
        raise FileNotFoundError(f"Missing: {slow_fif}")

    fast_list = mne.read_evokeds(fast_fif, condition=None)
    slow_list = mne.read_evokeds(slow_fif, condition=None)

    # 2) Build the histogram dataframe expected by plot_latency_erp_with_histograms():
    # must include columns: latency, condition
    #
    # Again: adapt to your real table path.
    turn_table = out_dir / "beh" / "turn_table.csv"
    if not turn_table.exists():
        raise FileNotFoundError(f"Missing: {turn_table}")

    df = pd.read_csv(turn_table)

    # Minimal example: median split on latency into 'fast'/'slow'
    # (Adjust if you already have condition labels stored.)
    med = df["latency"].median()
    hist_df = pd.DataFrame(
        {
            "latency": df["latency"].to_numpy(),
            "condition": (df["latency"] <= med).map({True: "fast", False: "slow"}).to_numpy(),
        }
    )

    # 3) Plot + save
    plot_latency_erp_with_histograms(
        fast_list=fast_list,
        slow_list=slow_list,
        df=hist_df,
        ymax=float(args.ymax),
        save_basepath=args.out,
    )
