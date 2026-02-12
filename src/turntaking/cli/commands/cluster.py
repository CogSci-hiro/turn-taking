# src/turntaking/cli/commands/cluster.py


import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import mne
import numpy as np

from turntaking.analysis.io.cluster import write_cluster_outputs
from turntaking.stats.cluster_test import ClusterTestParams, run_cluster_1samp_spatiotemporal
from turntaking.stats.cropping import crop_time_margins_samples

Kind = Literal["erp", "tfr"]


# =============================================================================
#                     ########################################
#                     #            CLI REGISTRATION           #
#                     ########################################
# =============================================================================
def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Register the `cluster` command.

    Usage example
    -------------
        python -m turntaking.cli.main cluster --config workflow/config.yaml --kind erp --contrast latency
        python -m turntaking.cli.main cluster --config workflow/config.yaml --kind tfr --contrast latency --band alpha
    """
    p = subparsers.add_parser(
        "cluster",
        help="Run cluster permutation tests (ERP/TFR).",
    )

    p.add_argument("--config", required=True, help="Path to YAML config file.")
    p.add_argument("--kind", required=True, choices=["erp", "tfr"], help="Which analysis kind to test.")
    p.add_argument("--contrast", required=True, help="Contrast name (e.g. latency, duration).")

    # TFR-only selector (required by your Snakemake rule for TFR clusters)
    p.add_argument(
        "--band",
        default=None,
        help="Band name (required when --kind tfr), e.g. alpha, beta.",
    )

    # Optional overrides (if omitted, values are read from config.yaml)
    p.add_argument("--n-permutations", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--tail", type=int, choices=[-1, 0, 1], default=None)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--ch-type", choices=["eeg", "mag", "grad"], default=None)


# =============================================================================
#                     ########################################
#                     #              CONFIG READ              #
#                     ########################################
# =============================================================================
def _load_params(cfg: Any, args: Any) -> ClusterTestParams:
    """
    Load cluster-test params from TurntakingConfig (attribute-based).

    Reads from:
      - cfg.analysis.erp.*  when args.kind == "erp"
      - cfg.analysis.tfr.*  when args.kind == "tfr"

    Expected YAML fields (matching your config.yaml)
    -----------------------------------------------
    analysis:
      erp:
        n_permutations: 1000
        threshold: null
        left_margin: 0.2
        right_margin: 0.5
        sfreq: 512
      tfr:
        n_permutations: 1000
        threshold: null
        left_margin: 0.2
        right_margin: 0.5
        sfreq: 512
    """
    analysis = getattr(cfg, "analysis", None)
    if analysis is None:
        raise ValueError("Config missing 'analysis' section (cfg.analysis).")

    section = getattr(analysis, args.kind, None)
    if section is None:
        raise ValueError(f"Config missing analysis.{args.kind} section.")

    params = ClusterTestParams(
        n_permutations=int(getattr(section, "n_permutations", 1024)),
        threshold=getattr(section, "threshold", None),
        tail=int(getattr(section, "tail", 0)),
        alpha=float(getattr(section, "alpha", 0.05)),
        seed=int(getattr(section, "seed", 0)),
        n_jobs=int(getattr(section, "n_jobs", 1)),
        ch_type=str(getattr(section, "ch_type", "eeg")),
    )

    # CLI overrides
    if args.n_permutations is not None:
        params = replace(params, n_permutations=int(args.n_permutations))
    if args.threshold is not None:
        params = replace(params, threshold=float(args.threshold))
    if args.tail is not None:
        params = replace(params, tail=int(args.tail))
    if args.n_jobs is not None:
        params = replace(params, n_jobs=int(args.n_jobs))
    if args.seed is not None:
        params = replace(params, seed=int(args.seed))
    if args.ch_type is not None:
        params = replace(params, ch_type=str(args.ch_type))

    return params


def _load_crop_settings(cfg: Any, kind: Kind) -> tuple[float, float, float]:
    """Load (left_margin, right_margin, sfreq) from cfg.analysis.{kind}."""
    analysis = getattr(cfg, "analysis", None)
    if analysis is None:
        raise ValueError("Config missing 'analysis' section (cfg.analysis).")

    section = getattr(analysis, kind, None)
    if section is None:
        raise ValueError(f"Config missing analysis.{kind} section.")

    left_margin = float(getattr(section, "left_margin"))
    right_margin = float(getattr(section, "right_margin"))
    sfreq = float(getattr(section, "sfreq"))
    return left_margin, right_margin, sfreq


def _load_out_root(cfg: Any) -> Path:
    io_cfg = getattr(cfg, "io", None)
    if io_cfg is None:
        raise ValueError("Config missing 'io' section (cfg.io).")
    return Path(getattr(io_cfg, "out_dir"))


def _load_erp_X_info_tmin(
    out_root: Path,
    *,
    contrast: str,
) -> tuple[np.ndarray, mne.Info, float]:
    evoked_data_path = out_root / "erp" / contrast / "evoked-data.npy"
    diff_ave_path = out_root / "erp" / contrast / "difference_ave.fif"

    arr = np.load(evoked_data_path)  # (N,3,C,T)
    if arr.ndim != 4 or arr.shape[1] != 3:
        raise ValueError(f"Unexpected ERP evoked-data.npy shape: {arr.shape} (expected (N,3,C,T)).")

    diff = arr[:, 2, :, :]  # (N,C,T)
    X = np.transpose(diff, (0, 2, 1)).astype(float)  # (N,T,C)

    evoked = mne.read_evokeds(diff_ave_path, condition=0)
    tmin_s = float(evoked.times[0])
    return X, evoked.info, tmin_s


def _load_tfr_X_info_tmin(
    out_root: Path,
    *,
    contrast: str,
    band: str,
) -> tuple[np.ndarray, mne.Info, float]:
    induced_data_path = out_root / "tfr" / contrast / band / "induced-data.npy"
    diff_ave_path = out_root / "tfr" / contrast / band / "difference_ave.fif"

    arr = np.load(induced_data_path)  # (N,3,C,T)
    if arr.ndim != 4 or arr.shape[1] != 3:
        raise ValueError(f"Unexpected TFR induced-data.npy shape: {arr.shape} (expected (N,3,C,T)).")

    diff = arr[:, 2, :, :]  # (N,C,T)
    X = np.transpose(diff, (0, 2, 1)).astype(float)  # (N,T,C)

    evoked = mne.read_evokeds(diff_ave_path, condition=0)
    tmin_s = float(evoked.times[0])
    return X, evoked.info, tmin_s


# =============================================================================
#                     ########################################
#                     #                 RUN                   #
#                     ########################################
# =============================================================================
def run(args: Any, cfg: Any) -> None:
    """
    Run cluster permutation tests and write stats artifacts.

    ERP inputs:
      {out_dir}/erp/{contrast}/evoked-data.npy         shape (N,3,C,T)
      {out_dir}/erp/{contrast}/difference_ave.fif

    TFR inputs:
      {out_dir}/tfr/{contrast}/{band}/induced-data.npy shape (N,3,C,T)
      {out_dir}/tfr/{contrast}/{band}/difference_ave.fif

    Outputs:
      ERP: {out_dir}/stats/erp/{contrast}/cluster_results.hdf5
           {out_dir}/stats/erp/{contrast}/cluster_summary.csv

      TFR: {out_dir}/stats/tfr/{contrast}/{band}/cluster_results.hdf5
           {out_dir}/stats/tfr/{contrast}/{band}/cluster_summary.csv
    """
    kind: Kind = args.kind
    contrast: str = str(args.contrast)
    band: str | None = getattr(args, "band", None)

    if kind == "tfr" and not band:
        raise ValueError("--band is required when --kind tfr")

    out_root = _load_out_root(cfg)
    params = _load_params(cfg, args)
    left_margin, right_margin, sfreq = _load_crop_settings(cfg, kind)

    # Load X + info
    if kind == "erp":
        X, info, data_tmin_s_orig = _load_erp_X_info_tmin(out_root, contrast=contrast)
        stats_out_dir = out_root / "stats" / "erp" / contrast
    else:
        assert band is not None
        X, info, data_tmin_s_orig = _load_tfr_X_info_tmin(out_root, contrast=contrast, band=band)
        stats_out_dir = out_root / "stats" / "tfr" / contrast / band

    print(
        f"[cluster] kind={kind} contrast={contrast}"
        f"{'' if band is None else f' band={band}'} "
        f"n_permutations={params.n_permutations} threshold={params.threshold} tail={params.tail} "
        f"left_margin={left_margin} right_margin={right_margin} sfreq={sfreq}"
    )

    # Crop margins (legacy-compatible)
    X_cropped, start_idx, end_idx = crop_time_margins_samples(
        X,
        sfreq=sfreq,
        left_margin=left_margin,
        right_margin=right_margin,
    )

    data_tmin_s = float(data_tmin_s_orig + (float(start_idx) / float(sfreq)))

    # Run cluster test (identical for ERP and band-averaged induced TFR)
    result = run_cluster_1samp_spatiotemporal(
        X_cropped,
        info=info,
        params=params,
        kind=kind,
        data_tmin_s=data_tmin_s,
        sfreq_hz=float(sfreq),
    )

    # Record crop metadata for reproducibility
    result.metadata["crop_left_margin"] = float(left_margin)
    result.metadata["crop_right_margin"] = float(right_margin)
    result.metadata["crop_sfreq_used"] = float(sfreq)
    result.metadata["crop_start_idx"] = int(start_idx)
    result.metadata["crop_end_idx"] = int(end_idx)

    if band is not None:
        result.metadata["band"] = str(band)

    write_cluster_outputs(stats_out_dir, result)
