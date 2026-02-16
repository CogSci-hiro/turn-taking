
"""Domain service for ERP/TFR cluster permutation analyses."""

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import mne
import numpy as np

from turntaking.stats.cluster_test import ClusterTestParams, run_cluster_1samp_spatiotemporal
from turntaking.stats.cropping import crop_time_margins_samples

Kind = Literal["erp", "tfr"]


def run_cluster(args: Any, cfg: Any) -> None:
    kind: Kind = args.kind
    contrast: str = str(args.contrast)
    band: str | None = getattr(args, "band", None)
    if kind == "tfr" and not band:
        raise ValueError("--band is required when --kind tfr")

    out_root = _load_out_root(cfg)
    params = _load_params(cfg, args)
    left_margin, right_margin, sfreq_cfg = _load_crop_settings(cfg, kind)
    X, info, data_tmin_s_orig, stats_out_dir = _load_data(kind, out_root, contrast, band)
    sfreq = float(sfreq_cfg) if sfreq_cfg is not None else float(info["sfreq"])

    X_cropped, start_idx, end_idx = crop_time_margins_samples(
        X,
        sfreq=sfreq,
        left_margin=left_margin,
        right_margin=right_margin,
    )
    data_tmin_s = float(data_tmin_s_orig + (float(start_idx) / float(sfreq)))
    result = run_cluster_1samp_spatiotemporal(
        X_cropped,
        info=info,
        params=params,
        kind=kind,
        data_tmin_s=data_tmin_s,
        sfreq_hz=float(sfreq),
    )
    _augment_metadata(result.metadata, left_margin, right_margin, sfreq, start_idx, end_idx, band)
    _write_cluster_outputs(kind, stats_out_dir, result)


def _load_data(
    kind: Kind,
    out_root: Path,
    contrast: str,
    band: str | None,
) -> tuple[np.ndarray, mne.Info, float, Path]:
    if kind == "erp":
        X, info, tmin = _load_erp_X_info_tmin(out_root, contrast=contrast)
        return X, info, tmin, out_root / "stats" / "erp" / contrast
    assert band is not None
    X, info, tmin = _load_tfr_X_info_tmin(out_root, contrast=contrast, band=band)
    return X, info, tmin, out_root / "stats" / "tfr" / contrast / band


def _load_params(cfg: Any, args: Any) -> ClusterTestParams:
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
    return _apply_cli_overrides(params, args)


def _apply_cli_overrides(params: ClusterTestParams, args: Any) -> ClusterTestParams:
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


def _load_crop_settings(cfg: Any, kind: Kind) -> tuple[float, float, float | None]:
    analysis = getattr(cfg, "analysis", None)
    if analysis is None:
        raise ValueError("Config missing 'analysis' section (cfg.analysis).")
    section = getattr(analysis, kind, None)
    if section is None:
        raise ValueError(f"Config missing analysis.{kind} section.")
    left_margin = float(getattr(section, "left_margin", 0.0))
    right_margin = float(getattr(section, "right_margin", 0.0))
    sfreq_raw = getattr(section, "sfreq", None)
    sfreq = float(sfreq_raw) if sfreq_raw is not None else None
    if sfreq is not None and sfreq <= 0:
        sfreq = None
    return left_margin, right_margin, sfreq


def _load_out_root(cfg: Any) -> Path:
    io_cfg = getattr(cfg, "io", None)
    if io_cfg is None:
        raise ValueError("Config missing 'io' section (cfg.io).")
    return Path(getattr(io_cfg, "out_dir"))


def _write_cluster_outputs(kind: Kind, out_dir: Path, result: Any) -> None:
    if kind == "erp":
        from turntaking.analysis.erp.io import write_cluster_outputs

        write_cluster_outputs(out_dir, result)
        return
    from turntaking.analysis.tfr.io import write_cluster_outputs

    write_cluster_outputs(out_dir, result)


def _augment_metadata(
    metadata: dict[str, Any],
    left_margin: float,
    right_margin: float,
    sfreq: float,
    start_idx: int,
    end_idx: int,
    band: str | None,
) -> None:
    metadata["crop_left_margin"] = float(left_margin)
    metadata["crop_right_margin"] = float(right_margin)
    metadata["crop_sfreq_used"] = float(sfreq)
    metadata["crop_start_idx"] = int(start_idx)
    metadata["crop_end_idx"] = int(end_idx)
    if band is not None:
        metadata["band"] = str(band)


def _load_erp_X_info_tmin(out_root: Path, *, contrast: str) -> tuple[np.ndarray, mne.Info, float]:
    evoked_data_path = out_root / "erp" / contrast / "evoked-data.npy"
    diff_ave_path = out_root / "erp" / contrast / "difference_ave.fif"
    arr = np.load(evoked_data_path)
    if arr.ndim != 4 or arr.shape[1] != 3:
        raise ValueError(f"Unexpected ERP evoked-data.npy shape: {arr.shape} (expected (N,3,C,T)).")
    X = np.transpose(arr[:, 2, :, :], (0, 2, 1)).astype(float)
    evoked = mne.read_evokeds(diff_ave_path, condition=0)
    return X, evoked.info, float(evoked.times[0])


def _load_tfr_X_info_tmin(out_root: Path, *, contrast: str, band: str) -> tuple[np.ndarray, mne.Info, float]:
    induced_data_path = out_root / "tfr" / contrast / band / "induced-data.npy"
    diff_ave_path = out_root / "tfr" / contrast / band / "difference_ave.fif"
    arr = np.load(induced_data_path)
    if arr.ndim != 4 or arr.shape[1] != 3:
        raise ValueError(f"Unexpected TFR induced-data.npy shape: {arr.shape} (expected (N,3,C,T)).")
    X = np.transpose(arr[:, 2, :, :], (0, 2, 1)).astype(float)
    evoked = mne.read_evokeds(diff_ave_path, condition=0)
    return X, evoked.info, float(evoked.times[0])
