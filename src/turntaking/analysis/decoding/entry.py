from __future__ import annotations

"""Decoding command service: resolves config/overrides and runs group decoding."""

import argparse
import glob
import re
from pathlib import Path
from typing import Any, Literal

from turntaking.analysis.decoding.dataset import DecodingDatasetParams
from turntaking.analysis.decoding.io import Hdf5CacheParams, load_subject_feature_cache_hdf5, save_decoding_scores, save_subject_feature_cache_hdf5
from turntaking.analysis.decoding.run_decoding import DecodingRunParams, run_group_decoding
from turntaking.analysis.selection import SelectionParams
from turntaking.analysis.utils.epochs import load_subject_epochs

Contrast = Literal["latency", "duration"]
_SUBJECT_RE = re.compile(r"(sub-\d{3})")


def run_decoding(args: argparse.Namespace, cfg: Any) -> None:
    epoch_dir = Path(_cfg_get(cfg, "io", "epoch_dir"))
    _resolve_epoch_paths_for_validation(args, cfg)
    subjects = _expand_subjects_from_config(cfg)
    if len(subjects) == 0:
        raise RuntimeError("No subjects resolved from config (after include/exclude).")
    base_out_dir = Path(args.out_dir) if args.out_dir else Path(_cfg_get(cfg, "io", "out_dir"))
    base_out_dir.mkdir(parents=True, exist_ok=True)
    dataset_params = _resolve_dataset_params(args, cfg)
    run_params = _resolve_run_params(args, cfg)
    load_cached_features_fn, save_cached_features_fn = _resolve_cache_io(args, cfg, base_out_dir)
    scores, times_s = run_group_decoding(
        subjects=subjects,
        epoch_dir=epoch_dir,
        dataset_params=dataset_params,
        run_params=run_params,
        load_subject_epochs_fn=load_subject_epochs,
        load_cached_features_fn=load_cached_features_fn,
        save_cached_features_fn=save_cached_features_fn,
    )
    save_decoding_scores(
        out_dir=base_out_dir,
        contrast=str(args.contrast),  # type: ignore[arg-type]
        scores=scores,
        times_s=times_s,
    )


def _discover_subjects_from_epochs(epoch_dir: Path) -> list[str]:
    subjects: set[str] = set()
    for path in epoch_dir.glob("sub-*_task-*_run-*_epochs-epo.fif"):
        match = _SUBJECT_RE.search(path.name)
        if match:
            subjects.add(match.group(1))
    return sorted(subjects)


def _cfg_get(cfg: Any, *keys: str) -> Any:
    current: Any = cfg
    for key in keys:
        current = current[key] if isinstance(current, dict) else getattr(current, key)
    return current


def _expand_subjects_from_config(cfg: Any) -> list[str]:
    epoch_dir = Path(_cfg_get(cfg, "io", "epoch_dir"))
    subjects_cfg = _cfg_get(cfg, "dataset", "subjects")
    mode = subjects_cfg.get("mode") if isinstance(subjects_cfg, dict) else getattr(subjects_cfg, "mode")
    exclude = set(subjects_cfg.get("exclude", [])) if isinstance(subjects_cfg, dict) else set(getattr(subjects_cfg, "exclude", []))
    include = list(subjects_cfg.get("include", [])) if isinstance(subjects_cfg, dict) else list(getattr(subjects_cfg, "include", []))
    if mode == "from_epochs":
        subjects = _discover_subjects_from_epochs(epoch_dir)
    elif mode == "explicit":
        subjects = include
    else:
        raise ValueError(f"Unsupported dataset.subjects.mode: {mode!r}")
    return sorted([subject for subject in subjects if subject not in exclude])


def _expand_epoch_paths_from_config(cfg: Any) -> list[Path]:
    epoch_dir = Path(_cfg_get(cfg, "io", "epoch_dir"))
    pattern = str(_cfg_get(cfg, "io", "epoch_pattern"))
    tasks = list(_cfg_get(cfg, "dataset", "tasks"))
    runs = [str(run) for run in _cfg_get(cfg, "dataset", "runs")]
    subjects = _expand_subjects_from_config(cfg)
    invalid_pairs = {(str(pair[0]), str(pair[1])) for pair in _cfg_get(cfg, "dataset", "invalid_subject_run")}
    paths: list[Path] = []
    for subject in subjects:
        for task in tasks:
            for run in runs:
                if (subject, run) in invalid_pairs:
                    continue
                path = epoch_dir / pattern.format(subject=subject, task=task, run=run)
                if path.is_file():
                    paths.append(path)
    paths = sorted(paths)
    if len(paths) == 0:
        raise RuntimeError(
            "No epoch files found after config expansion. "
            f"epoch_dir={epoch_dir}, pattern={pattern}"
        )
    return paths


def _resolve_epoch_paths_for_validation(args: argparse.Namespace, cfg: Any) -> list[Path]:
    if args.epochs_glob:
        epoch_paths = [Path(path) for path in sorted(glob.glob(args.epochs_glob)) if Path(path).is_file()]
        if len(epoch_paths) == 0:
            raise RuntimeError(f"No files matched --epochs-glob: {args.epochs_glob}")
        return epoch_paths
    return _expand_epoch_paths_from_config(cfg)


def _resolve_dataset_params(args: argparse.Namespace, cfg: Any) -> DecodingDatasetParams:
    min_latency = args.min_latency if args.min_latency is not None else float(_cfg_get(cfg, "constraints", "min_latency"))
    max_latency = args.max_latency if args.max_latency is not None else float(_cfg_get(cfg, "constraints", "max_latency"))
    min_self_duration = (
        args.min_response_duration
        if args.min_response_duration is not None
        else float(_cfg_get(cfg, "constraints", "min_response_duration"))
    )
    sfreq_hz = float(args.sfreq) if args.sfreq is not None else float(_cfg_get(cfg, "analysis", "decoding", "sfreq"))
    selection_params = SelectionParams(
        min_latency=min_latency,
        max_latency=max_latency,
        min_self_duration=min_self_duration,
    )
    return DecodingDatasetParams(
        contrast=str(args.contrast),  # type: ignore[arg-type]
        selection=selection_params,
        sfreq_hz=sfreq_hz,
    )


def _resolve_run_params(args: argparse.Namespace, cfg: Any) -> DecodingRunParams:
    decoding_cfg = _cfg_get(cfg, "analysis", "decoding")
    n_splits = int(args.n_splits) if args.n_splits is not None else int(_cfg_get(cfg, "analysis", "decoding", "n_splits"))
    seed = int(args.seed) if args.seed is not None else int(decoding_cfg.get("seed", 0) if isinstance(decoding_cfg, dict) else getattr(decoding_cfg, "seed", 0))
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else int(decoding_cfg.get("n_jobs", 10) if isinstance(decoding_cfg, dict) else getattr(decoding_cfg, "n_jobs", 10))
    return DecodingRunParams(n_splits=n_splits, seed=seed, n_jobs=n_jobs)


def _resolve_cache_io(args: argparse.Namespace, cfg: Any, out_dir: Path):
    if not args.cache_features:
        return None, None
    cache_params = _cache_params(cfg)

    def _load(subject: str):
        return load_subject_feature_cache_hdf5(
            out_dir=out_dir,
            contrast=str(args.contrast),  # type: ignore[arg-type]
            subject=subject,
        )

    def _save(subject: str, X, y, times_s):
        save_subject_feature_cache_hdf5(
            out_dir=out_dir,
            contrast=str(args.contrast),  # type: ignore[arg-type]
            subject=subject,
            X=X,
            y=y,
            times_s=times_s,
            cache_params=cache_params,
        )

    return _load, _save


def _cache_params(cfg: Any) -> Hdf5CacheParams:
    decoding_cfg = _cfg_get(cfg, "analysis", "decoding")
    cache_cfg = decoding_cfg.get("cache", {}) if isinstance(decoding_cfg, dict) else getattr(decoding_cfg, "cache", {})
    compression = cache_cfg.get("compression", "gzip") if isinstance(cache_cfg, dict) else getattr(cache_cfg, "compression", "gzip")
    compression_level = int(cache_cfg.get("compression_level", 4)) if isinstance(cache_cfg, dict) else int(getattr(cache_cfg, "compression_level", 4))
    dtype = str(cache_cfg.get("dtype", "float32")) if isinstance(cache_cfg, dict) else str(getattr(cache_cfg, "dtype", "float32"))
    return Hdf5CacheParams(
        compression=compression,
        compression_level=compression_level,
        x_dtype=dtype,
    )
