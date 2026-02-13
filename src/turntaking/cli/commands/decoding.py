# src/turntaking/cli/commands/decoding.py


import argparse
import glob
import re
from pathlib import Path
from typing import Any, Literal

from turntaking.analysis.decoding.dataset import DecodingDatasetParams
from turntaking.analysis.io.decoding import save_decoding_scores
from turntaking.analysis.decoding.run_decoding import DecodingRunParams, run_group_decoding
from turntaking.analysis.io.epochs import load_subject_epochs
from turntaking.analysis.selection import SelectionParams

Contrast = Literal["latency", "duration"]

_SUBJECT_RE = re.compile(r"(sub-\d{3})")


def _discover_subjects_from_epochs(epoch_dir: Path) -> list[str]:
    subjects: set[str] = set()
    for path in epoch_dir.glob("sub-*_task-*_run-*_epochs-epo.fif"):
        match = _SUBJECT_RE.search(path.name)
        if match:
            subjects.add(match.group(1))
    return sorted(subjects)


def _cfg_get(cfg: Any, *keys: str) -> Any:
    """Support cfg as dict-like or attribute-like."""
    current: Any = cfg
    for key in keys:
        if isinstance(current, dict):
            current = current[key]
        else:
            current = getattr(current, key)
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

    subjects = [s for s in subjects if s not in exclude]
    return sorted(subjects)


def _expand_epoch_paths_from_config(cfg: Any) -> list[Path]:
    """
    Expand epoch file paths (like ERP command) for validation / dev overrides.

    Decoding itself loads epochs by subject via `load_subject_epochs()`, so this is
    mainly used when the user passes --epochs-glob or when we want to sanity-check
    that the config expansion yields something.
    """
    epoch_dir = Path(_cfg_get(cfg, "io", "epoch_dir"))
    pattern = str(_cfg_get(cfg, "io", "epoch_pattern"))

    tasks = list(_cfg_get(cfg, "dataset", "tasks"))
    runs = [str(r) for r in _cfg_get(cfg, "dataset", "runs")]

    subjects = _expand_subjects_from_config(cfg)

    invalid_pairs: set[tuple[str, str]] = set()
    for pair in _cfg_get(cfg, "dataset", "invalid_subject_run"):
        invalid_pairs.add((str(pair[0]), str(pair[1])))

    paths: list[Path] = []
    for subject in subjects:
        for task in tasks:
            for run in runs:
                if (subject, run) in invalid_pairs:
                    continue
                filename = pattern.format(subject=subject, task=task, run=run)
                path = epoch_dir / filename
                if path.is_file():
                    paths.append(path)

    paths = sorted(paths)
    if len(paths) == 0:
        raise RuntimeError(
            "No epoch files found after config expansion. "
            f"epoch_dir={epoch_dir}, pattern={pattern}"
        )
    return paths


def add_subparser(subparsers: Any) -> None:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "decoding",
        help="Run ERP temporal-generalization decoding and write scores.npy/times.npy.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")

    # Optional overrides (dev convenience)
    parser.add_argument("--epochs-glob", default=None, help="Override epoch file glob (validation only).")
    parser.add_argument("--out-dir", default=None, help="Override base output directory (writes into decoding/erp/*).")
    parser.add_argument(
        "--contrast",
        choices=["latency", "duration"],
        required=True,
        help="Which contrast to decode.",
    )

    # Decoding hyperparameters (optional overrides)
    parser.add_argument("--sfreq", type=float, default=None, help="Override decoding sfreq (Hz).")
    parser.add_argument("--n-splits", type=int, default=None, help="Override number of CV folds.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed for deterministic CV.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Override parallel jobs for decoding.")

    # Optional feature caching
    parser.add_argument(
        "--cache-features",
        action="store_true",
        help="Cache per-subject (X,y,times) features to HDF5 for faster iteration.",
    )

    # Optional overrides for selection thresholds
    parser.add_argument("--min-latency", type=float, default=None)
    parser.add_argument("--max-latency", type=float, default=None)
    parser.add_argument("--min-response-duration", type=float, default=None)


def run(args: argparse.Namespace, cfg: Any) -> None:
    # 0) Basic config-driven inputs
    epoch_dir = Path(_cfg_get(cfg, "io", "epoch_dir"))

    # 1) Optional validation via --epochs-glob or config expansion
    #    (Decoding uses subject-wise loader; this step is just to fail early if paths are wrong.)
    if args.epochs_glob:
        epoch_paths = [Path(p) for p in sorted(glob.glob(args.epochs_glob)) if Path(p).is_file()]
        if len(epoch_paths) == 0:
            raise RuntimeError(f"No files matched --epochs-glob: {args.epochs_glob}")
    else:
        _ = _expand_epoch_paths_from_config(cfg)

    # 2) Subjects (deterministic order)
    subjects = _expand_subjects_from_config(cfg)
    if len(subjects) == 0:
        raise RuntimeError("No subjects resolved from config (after include/exclude).")

    # 3) Output dir: CLI override or cfg.io.out_dir
    #    Writes into: {base_out_dir}/decoding/erp/{contrast}/scores.npy and times.npy
    base_out_dir = Path(args.out_dir) if args.out_dir else Path(_cfg_get(cfg, "io", "out_dir"))
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # 4) Selection thresholds: CLI override or cfg.constraints.*
    min_latency = args.min_latency if args.min_latency is not None else float(_cfg_get(cfg, "constraints", "min_latency"))
    max_latency = args.max_latency if args.max_latency is not None else float(_cfg_get(cfg, "constraints", "max_latency"))
    min_self_duration = (
        args.min_response_duration
        if args.min_response_duration is not None
        else float(_cfg_get(cfg, "constraints", "min_response_duration"))
    )

    selection_params = SelectionParams(
        min_latency=min_latency,
        max_latency=max_latency,
        min_self_duration=min_self_duration,
    )

    # 5) Decoding parameters: CLI override or cfg.analysis.decoding.*
    sfreq_hz = float(args.sfreq) if args.sfreq is not None else float(_cfg_get(cfg, "analysis", "decoding", "sfreq"))
    n_splits = int(args.n_splits) if args.n_splits is not None else int(_cfg_get(cfg, "analysis", "decoding", "n_splits"))

    seed = int(args.seed) if args.seed is not None else int(getattr(_cfg_get(cfg, "analysis", "decoding"), "seed", 0) if not isinstance(_cfg_get(cfg, "analysis", "decoding"), dict) else _cfg_get(cfg, "analysis", "decoding").get("seed", 0))
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else int(getattr(_cfg_get(cfg, "analysis", "decoding"), "n_jobs", 10) if not isinstance(_cfg_get(cfg, "analysis", "decoding"), dict) else _cfg_get(cfg, "analysis", "decoding").get("n_jobs", 10))

    dataset_params = DecodingDatasetParams(
        contrast=str(args.contrast),  # type: ignore[arg-type]
        selection=selection_params,
        sfreq_hz=sfreq_hz,
    )
    run_params = DecodingRunParams(
        n_splits=n_splits,
        seed=seed,
        n_jobs=n_jobs,
    )

    # 6) Optional caching (HDF5). If not implemented yet, error clearly if requested.
    load_cached_features_fn = None
    save_cached_features_fn = None
    if args.cache_features:
        try:
            from turntaking.analysis.io.decoding import (
                Hdf5CacheParams,
                load_subject_feature_cache_hdf5,
                save_subject_feature_cache_hdf5
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "--cache-features was requested, but HDF5 cache helpers are not available. "
                "Implement turntaking.analysis.decoding.hdf5_cache or disable caching."
            ) from exc

        cache_cfg = _cfg_get(cfg, "analysis", "decoding").get("cache", {}) if isinstance(_cfg_get(cfg, "analysis", "decoding"), dict) else getattr(_cfg_get(cfg, "analysis", "decoding"), "cache", {})
        compression = cache_cfg.get("compression", "gzip") if isinstance(cache_cfg, dict) else getattr(cache_cfg, "compression", "gzip")
        compression_level = int(cache_cfg.get("compression_level", 4)) if isinstance(cache_cfg, dict) else int(getattr(cache_cfg, "compression_level", 4))
        dtype = str(cache_cfg.get("dtype", "float32")) if isinstance(cache_cfg, dict) else str(getattr(cache_cfg, "dtype", "float32"))

        cache_params = Hdf5CacheParams(
            compression=compression,
            compression_level=compression_level,
            x_dtype=dtype,
        )

        def _load(subject: str):
            return load_subject_feature_cache_hdf5(
                out_dir=base_out_dir,
                contrast=str(args.contrast),  # type: ignore[arg-type]
                subject=subject,
            )

        def _save(subject: str, X, y, times_s):
            save_subject_feature_cache_hdf5(
                out_dir=base_out_dir,
                contrast=str(args.contrast),  # type: ignore[arg-type]
                subject=subject,
                X=X,
                y=y,
                times_s=times_s,
                cache_params=cache_params,
            )

        load_cached_features_fn = _load
        save_cached_features_fn = _save

    # 7) Run decoding (end-to-end)
    scores, times_s = run_group_decoding(
        subjects=subjects,
        epoch_dir=epoch_dir,
        dataset_params=dataset_params,
        run_params=run_params,
        load_subject_epochs_fn=load_subject_epochs,
        load_cached_features_fn=load_cached_features_fn,
        save_cached_features_fn=save_cached_features_fn,
    )

    # 8) Save outputs (ONLY scores + times)
    save_decoding_scores(
        out_dir=base_out_dir,
        contrast=str(args.contrast),  # type: ignore[arg-type]
        scores=scores,
        times_s=times_s,
    )
