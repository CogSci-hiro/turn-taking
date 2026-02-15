# src/turntaking/cli/commands/tfr.py


import argparse
import glob
import re
from pathlib import Path
from typing import Any

from turntaking.analysis.datasets.evoked_dataset import build_evoked_dataset
from turntaking.analysis.tfr.io import write_tfr_outputs
from turntaking.analysis.selection import SelectionParams

_SUBJECT_RE = re.compile(r"(sub-\d{3})")


def _discover_subjects_from_epochs(epoch_dir: Path) -> list[str]:
    subjects: set[str] = set()
    for p in epoch_dir.glob("sub-*_task-*_run-*_epochs-epo.fif"):
        m = _SUBJECT_RE.search(p.name)
        if m:
            subjects.add(m.group(1))
    return sorted(subjects)


def _expand_epoch_paths(cfg: Any) -> list[Path]:
    epoch_dir = Path(cfg.io.epoch_dir)
    pattern = str(cfg.io.epoch_pattern)

    tasks = list(cfg.dataset.tasks)
    runs = [str(r) for r in cfg.dataset.runs]

    subjects_cfg = cfg.dataset.subjects
    if subjects_cfg.mode == "from_epochs":
        subjects = _discover_subjects_from_epochs(epoch_dir)
    else:
        subjects = list(subjects_cfg.include)

    exclude = set(subjects_cfg.exclude)
    subjects = [s for s in subjects if s not in exclude]

    invalid_pairs = {(str(s), str(r)) for s, r in cfg.dataset.invalid_subject_run}

    paths: list[Path] = []
    for subject in subjects:
        for task in tasks:
            for run in runs:
                if (subject, run) in invalid_pairs:
                    continue
                fname = pattern.format(subject=subject, task=task, run=run)
                p = epoch_dir / fname
                if p.is_file():
                    paths.append(p)

    paths = sorted(paths)
    if len(paths) == 0:
        raise RuntimeError(
            "No epoch files found after config expansion. "
            f"epoch_dir={epoch_dir}, epoch_pattern={pattern}"
        )
    return paths


def add_subparser(subparsers: Any) -> None:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "tfr",
        help="Run band-limited induced (Hilbert) TFR data generation (ERP-like contract).",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")

    # Optional overrides (dev convenience)
    parser.add_argument("--epochs-glob", default=None, help="Override epoch file glob.")
    parser.add_argument("--out-dir", default=None, help="Override output directory (base; 'tfr' appended).")
    parser.add_argument(
        "--contrast",
        choices=["latency", "duration"],
        default=None,
        help="Override: run only one contrast (otherwise run all from config).",
    )
    parser.add_argument(
        "--band",
        default=None,
        help="Override: run only one band (otherwise run all from config.analysis.bands).",
    )

    # Optional overrides for selection thresholds
    parser.add_argument("--min-latency", type=float, default=None)
    parser.add_argument("--max-latency", type=float, default=None)
    parser.add_argument("--min-response-duration", type=float, default=None)

    # Optional overrides for TFR settings
    parser.add_argument("--sfreq", type=float, default=None)


def run(args: argparse.Namespace, cfg: Any) -> None:
    # 1) Epoch files: CLI override or config expansion
    if args.epochs_glob:
        epoch_paths = [Path(p) for p in sorted(glob.glob(args.epochs_glob)) if Path(p).is_file()]
        if len(epoch_paths) == 0:
            raise RuntimeError(f"No files matched --epochs-glob: {args.epochs_glob}")
    else:
        epoch_paths = _expand_epoch_paths(cfg)

    # 2) Output root: base/tfr
    base_out_dir = (
        Path(args.out_dir) / "tfr"
        if args.out_dir
        else (Path(cfg.io.out_dir) / "tfr")
    )
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # 3) Contrasts: CLI override or cfg.analysis.contrasts
    contrasts = [args.contrast] if args.contrast else list(cfg.analysis.contrasts)
    if len(contrasts) == 0:
        raise RuntimeError("analysis.contrasts is empty and no --contrast override was provided.")

    # 4) Bands: CLI override or cfg.analysis.bands
    bands = [args.band] if args.band else list(cfg.analysis.bands)
    if len(bands) == 0:
        raise RuntimeError("analysis.bands is empty and no --band override was provided.")

    # 5) Selection thresholds: CLI override or cfg.constraints.*
    min_latency = args.min_latency if args.min_latency is not None else float(cfg.constraints.min_latency)
    max_latency = args.max_latency if args.max_latency is not None else float(cfg.constraints.max_latency)
    min_self_duration = (
        args.min_response_duration
        if args.min_response_duration is not None
        else float(cfg.constraints.min_response_duration)
    )
    selection_params = SelectionParams(
        min_latency=min_latency,
        max_latency=max_latency,
        min_self_duration=min_self_duration,
    )

    # 6) TFR settings
    sfreq = args.sfreq if args.sfreq is not None else float(cfg.analysis.tfr.sfreq)

    # 7) Run for each contrast × band
    for contrast in contrasts:
        for band in bands:
            out_dir = base_out_dir / str(contrast) / str(band)
            out_dir.mkdir(parents=True, exist_ok=True)

            result = build_evoked_dataset(
                epoch_paths=epoch_paths,
                kind="tfr",
                contrast=str(contrast),
                selection_params=selection_params,
                band=str(band),
                sfreq=sfreq,
            )

            write_tfr_outputs(
                out_dir,
                contrast=str(contrast),
                band=str(band),
                evokeds_cond_1=result.evokeds_cond_1,
                evokeds_cond_2=result.evokeds_cond_2,
                evokeds_difference=result.evokeds_difference,
                induced_data=result.evoked_data,  # same shape semantics (N,3,C,T)
                n_trials=result.n_trials,
                metadata=result.results,
                overwrite=True,
            )
