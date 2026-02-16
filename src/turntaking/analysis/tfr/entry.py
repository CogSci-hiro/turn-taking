
"""TFR command service: expands config inputs and writes induced-TFR artifacts."""

import argparse
import glob
import re
from pathlib import Path
from typing import Any

from turntaking.analysis.datasets.evoked_dataset import build_evoked_dataset
from turntaking.analysis.selection import SelectionParams
from turntaking.analysis.tfr.io import write_tfr_outputs

_SUBJECT_RE = re.compile(r"(sub-\d{3})")


def run_tfr(args: argparse.Namespace, cfg: Any) -> None:
    epoch_paths = _resolve_epoch_paths(args, cfg)
    base_out_dir = _resolve_base_out_dir(args, cfg)
    contrasts = _resolve_contrasts(args, cfg)
    bands = _resolve_bands(args, cfg)
    selection_params = _resolve_selection_params(args, cfg)
    sfreq = args.sfreq if args.sfreq is not None else float(cfg.analysis.tfr.sfreq)
    for contrast in contrasts:
        for band in bands:
            _run_single_combo(
                contrast=str(contrast),
                band=str(band),
                base_out_dir=base_out_dir,
                epoch_paths=epoch_paths,
                selection_params=selection_params,
                sfreq=sfreq,
            )


def _discover_subjects_from_epochs(epoch_dir: Path) -> list[str]:
    subjects: set[str] = set()
    for path in epoch_dir.glob("sub-*_task-*_run-*_epochs-epo.fif"):
        match = _SUBJECT_RE.search(path.name)
        if match:
            subjects.add(match.group(1))
    return sorted(subjects)


def _expand_epoch_paths(cfg: Any) -> list[Path]:
    epoch_dir = Path(cfg.io.epoch_dir)
    pattern = str(cfg.io.epoch_pattern)
    tasks = list(cfg.dataset.tasks)
    runs = [str(run) for run in cfg.dataset.runs]
    subjects_cfg = cfg.dataset.subjects
    subjects = _discover_subjects_from_epochs(epoch_dir) if subjects_cfg.mode == "from_epochs" else list(subjects_cfg.include)
    subjects = [subject for subject in subjects if subject not in set(subjects_cfg.exclude)]
    invalid_pairs = {(str(subject), str(run)) for subject, run in cfg.dataset.invalid_subject_run}
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
            f"epoch_dir={epoch_dir}, epoch_pattern={pattern}"
        )
    return paths


def _resolve_epoch_paths(args: argparse.Namespace, cfg: Any) -> list[Path]:
    if args.epochs_glob:
        epoch_paths = [Path(path) for path in sorted(glob.glob(args.epochs_glob)) if Path(path).is_file()]
        if len(epoch_paths) == 0:
            raise RuntimeError(f"No files matched --epochs-glob: {args.epochs_glob}")
        return epoch_paths
    return _expand_epoch_paths(cfg)


def _resolve_base_out_dir(args: argparse.Namespace, cfg: Any) -> Path:
    base_out_dir = Path(args.out_dir) / "tfr" if args.out_dir else (Path(cfg.io.out_dir) / "tfr")
    base_out_dir.mkdir(parents=True, exist_ok=True)
    return base_out_dir


def _resolve_contrasts(args: argparse.Namespace, cfg: Any) -> list[str]:
    contrasts = [args.contrast] if args.contrast else list(cfg.analysis.contrasts)
    if len(contrasts) == 0:
        raise RuntimeError("analysis.contrasts is empty and no --contrast override was provided.")
    return contrasts


def _resolve_bands(args: argparse.Namespace, cfg: Any) -> list[str]:
    bands = [args.band] if args.band else list(cfg.analysis.bands)
    if len(bands) == 0:
        raise RuntimeError("analysis.bands is empty and no --band override was provided.")
    return bands


def _resolve_selection_params(args: argparse.Namespace, cfg: Any) -> SelectionParams:
    min_latency = args.min_latency if args.min_latency is not None else float(cfg.constraints.min_latency)
    max_latency = args.max_latency if args.max_latency is not None else float(cfg.constraints.max_latency)
    min_self_duration = (
        args.min_response_duration
        if args.min_response_duration is not None
        else float(cfg.constraints.min_response_duration)
    )
    return SelectionParams(
        min_latency=min_latency,
        max_latency=max_latency,
        min_self_duration=min_self_duration,
    )


def _run_single_combo(
    *,
    contrast: str,
    band: str,
    base_out_dir: Path,
    epoch_paths: list[Path],
    selection_params: SelectionParams,
    sfreq: float,
) -> None:
    out_dir = base_out_dir / contrast / band
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_evoked_dataset(
        epoch_paths=epoch_paths,
        kind="tfr",
        contrast=contrast,
        selection_params=selection_params,
        band=band,
        sfreq=sfreq,
    )
    write_tfr_outputs(
        out_dir,
        contrast=contrast,
        band=band,
        evokeds_cond_1=result.evokeds_cond_1,
        evokeds_cond_2=result.evokeds_cond_2,
        evokeds_difference=result.evokeds_difference,
        induced_data=result.evoked_data,
        n_trials=result.n_trials,
        metadata=result.results,
        overwrite=True,
    )
