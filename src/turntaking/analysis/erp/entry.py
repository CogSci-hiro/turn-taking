from __future__ import annotations

"""ERP command service: orchestrates config expansion and ERP artifact generation."""

import argparse
import glob
import re
from pathlib import Path
from typing import Any

from turntaking.analysis.datasets.evoked_dataset import build_evoked_dataset
from turntaking.analysis.erp.io import save_erp_results
from turntaking.analysis.selection import SelectionParams

_SUBJECT_RE = re.compile(r"(sub-\d{3})")


def run_erp(args: argparse.Namespace, cfg: Any) -> None:
    epoch_paths = _resolve_epoch_paths(args, cfg)
    base_out_dir = _resolve_base_out_dir(args, cfg)
    contrasts = _resolve_contrasts(args, cfg)
    selection_params = _resolve_selection_params(args, cfg)
    for contrast in contrasts:
        _run_single_contrast(
            contrast=str(contrast),
            base_out_dir=base_out_dir,
            epoch_paths=epoch_paths,
            selection_params=selection_params,
        )


def _discover_subjects_from_epochs(epoch_dir: Path) -> list[str]:
    subjects: set[str] = set()
    for path in epoch_dir.glob("sub-*_task-*_run-*_epochs-epo.fif"):
        match = _SUBJECT_RE.search(path.name)
        if match:
            subjects.add(match.group(1))
    return sorted(subjects)


def _cfg_get(cfg: Any, *keys: str) -> Any:
    current = cfg
    for key in keys:
        current = current[key] if isinstance(current, dict) else getattr(current, key)
    return current


def _expand_epoch_paths_from_config(cfg: Any) -> list[Path]:
    epoch_dir = Path(_cfg_get(cfg, "io", "epoch_dir"))
    pattern = str(_cfg_get(cfg, "io", "epoch_pattern"))
    tasks = list(_cfg_get(cfg, "dataset", "tasks"))
    runs = [str(run) for run in _cfg_get(cfg, "dataset", "runs")]
    subjects = _subjects_from_cfg(cfg, epoch_dir)
    invalid_pairs = {(str(subject), str(run)) for subject, run in _cfg_get(cfg, "dataset", "invalid_subject_run")}
    paths = _expanded_paths(epoch_dir, pattern, tasks, runs, subjects, invalid_pairs)
    if len(paths) == 0:
        raise RuntimeError(
            "No epoch files found after config expansion. "
            f"epoch_dir={epoch_dir}, pattern={pattern}"
        )
    return paths


def _subjects_from_cfg(cfg: Any, epoch_dir: Path) -> list[str]:
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
    return [subject for subject in subjects if subject not in exclude]


def _expanded_paths(
    epoch_dir: Path,
    pattern: str,
    tasks: list[str],
    runs: list[str],
    subjects: list[str],
    invalid_pairs: set[tuple[str, str]],
) -> list[Path]:
    paths: list[Path] = []
    for subject in subjects:
        for task in tasks:
            for run in runs:
                if (subject, run) in invalid_pairs:
                    continue
                path = epoch_dir / pattern.format(subject=subject, task=task, run=run)
                if path.is_file():
                    paths.append(path)
    return sorted(paths)


def _resolve_epoch_paths(args: argparse.Namespace, cfg: Any) -> list[Path]:
    if args.epochs_glob:
        epoch_paths = [Path(path) for path in sorted(glob.glob(args.epochs_glob)) if Path(path).is_file()]
        if len(epoch_paths) == 0:
            raise RuntimeError(f"No files matched --epochs-glob: {args.epochs_glob}")
        return epoch_paths
    return _expand_epoch_paths_from_config(cfg)


def _resolve_base_out_dir(args: argparse.Namespace, cfg: Any) -> Path:
    base_out_dir = Path(args.out_dir) if args.out_dir else (Path(cfg.io.out_dir) / "erp")
    base_out_dir.mkdir(parents=True, exist_ok=True)
    return base_out_dir


def _resolve_contrasts(args: argparse.Namespace, cfg: Any) -> list[str]:
    contrasts = [args.contrast] if args.contrast else list(cfg.analysis.contrasts)
    if len(contrasts) == 0:
        raise RuntimeError("analysis.contrasts is empty and no --contrast override was provided.")
    return contrasts


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


def _run_single_contrast(
    *,
    contrast: str,
    base_out_dir: Path,
    epoch_paths: list[Path],
    selection_params: SelectionParams,
) -> None:
    out_dir = base_out_dir / contrast
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_evoked_dataset(
        epoch_paths=epoch_paths,
        kind="erp",
        contrast=contrast,
        selection_params=selection_params,
    )
    save_erp_results(
        out_dir,
        contrast=contrast,
        evokeds_cond_1=result.evokeds_cond_1,
        evokeds_cond_2=result.evokeds_cond_2,
        evokeds_difference=result.evokeds_difference,
        evoked_data=result.evoked_data,
        n_trials=result.n_trials,
        results=result.results,
        offsets=result.offsets,
        overwrite=True,
    )
