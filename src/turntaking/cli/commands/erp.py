# src/turntaking/cli/commands/erp.py

import argparse
import glob
import re
from pathlib import Path
from typing import Any

from turntaking.analysis.datasets.evoked_dataset import build_evoked_dataset
from turntaking.analysis.io import save_evokeds, save_table
from turntaking.analysis.selection import SelectionParams


_SUBJECT_RE = re.compile(r"(sub-\d{3})")


def add_subparser(subparsers: Any) -> None:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "erp",
        help="Generate ERP outputs from epochs (config-driven by default).",
    )
    parser.add_argument("--config", required=True, help="Path to analysis config YAML.")

    # Optional overrides (dev convenience; not required for normal workflow)
    parser.add_argument(
        "--epochs-glob",
        default=None,
        help="Optional override: glob pattern for epoch files. If omitted, derived from config.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional override: output directory. If omitted, uses io.out_dir from config.",
    )
    parser.add_argument(
        "--contrast",
        choices=["latency", "duration"],
        default=None,
        help="Optional override: run only one contrast. If omitted, runs all contrasts in config.",
    )

    # Optional overrides for selection thresholds (otherwise from cfg.constraints)
    parser.add_argument("--min-latency", type=float, default=None)
    parser.add_argument("--max-latency", type=float, default=None)
    parser.add_argument("--min-response-duration", type=float, default=None)


def _cfg_get(cfg: Any, *keys: str) -> Any:
    """Support cfg as dict-like or attribute-like."""
    cur = cfg
    for k in keys:
        if isinstance(cur, dict):
            cur = cur[k]
        else:
            cur = getattr(cur, k)
    return cur


def _discover_subjects_from_epochs(epoch_dir: Path) -> list[str]:
    candidates = epoch_dir.glob("sub-*_task-*_run-*_epochs-epo.fif")
    subjects: set[str] = set()
    for p in candidates:
        m = _SUBJECT_RE.search(p.name)
        if m:
            subjects.add(m.group(1))
    return sorted(subjects)


def _expand_epoch_paths_from_config(cfg: Any) -> list[Path]:
    epoch_dir = Path(_cfg_get(cfg, "io", "epoch_dir"))
    pattern = str(_cfg_get(cfg, "io", "epoch_pattern"))

    tasks = list(_cfg_get(cfg, "dataset", "tasks"))
    runs = [str(r) for r in _cfg_get(cfg, "dataset", "runs")]

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

    invalid_pairs = set()
    for pair in _cfg_get(cfg, "dataset", "invalid_subject_run"):
        # pair is like ["sub-004", "1"]
        invalid_pairs.add((str(pair[0]), str(pair[1])))

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
            f"epoch_dir={epoch_dir}, pattern={pattern}"
        )
    return paths


def run(args: argparse.Namespace, cfg: Any) -> None:
    # Outputs
    base_out_dir = Path(args.out_dir) if args.out_dir else Path(_cfg_get(cfg, "io", "out_dir")) / "erp"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Contrasts
    contrasts = [args.contrast] if args.contrast else list(_cfg_get(cfg, "analysis", "contrasts"))
    if len(contrasts) == 0:
        raise RuntimeError("analysis.contrasts is empty and no --contrast override was provided.")

    # Epoch inputs
    if args.epochs_glob:
        epoch_paths = [Path(p) for p in sorted(glob.glob(args.epochs_glob)) if Path(p).is_file()]
        if len(epoch_paths) == 0:
            raise RuntimeError(f"No files matched --epochs-glob: {args.epochs_glob}")
    else:
        epoch_paths = _expand_epoch_paths_from_config(cfg)

    # Selection thresholds (CLI overrides > config)
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

    # Run one or many contrasts
    for contrast in contrasts:
        out_dir = base_out_dir / str(contrast)
        out_dir.mkdir(parents=True, exist_ok=True)

        result = build_evoked_dataset(
            epoch_paths=epoch_paths,
            kind="erp",
            contrast=str(contrast),
            selection_params=selection_params,
        )

        save_evokeds({"grand_diff": result.difference}, out_dir=out_dir)
        save_table(result.metadata, out_dir / "metadata.parquet")
        save_table(result.n_trials, out_dir / "n_trials.csv")
