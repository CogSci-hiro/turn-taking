# =============================================================================
#                               Shared helpers
# =============================================================================
#
# Small, pure helper functions used across multiple Snakemake rule files.
# Keep logic here to avoid duplicating filesystem parsing in each rule file.
#

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from snakemake.exceptions import WorkflowError


# =============================================================================
# Conda (optional)
# =============================================================================
# NOTE:
# This stanza is harmless unless you run Snakemake with --use-conda.
# It mirrors the style of your other project.
conda:
    CONDA_PY_ENV


# =============================================================================
# Basic config accessors
# =============================================================================
def epoch_dir() -> Path:
    p = Path(config["io"]["epoch_dir"])
    if not p.exists():
        raise WorkflowError(f"io.epoch_dir does not exist: {p}")
    return p


def epoch_pattern() -> str:
    return str(config["io"]["epoch_pattern"])


def out_dir() -> Path:
    return Path(config["io"]["out_dir"])


def entrypoint() -> Path:
    # Anchor project root to workflow directory (more reliable than project.workdir)
    project_root = Path(workflow.basedir).resolve().parent
    p = project_root / "src" / "turntaking" / "cli" / "main.py"
    if not p.exists():
        raise WorkflowError(f"Could not find CLI entrypoint at: {p}")
    return p


def sentinel_path(analysis_name: str) -> Path:
    return out_dir() / analysis_name / "_DONE"


# =============================================================================
# Epoch discovery (filesystem only; no BIDS root)
# =============================================================================
_SUBJECT_RE = re.compile(r"sub-\d+")
_TASK_RE = re.compile(r"task-[A-Za-z0-9_-]+")
_RUN_RE = re.compile(r"run-\d+")


def _glob_from_pattern(pattern: str) -> str:
    # Replace {wildcards} with * for globbing
    return re.sub(r"\{[^}]+\}", "*", pattern)


def extract_entities(path: Path) -> Dict[str, str]:
    """Extract subject/task/run tokens from a file path or name.

    This relies only on tokens like 'sub-005', 'task-conversation', 'run-8'
    appearing in the path parts or filename.

    Returns
    -------
    entities : dict
        Keys: 'subject' (e.g., 'sub-005'), 'task' (e.g., 'conversation'),
        'run' (int-like string, no leading zeros).
    """
    parts = list(path.parts)
    fname = path.name

    subject = next((p for p in parts if _SUBJECT_RE.fullmatch(p)), None)
    if subject is None:
        m = _SUBJECT_RE.search(fname)
        subject = m.group(0) if m else None

    task_tok = next((p for p in parts if _TASK_RE.fullmatch(p)), None)
    if task_tok is None:
        m = _TASK_RE.search(fname)
        task_tok = m.group(0) if m else None

    run_tok = next((p for p in parts if _RUN_RE.fullmatch(p)), None)
    if run_tok is None:
        m = _RUN_RE.search(fname)
        run_tok = m.group(0) if m else None

    if subject is None or task_tok is None or run_tok is None:
        raise WorkflowError(
            "Could not extract subject/task/run from epoch path. "
            f"Got subject={subject}, task={task_tok}, run={run_tok} for: {path}"
        )

    task = task_tok.split("-", 1)[1]
    run_int = int(run_tok.split("-", 1)[1])

    return {"subject": subject, "task": task, "run": str(run_int)}


def discover_epoch_files() -> List[Path]:
    ep_dir = epoch_dir()
    pat = epoch_pattern()
    glob_pat = _glob_from_pattern(pat)
    files = sorted(ep_dir.glob(glob_pat))
    if not files:
        raise WorkflowError(
            "No epoch files found. "
            f"epoch_dir={ep_dir}, epoch_pattern={pat}, glob={glob_pat}"
        )
    return files


# =============================================================================
# Filtering (include/exclude/tasks/runs/invalid pairs)
# =============================================================================
def _subjects_cfg() -> Dict:
    return dict(config.get("dataset", {}).get("subjects", {}))


def include_subjects() -> Optional[Set[str]]:
    inc = _subjects_cfg().get("include")
    if inc:
        return {_norm_subject(x) for x in inc}
    return None

def exclude_subjects() -> Set[str]:
    return {_norm_subject(x) for x in _subjects_cfg().get("exclude", [])}

def tasks_cfg() -> Optional[Set[str]]:
    tasks = config.get("dataset", {}).get("tasks")
    if tasks:
        return {_norm_task(t) for t in tasks}
    return None

def runs_cfg() -> Optional[Set[int]]:
    runs = config.get("dataset", {}).get("runs")
    if runs:
        return {_norm_run(r) for r in runs}
    return None

def invalid_subject_run() -> Set[Tuple[str, int]]:
    pairs = config.get("dataset", {}).get("invalid_subject_run", [])
    return {(_norm_subject(s), _norm_run(r)) for s, r in pairs}


def keep_epoch_file(p: Path) -> bool:
    ent = extract_entities(p)
    subject = ent["subject"]
    task = ent["task"]
    run = int(ent["run"])

    exc = exclude_subjects()
    inc = include_subjects()
    tasks = tasks_cfg()
    runs = runs_cfg()
    invalid = invalid_subject_run()

    if subject in exc:
        return False
    if inc is not None and subject not in inc:
        return False
    if tasks is not None and task not in tasks:
        return False
    if runs is not None and run not in runs:
        return False
    if (subject, run) in invalid:
        return False

    return True


def epoch_inputs() -> List[str]:
    files_all = discover_epoch_files()
    files_keep = [p for p in files_all if keep_epoch_file(p)]

    if not files_keep:
        # show a small sample + the normalized filters
        inc = include_subjects()
        exc = exclude_subjects()
        tasks = tasks_cfg()
        runs = runs_cfg()
        invalid = invalid_subject_run()

        sample = "\n".join(str(p.name) for p in files_all[:10])
        raise WorkflowError(
            "After applying config filters, no epoch files remain.\n\n"
            f"Discovered: {len(files_all)} files\n"
            f"Kept:       {len(files_keep)} files\n\n"
            f"Normalized filters:\n"
            f"  include_subjects={sorted(inc) if inc else None}\n"
            f"  exclude_subjects={sorted(exc) if exc else []}\n"
            f"  tasks={sorted(tasks) if tasks else None}\n"
            f"  runs={sorted(runs) if runs else None}\n"
            f"  invalid_subject_run(n)={len(invalid)}\n\n"
            f"Sample discovered filenames:\n{sample}\n"
        )

    return [str(p) for p in files_keep]


def _norm_subject(s: str) -> str:
    s = str(s).strip()
    if s.startswith("sub-"):
        return s
    # allow "5", "05", "005"
    if re.fullmatch(r"\d+",s):
        return f"sub-{int(s):03d}"
    return s  # fallback (leave as-is)


def _norm_task(t: str) -> str:
    t = str(t).strip()
    if t.startswith("task-"):
        return t.split("-", 1)[1]
    return t


def _norm_run(r) -> int:
    r = str(r).strip()
    if r.startswith("run-"):
        r = r.split("-", 1)[1]
    return int(r)
