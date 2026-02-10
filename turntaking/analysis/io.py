"""turntaking.analysis.io

I/O helpers and filename conventions.

This module should stay boring:
- load/save helpers
- filename parsing utilities
- no analysis logic
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mne
import numpy as np
import pandas as pd


EpochFileKind = Literal["epo", "epochs"]


@dataclass(frozen=True)
class EpochFileInfo:
    """Parsed identifiers from an epochs file path.

    Notes
    -----
    Expected (example) filename pattern:
        sub-005_task-conversation_run-1_epochs-epo.fif

    Fields are best-effort; anything not found is set to None.
    """

    subject: str | None
    task: str | None
    run: int | None
    stem: str


_EPOCHS_RE = re.compile(
    r"(?P<subject>sub-[0-9A-Za-z]+)"
    r"(?:_task-(?P<task>[0-9A-Za-z-]+))?"
    r"(?:_run-(?P<run>[0-9]+))?"
    r".*?epochs.*?\.fif$"
)


def parse_epochs_path(path: Path) -> EpochFileInfo:
    """Parse subject/task/run identifiers from an epochs FIF path."""
    m = _EPOCHS_RE.search(path.name)
    if m is None:
        return EpochFileInfo(subject=None, task=None, run=None, stem=path.stem)
    run = m.group("run")
    return EpochFileInfo(
        subject=m.group("subject"),
        task=m.group("task"),
        run=int(run) if run is not None else None,
        stem=path.stem,
    )


def load_epochs(path: Path) -> mne.BaseEpochs:
    """Load MNE epochs from disk."""
    return mne.read_epochs(path, preload=False, verbose="ERROR")


def save_evokeds(evokeds: dict[str, mne.Evoked], out_dir: Path) -> None:
    """Save evoked responses with standard naming."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, evk in evokeds.items():
        evk.save(out_dir / f"{name}-ave.fif", overwrite=True)


def save_array(array: np.ndarray, path: Path) -> None:
    """Save NumPy array to ``.npy``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save table based on extension (csv/parquet)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path}")
