
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import mne
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EpochFileInfo:
    subject: str
    run: int


_EPOCHS_RE = re.compile(
    r".*[/\\](?P<subject>sub-\d+)_.*_run-(?P<run>\d+)_epochs-epo\.fif$"
)


def parse_epochs_filepath(path: Path) -> EpochFileInfo:
    """Parse subject/run from an epochs filepath.

    Expected pattern (example)
    --------------------------
        sub-005_task-conversation_run-1_epochs-epo.fif
    """
    match = _EPOCHS_RE.match(str(path))
    if match is None:
        raise ValueError(f"Could not parse subject/run from epochs path: {path}")
    subject = match.group("subject").replace("sub-", "")
    run = int(match.group("run"))
    return EpochFileInfo(subject=subject, run=run)


def load_epochs(path: Path) -> mne.BaseEpochs:
    """Load epochs from disk."""
    return mne.read_epochs(str(path), preload=False)


def save_evokeds(evokeds: Dict[str, mne.Evoked], out_dir: Path) -> None:
    """Save evoked responses with standard naming."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, evk in evokeds.items():
        evk_path = out_dir / f"{key}-ave.fif"
        evk.save(str(evk_path), overwrite=True)


def save_array(array: np.ndarray, path: Path) -> None:
    """Save numpy array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array)


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame (csv/parquet decided by extension)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported table extension: {suffix}")
