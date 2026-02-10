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


_EPOCHS_FILENAME_PATTERN = re.compile(
    r"sub-(?P<subject>\d+)_task-(?P<task>[A-Za-z0-9]+)_run-(?P<run>\d+)_epochs-epo\.fif$"
)


@dataclass(frozen=True)
class EpochFileInfo:
    """Parsed information from an epochs filepath.

    Usage example
    -------------
        info = parse_epochs_filepath(Path(".../sub-005_task-conversation_run-1_epochs-epo.fif"))
        assert info.subject == "005"
        assert info.run == 1
    """

    subject: str
    task: str
    run: int
    path: Path


def parse_epochs_filepath(path: Path) -> EpochFileInfo:
    """Parse subject/task/run from an epochs filepath.

    Parameters
    ----------
    path
        Path like ``sub-005_task-conversation_run-1_epochs-epo.fif``.

    Returns
    -------
    info
        Parsed :class:`~turntaking.analysis.io.EpochFileInfo`.

    Raises
    ------
    ValueError
        If the filename does not match the expected pattern.
    """
    match = _EPOCHS_FILENAME_PATTERN.search(path.name)
    if match is None:
        raise ValueError(
            "Unrecognized epochs filename. Expected pattern like "
            "'sub-005_task-conversation_run-1_epochs-epo.fif'. "
            f"Got: {path.name}"
        )
    return EpochFileInfo(
        subject=match.group("subject"),
        task=match.group("task"),
        run=int(match.group("run")),
        path=path,
    )


def load_epochs(path: Path) -> mne.BaseEpochs:
    """Load epochs from disk.

    Parameters
    ----------
    path
        Path to an MNE epochs FIF file.

    Returns
    -------
    epochs
        Loaded epochs.

    Usage example
    -------------
        epochs = load_epochs(Path("sub-005_task-conversation_run-1_epochs-epo.fif"))
    """
    return mne.read_epochs(path, preload=True, verbose=False)


def save_evokeds(evokeds: dict[str, mne.Evoked], out_dir: Path) -> None:
    """Save evoked responses with standard naming.

    Parameters
    ----------
    evokeds
        Mapping from name -> Evoked.
    out_dir
        Output directory. Will be created if it doesn't exist.

    Notes
    -----
    Files are saved as ``{name}-ave.fif``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, evoked in evokeds.items():
        out_path = out_dir / f"{name}-ave.fif"
        evoked.save(out_path, overwrite=True)


def save_array(array: np.ndarray, path: Path) -> None:
    """Save numpy array to ``.npy``.

    Usage example
    -------------
        save_array(scores, Path("scores.npy"))
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as CSV or Parquet (based on suffix).

    Parameters
    ----------
    df
        Table to save.
    path
        Destination path. Use ``.csv`` or ``.parquet``.

    Usage example
    -------------
        save_table(df, Path("metadata.parquet"))
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported table extension: {path.suffix} (use .csv or .parquet)")
