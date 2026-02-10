"""
IO helpers and filename conventions for analysis outputs.
"""

from pathlib import Path

import mne
import numpy as np
import pandas as pd


def load_epochs(path: Path) -> mne.BaseEpochs:
    """Load epochs from disk."""
    raise NotImplementedError


def save_evokeds(evokeds: dict[str, mne.Evoked], out_dir: Path) -> None:
    """Save evoked responses with standard naming."""
    raise NotImplementedError


def save_array(array: np.ndarray, path: Path) -> None:
    """Save numpy array."""
    raise NotImplementedError


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame (csv/parquet decided by extension)."""
    raise NotImplementedError
