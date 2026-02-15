from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from turntaking.analysis.utils.io import (
    ensure_dir_exists,
    save_array_nd,
    save_dataframe_csv,
    save_hdf5_dataset,
)

__all__ = [
    "save_table",
    "save_table_csv",
    "save_npy",
    "save_hdf5",
]


def save_table(df: pd.DataFrame, path: Path) -> None:
    ensure_dir_exists(path.parent)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported table extension: {suffix}")


def save_table_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as CSV (no index), ensuring parent dirs exist."""
    save_dataframe_csv(df, path)


def save_npy(arr: np.ndarray, path: Path) -> None:
    """Save a NumPy array as .npy, ensuring parent dirs exist."""
    save_array_nd(arr, path)


def save_hdf5(path: Path, payload: Mapping[str, Any]) -> None:
    """Backward-compatible wrapper around shared HDF5 serialization."""
    save_hdf5_dataset(path, payload)
