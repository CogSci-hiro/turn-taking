from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd


def save_array(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported table extension: {suffix}")


def save_hdf5(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, (int, float, str, bytes)):
                f.attrs[key] = value
            else:
                f.create_dataset(key, data=np.asarray(value))
