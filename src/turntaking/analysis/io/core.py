from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported table extension: {suffix}")


def save_table_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as CSV (no index), ensuring parent dirs exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_npy(arr: np.ndarray, path: Path) -> None:
    """Save a NumPy array as .npy, ensuring parent dirs exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def save_hdf5(path: Path, payload: Mapping[str, Any]) -> None:
    """
    Minimal deterministic HDF5 serializer.

    Strategy
    --------
    - Scalars -> HDF5 attrs
    - ndarray -> dataset
    - list/tuple -> dataset if numeric/string; otherwise JSON bytes
    - everything else -> JSON bytes

    Notes
    -----
    Avoids object dtype failures and NumPy 2.0 `np.string_` pitfalls.
    """
    import json
    import h5py

    def _json_default(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (bytes, bytearray)):
            return {"__bytes__": True, "data": bytes(obj).decode("latin-1")}
        return str(obj)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in payload.items():
            if value is None:
                continue

            k = str(key)

            if isinstance(value, (int, float, str, bytes, np.integer, np.floating)):
                f.attrs[k] = value
                continue

            if isinstance(value, np.ndarray):
                if value.dtype == object:
                    payload_bytes = json.dumps(
                        value.tolist(), sort_keys=True, default=_json_default
                    ).encode("utf-8")
                    f.create_dataset(k, data=np.bytes_(payload_bytes))
                else:
                    f.create_dataset(k, data=value)
                continue

            if isinstance(value, (list, tuple)):
                arr = np.asarray(value)
                if arr.dtype == object:
                    payload_bytes = json.dumps(
                        value, sort_keys=True, default=_json_default
                    ).encode("utf-8")
                    f.create_dataset(k, data=np.bytes_(payload_bytes))
                else:
                    f.create_dataset(k, data=arr)
                continue

            payload_bytes = json.dumps(value, sort_keys=True, default=_json_default).encode("utf-8")
            f.create_dataset(k, data=np.bytes_(payload_bytes))
