"""
Low-level I/O utilities for analysis artifacts.

This module centralizes small, deterministic file-writing helpers so analysis
domains can implement their artifact contracts without duplicating boilerplate.

Design goals
------------
- Minimal surprise: keep dtype/shape unchanged unless explicitly requested.
- Deterministic outputs: stable CSV/HDF5 layout and JSON serialization.
- Domain-agnostic: no knowledge of ERP/TFR/decoding semantics.
"""

import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd

__all__ = [
    "ensure_dir_exists",
    "save_table",
    "save_table_csv",
    "save_npy",
    "save_hdf5",
    "save_array_nd",
    "save_dataframe_csv",
    "save_hdf5_dataset",
]


def ensure_dir_exists(path: str | Path) -> Path:
    """
    Ensure a directory exists and return it as a ``Path``.

    This helper centralizes parent directory creation for analysis outputs so
    domain modules can focus on computation and artifact contracts.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_array_nd(arr: np.ndarray, path: str | Path) -> Path:
    """
    Save an n-dimensional NumPy array as ``.npy``.

    Responsibility
    --------------
    - Creates parent directories.
    - Persists the array using ``numpy.save`` without changing dtype/shape.
    """
    out_path = Path(path)
    ensure_dir_exists(out_path.parent)
    np.save(out_path, arr)
    return out_path


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    """
    Save a DataFrame as ``.csv`` or ``.parquet``.

    This is a generic tabular helper used across analysis domains.
    """
    out_path = Path(path)
    ensure_dir_exists(out_path.parent)
    suffix = out_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(out_path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(out_path, index=False)
    else:
        raise ValueError(f"Unsupported table extension: {suffix}")
    return out_path


def save_dataframe_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """
    Save a DataFrame as CSV without index.

    Responsibility
    --------------
    - Creates parent directories.
    - Writes deterministic tabular outputs used by analysis pipelines.
    """
    out_path = Path(path)
    ensure_dir_exists(out_path.parent)
    df.to_csv(out_path, index=False)
    return out_path


def save_table_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Backward-compatible alias for CSV table outputs."""
    return save_dataframe_csv(df, path)


def save_npy(arr: np.ndarray, path: str | Path) -> Path:
    """Backward-compatible alias for `.npy` outputs."""
    return save_array_nd(arr, path)


def save_hdf5_dataset(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """
    Serialize a mapping to HDF5 with deterministic handling of mixed payloads.

    Strategy
    --------
    - ``ndarray`` values become datasets (object arrays are JSON-encoded).
    - ``list``/``tuple`` become datasets when possible, otherwise JSON bytes.
    - Scalars and other Python objects are JSON-encoded bytes.
    - ``None`` values are skipped.

    Notes
    -----
    HDF5 has no native "arbitrary Python object" type. For mixed payloads we
    encode JSON as UTF-8 bytes (stored as ``np.bytes_``) and keep arrays as
    proper datasets when possible.
    """
    out_path = Path(path)
    ensure_dir_exists(out_path.parent)

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

    with h5py.File(out_path, "w") as handle:
        for key, value in payload.items():
            if value is None:
                continue

            dataset_name = str(key)

            if isinstance(value, np.bytes_):
                handle.create_dataset(dataset_name, data=value)
                continue

            if isinstance(value, np.ndarray):
                if value.dtype == object:
                    blob = json.dumps(
                        value.tolist(),
                        sort_keys=True,
                        default=_json_default,
                    ).encode("utf-8")
                    handle.create_dataset(dataset_name, data=np.bytes_(blob))
                else:
                    handle.create_dataset(dataset_name, data=value)
                continue

            if isinstance(value, (list, tuple)):
                arr = np.asarray(value)
                if arr.dtype == object:
                    blob = json.dumps(
                        value,
                        sort_keys=True,
                        default=_json_default,
                    ).encode("utf-8")
                    handle.create_dataset(dataset_name, data=np.bytes_(blob))
                else:
                    handle.create_dataset(dataset_name, data=arr)
                continue

            blob = json.dumps(value, sort_keys=True, default=_json_default).encode("utf-8")
            handle.create_dataset(dataset_name, data=np.bytes_(blob))

    return out_path


def save_hdf5(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Backward-compatible alias for HDF5 mapping serialization."""
    return save_hdf5_dataset(path, payload)
