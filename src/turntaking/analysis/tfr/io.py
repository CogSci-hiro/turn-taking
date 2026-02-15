from __future__ import annotations

"""
TFR domain I/O helpers.

This module defines the file-output contract for induced TFR artifacts and
contains only I/O responsibilities (naming, validation, persistence).
Computation remains in ``turntaking.analysis.tfr.core``.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mne
import numpy as np
import pandas as pd

from turntaking.stats.cluster_test import ClusterTestResult
from turntaking.analysis.utils.io import (
    ensure_dir_exists,
    save_array_nd,
    save_dataframe_csv,
    save_hdf5_dataset,
)

__all__ = [
    "TfrConditionNames",
    "get_tfr_condition_names",
    "write_tfr_outputs",
    "write_cluster_outputs",
    "read_cluster_outputs",
]


@dataclass(frozen=True)
class TfrConditionNames:
    cond_1: str
    cond_2: str


def get_tfr_condition_names(contrast: str) -> TfrConditionNames:
    """Map TFR contrast names to condition labels used in output filenames."""
    if contrast == "duration":
        return TfrConditionNames(cond_1="long", cond_2="short")
    if contrast == "latency":
        return TfrConditionNames(cond_1="fast", cond_2="slow")
    raise ValueError(f"Unknown contrast: {contrast!r}. Expected 'duration' or 'latency'.")


def _build_output_paths(out_dir: Path, names: TfrConditionNames) -> dict[str, Path]:
    return {
        "difference": out_dir / "difference_ave.fif",
        "condition_1": out_dir / f"{names.cond_1}_ave.fif",
        "condition_2": out_dir / f"{names.cond_2}_ave.fif",
        "induced": out_dir / "induced-data.npy",
        "n_trials": out_dir / "n_trials.csv",
        "metadata": out_dir / "metadata.hdf5",
    }


def _validate_write_inputs(
    evokeds_cond_1: Sequence[mne.Evoked],
    evokeds_cond_2: Sequence[mne.Evoked],
    evokeds_difference: Sequence[mne.Evoked],
    induced_data: np.ndarray,
) -> None:
    if len(evokeds_cond_1) != len(evokeds_cond_2):
        raise ValueError(
            f"Cond lists must match in length: {len(evokeds_cond_1)} vs {len(evokeds_cond_2)}"
        )
    if len(evokeds_difference) != len(evokeds_cond_1):
        raise ValueError(
            f"Difference list must match subject count: {len(evokeds_difference)} vs {len(evokeds_cond_1)}"
        )
    if induced_data.ndim < 3:
        raise ValueError(f"induced_data looks wrong (expected >=3 dims), got shape={induced_data.shape}")


def _enrich_metadata(
    metadata: Mapping[str, Any],
    *,
    contrast: str,
    band: str,
    names: TfrConditionNames,
    induced_data: np.ndarray,
) -> dict[str, Any]:
    meta = dict(metadata)
    meta.setdefault("kind", "tfr")
    meta.setdefault("contrast", str(contrast))
    meta.setdefault("band", str(band))
    meta.setdefault("condition_1", names.cond_1)
    meta.setdefault("condition_2", names.cond_2)
    meta.setdefault("induced_data_shape", np.array(induced_data.shape, dtype=int))
    return meta


def write_tfr_outputs(
    out_dir: Path,
    *,
    contrast: str,
    band: str,
    evokeds_cond_1: Sequence[mne.Evoked],
    evokeds_cond_2: Sequence[mne.Evoked],
    evokeds_difference: Sequence[mne.Evoked],
    induced_data: np.ndarray,
    n_trials: pd.DataFrame,
    metadata: Mapping[str, Any],
    overwrite: bool = True,
) -> None:
    """Write induced-TFR artifacts for one contrast × one band."""
    out_dir = ensure_dir_exists(out_dir)
    names = get_tfr_condition_names(contrast)
    paths = _build_output_paths(out_dir, names)
    _validate_write_inputs(evokeds_cond_1, evokeds_cond_2, evokeds_difference, induced_data)
    meta = _enrich_metadata(
        metadata,
        contrast=contrast,
        band=band,
        names=names,
        induced_data=induced_data,
    )

    mne.write_evokeds(paths["condition_1"].as_posix(), list(evokeds_cond_1), overwrite=overwrite)
    mne.write_evokeds(paths["condition_2"].as_posix(), list(evokeds_cond_2), overwrite=overwrite)
    mne.write_evokeds(paths["difference"].as_posix(), list(evokeds_difference), overwrite=overwrite)

    save_array_nd(induced_data, paths["induced"])
    save_dataframe_csv(n_trials, paths["n_trials"])

    if paths["metadata"].exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite metadata: {paths['metadata']}")
    save_hdf5_dataset(paths["metadata"], meta)

    required = list(paths.values())
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing TFR outputs after write: {missing} (out_dir={out_dir})")


def write_cluster_outputs(out_dir: Path, result: ClusterTestResult) -> None:
    out_dir = Path(out_dir)
    payload = _cluster_payload(result)
    save_hdf5_dataset(out_dir / "cluster_results.hdf5", payload)
    save_dataframe_csv(_cluster_summary(result), out_dir / "cluster_summary.csv")


def _cluster_payload(result: ClusterTestResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "t-values": np.asarray(result.t_values, dtype=float),
        "p-values": np.asarray(result.p_values, dtype=float),
        "h0": np.asarray(result.h0, dtype=float),
        "meta/json": np.bytes_(json.dumps(result.metadata, sort_keys=True).encode("utf-8")),
    }
    for idx, cluster in enumerate(result.clusters):
        for dim_i, inds in enumerate(cluster):
            payload[f"clusters/{dim_i}-{idx}"] = np.asarray(inds, dtype=int)
    return payload


def _cluster_summary(result: ClusterTestResult) -> pd.DataFrame:
    p_values = np.asarray(result.p_values, dtype=float)
    return pd.DataFrame(
        [
            {
                **result.metadata,
                "n_clusters": int(p_values.size),
                "min_p": float(np.min(p_values)) if p_values.size else float("nan"),
                "n_p_lt_0_05": int(np.sum(p_values < 0.05)) if p_values.size else 0,
            }
        ]
    )


def read_cluster_outputs(path: Path) -> ClusterTestResult:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cluster results not found: {path}")
    try:
        import h5py
    except Exception as exc:  # noqa: BLE001
        raise ImportError("h5py is required to read cluster_results.hdf5.") from exc
    with h5py.File(path, "r") as handle:
        t_values = np.asarray(handle["t-values"], dtype=float)
        p_values = np.asarray(handle["p-values"], dtype=float)
        h0 = np.asarray(handle["h0"], dtype=float)
        metadata = _read_cluster_metadata(handle)
        clusters = _read_cluster_index_groups(handle)
    return ClusterTestResult(t_values=t_values, p_values=p_values, h0=h0, clusters=clusters, metadata=metadata)


def _read_cluster_metadata(handle: Any) -> dict[str, Any]:
    if "meta/json" not in handle:
        return {}
    raw = handle["meta/json"][()]
    text = raw.decode("utf-8") if isinstance(raw, (bytes, np.bytes_)) else bytes(raw).decode("utf-8")
    return json.loads(text)


def _read_cluster_index_groups(handle: Any) -> list[tuple[np.ndarray, ...]]:
    group = handle.get("clusters", None)
    if group is None:
        return []
    by_cluster: dict[int, dict[int, np.ndarray]] = {}
    for name in group.keys():
        dim_str, idx_str = name.split("-", 1)
        cluster_idx = int(idx_str)
        dim_idx = int(dim_str)
        by_cluster.setdefault(cluster_idx, {})[dim_idx] = np.asarray(group[name], dtype=int)
    clusters: list[tuple[np.ndarray, ...]] = []
    for cluster_idx in sorted(by_cluster.keys()):
        dims = by_cluster[cluster_idx]
        max_dim = max(dims.keys()) if dims else -1
        clusters.append(tuple(dims[dim_idx] for dim_idx in range(max_dim + 1)))
    return clusters
