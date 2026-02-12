# src/turntaking/analysis/io/cluster.py

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from turntaking.analysis.io.core import save_hdf5, save_table
from turntaking.stats.cluster_test import ClusterTestResult


def write_cluster_outputs(
    out_dir: Path,
    result: ClusterTestResult,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    hdf5_payload: dict[str, object] = {
        "t-values": np.asarray(result.t_values, dtype=float),
        "p-values": np.asarray(result.p_values, dtype=float),
        "h0": np.asarray(result.h0, dtype=float),
    }

    # clusters: store index arrays as int
    for idx, cluster in enumerate(result.clusters):
        for dim_i, inds in enumerate(cluster):
            hdf5_payload[f"clusters/{dim_i}-{idx}"] = np.asarray(inds, dtype=int)

    # ✅ Store metadata as a single JSON-encoded UTF-8 string (HDF5-friendly)
    meta_json = json.dumps(result.metadata, sort_keys=True)
    hdf5_payload["meta/json"] = np.bytes_(meta_json.encode("utf-8"))

    save_hdf5(out_dir / "cluster_results.hdf5", hdf5_payload)

    # Summary CSV (human-readable)
    p = result.p_values
    summary = pd.DataFrame([{
        **result.metadata,
        "n_clusters": int(p.size),
        "min_p": float(np.min(p)) if p.size else float("nan"),
        "n_p_lt_0_05": int(np.sum(p < 0.05)) if p.size else 0,
    }])
    save_table(summary, out_dir / "cluster_summary.csv")


def read_cluster_outputs(path: Path) -> ClusterTestResult:
    """
    Read cluster test outputs written by `write_cluster_outputs()`.

    Parameters
    ----------
    path
        Path to the `cluster_results.hdf5` file.

    Returns
    -------
    result
        ClusterTestResult containing t-values, p-values, h0, clusters, and metadata.

    Notes
    -----
    This function is the inverse of `write_cluster_outputs()`. It reconstructs cluster
    index arrays saved under `clusters/{dim_i}-{idx}` and decodes JSON metadata stored
    as UTF-8 bytes at `meta/json`.

    Usage example
    -------------
        result = read_cluster_outputs(Path(".../cluster_results.hdf5"))
        t = result.t_values
        clusters = result.clusters
    """
    if not path.exists():
        raise FileNotFoundError(f"Cluster results not found: {path}")

    try:
        import h5py  # type: ignore
    except Exception as e:
        raise ImportError(
            "h5py is required to read cluster_results.hdf5. "
            "Add `h5py` to your environment."
        ) from e

    with h5py.File(path, "r") as f:
        t_values = np.asarray(f["t-values"], dtype=float)
        p_values = np.asarray(f["p-values"], dtype=float)
        h0 = np.asarray(f["h0"], dtype=float)

        # Metadata JSON (optional for backward compatibility)
        metadata: dict[str, Any]
        if "meta/json" in f:
            meta_raw = f["meta/json"][()]
            if isinstance(meta_raw, (bytes, np.bytes_)):
                metadata = json.loads(meta_raw.decode("utf-8"))
            else:
                metadata = json.loads(bytes(meta_raw).decode("utf-8"))
        else:
            metadata = {}

        clusters_group = f.get("clusters", None)
        clusters: list[tuple[np.ndarray, ...]] = []
        if clusters_group is not None:
            by_cluster: dict[int, dict[int, np.ndarray]] = {}
            for name in clusters_group.keys():
                dim_str, idx_str = name.split("-", 1)
                dim_i = int(dim_str)
                cluster_idx = int(idx_str)
                inds = np.asarray(clusters_group[name], dtype=int)
                by_cluster.setdefault(cluster_idx, {})[dim_i] = inds

            for cluster_idx in sorted(by_cluster.keys()):
                dims = by_cluster[cluster_idx]
                max_dim = max(dims.keys()) if dims else -1
                cluster_tuple = tuple(dims[dim_i] for dim_i in range(max_dim + 1))
                clusters.append(cluster_tuple)

    return ClusterTestResult(
        t_values=t_values,
        p_values=p_values,
        h0=h0,
        clusters=clusters,
        metadata=metadata,
    )
