# src/turntaking/analysis/io/cluster.py

from pathlib import Path
import numpy as np
import pandas as pd

from turntaking.analysis.io.core import save_hdf5, save_table
from turntaking.stats.cluster_test import ClusterTestResult


def write_cluster_outputs(
    out_dir: Path,
    result: ClusterTestResult,
) -> None:
    """
    Write cluster test outputs deterministically.

    Outputs
    -------
    - cluster_results.hdf5
    - cluster_summary.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # HDF5 payload
    hdf5_payload: dict[str, object] = {
        "t-values": result.t_values,
        "p-values": result.p_values,
        "h0": result.h0,
    }

    # clusters as indexed datasets
    # clusters are tuples of index arrays (time_inds, space_inds) for spatio_temporal_cluster_1samp_test
    for idx, cluster in enumerate(result.clusters):
        for dim_i, inds in enumerate(cluster):
            hdf5_payload[f"clusters/{dim_i}-{idx}"] = np.asarray(inds, dtype=int)

    # metadata (store as small arrays / strings)
    for k, v in result.metadata.items():
        hdf5_payload[f"meta/{k}"] = np.array([v], dtype=object)

    save_hdf5(out_dir / "cluster_results.hdf5", hdf5_payload)

    # Summary CSV
    p = result.p_values
    summary = pd.DataFrame([{
        **result.metadata,
        "n_clusters": int(p.size),
        "min_p": float(np.min(p)) if p.size else float("nan"),
        "n_p_lt_0_05": int(np.sum(p < 0.05)) if p.size else 0,
    }])
    save_table(out_dir / "cluster_summary.csv", summary)
