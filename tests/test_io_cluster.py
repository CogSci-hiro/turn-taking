from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turntaking.analysis.io.cluster import read_cluster_outputs, write_cluster_outputs
from turntaking.analysis.io.decoding_cluster import load_decoding_cluster_results_hdf5
from turntaking.stats.cluster_test import ClusterTestResult


def _sample_cluster_result() -> ClusterTestResult:
    return ClusterTestResult(
        t_values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        clusters=[(np.array([0, 1]), np.array([1, 0]))],
        p_values=np.array([0.04]),
        h0=np.array([0.1, 0.2, 0.3]),
        metadata={"kind": "erp", "seed": 0},
    )


def test_write_and_read_cluster_outputs_round_trip(tmp_path):
    """Ensures cluster result serialization is reversible so saved statistics remain trustworthy."""
    out_dir = tmp_path / "cluster"
    result = _sample_cluster_result()
    write_cluster_outputs(out_dir, result)

    loaded = read_cluster_outputs(out_dir / "cluster_results.hdf5")
    np.testing.assert_array_equal(loaded.t_values, result.t_values)
    np.testing.assert_array_equal(loaded.p_values, result.p_values)
    np.testing.assert_array_equal(loaded.h0, result.h0)
    assert loaded.metadata == result.metadata
    assert len(loaded.clusters) == 1
    np.testing.assert_array_equal(loaded.clusters[0][0], np.array([0, 1]))
    np.testing.assert_array_equal(loaded.clusters[0][1], np.array([1, 0]))

    summary = pd.read_csv(out_dir / "cluster_summary.csv")
    assert set(["n_clusters", "min_p", "n_p_lt_0_05"]).issubset(summary.columns)
    assert int(summary.loc[0, "n_clusters"]) == 1


def test_read_cluster_outputs_requires_existing_file(tmp_path):
    """Gives explicit failure when expected cluster output files are missing."""
    with pytest.raises(FileNotFoundError, match="Cluster results not found"):
        read_cluster_outputs(tmp_path / "none.h5")


def test_load_decoding_cluster_results_hdf5_reads_indexed_clusters(tmp_path):
    """Validates decoding cluster loader parses train/test index pairs and core arrays."""
    import h5py

    path = tmp_path / "decoding_cluster.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("t-values", data=np.eye(3))
        h5.create_dataset("p-values", data=np.array([0.01, 0.20]))
        h5.create_dataset("h0", data=np.array([0.1, 0.2]))
        h5.create_dataset("clusters/train-0", data=np.array([0, 1]))
        h5.create_dataset("clusters/test-0", data=np.array([1, 2]))

    out = load_decoding_cluster_results_hdf5(path)
    np.testing.assert_array_equal(out.t_values, np.eye(3))
    np.testing.assert_array_equal(out.p_values, np.array([0.01, 0.20]))
    np.testing.assert_array_equal(out.h0, np.array([0.1, 0.2]))
    assert len(out.clusters) == 1
    np.testing.assert_array_equal(out.clusters[0][0], np.array([0, 1]))
    np.testing.assert_array_equal(out.clusters[0][1], np.array([1, 2]))
