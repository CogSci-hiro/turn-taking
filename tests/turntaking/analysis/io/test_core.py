
"""Tests for low-level table/array/HDF5 serialization helpers."""

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from turntaking.analysis.io.core import save_hdf5, save_npy, save_table, save_table_csv


def test_save_table_writes_csv_and_parquet(tmp_path):
    """Confirms supported table formats round-trip correctly for downstream workflow rules."""
    df = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})
    csv_path = tmp_path / "tables" / "x.csv"
    pq_path = tmp_path / "tables" / "x.parquet"

    save_table(df, csv_path)
    save_table(df, pq_path)

    pd.testing.assert_frame_equal(pd.read_csv(csv_path), df)
    pd.testing.assert_frame_equal(pd.read_parquet(pq_path), df)


def test_save_table_rejects_unknown_extension(tmp_path):
    """Prevents ambiguous serialization behavior by failing on unsupported extensions."""
    with pytest.raises(ValueError, match="Unsupported table extension"):
        save_table(pd.DataFrame({"x": [1]}), tmp_path / "x.unsupported")


def test_save_table_csv_and_save_npy_create_parent_dirs(tmp_path):
    """Ensures convenience writers always create parent directories to avoid brittle caller code."""
    df = pd.DataFrame({"k": [1]})
    arr = np.array([[1, 2], [3, 4]])
    csv_path = tmp_path / "nested" / "dir" / "a.csv"
    npy_path = tmp_path / "nested" / "dir" / "a.npy"

    save_table_csv(df, csv_path)
    save_npy(arr, npy_path)

    assert csv_path.exists()
    assert npy_path.exists()
    np.testing.assert_array_equal(np.load(npy_path), arr)


def test_save_hdf5_serializes_arrays_lists_and_json_payloads(tmp_path):
    """Verifies deterministic HDF5 encoding for mixed payload types used by analysis outputs."""
    path = tmp_path / "mixed" / "payload.h5"
    payload = {
        "arr_float": np.array([1.0, 2.0]),
        "list_numeric": [1, 2, 3],
        "list_object": [{"a": 1}, {"b": 2}],
        "obj_array": np.array([{"x": 1}, {"y": 2}], dtype=object),
        "meta": {"subject": "sub-001", "seed": 0},
        "none_skipped": None,
    }
    save_hdf5(path, payload)

    with h5py.File(path, "r") as h5:
        np.testing.assert_array_equal(h5["arr_float"][()], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(h5["list_numeric"][()], np.array([1, 2, 3]))
        assert "none_skipped" not in h5

        list_obj_decoded = json.loads(bytes(h5["list_object"][()]).decode("utf-8"))
        obj_array_decoded = json.loads(bytes(h5["obj_array"][()]).decode("utf-8"))
        meta_decoded = json.loads(bytes(h5["meta"][()]).decode("utf-8"))

    assert list_obj_decoded == [{"a": 1}, {"b": 2}]
    assert obj_array_decoded == [{"x": 1}, {"y": 2}]
    assert meta_decoded == {"seed": 0, "subject": "sub-001"}
