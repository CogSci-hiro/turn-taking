"""Scientific integration test.

These tests verify that outputs produced by Snakemake match our frozen baselines
within tight numerical tolerances. They guard against unintended scientific
changes during refactoring.

Do not relax tolerances without an experimental justification.
"""


from pathlib import Path

import numpy as np
import pandas as pd


def _assert_file_exists(path: Path, label: str) -> None:
    assert path.exists(), f"Missing {label}: {path}"


def _compare_evoked_fif(actual: Path, expected: Path, compare_helpers: dict, label: str) -> None:
    read_evoked = compare_helpers["read_evoked"]
    assert_allclose_arrays = compare_helpers["assert_allclose_arrays"]
    assert_no_nans_array = compare_helpers["assert_no_nans_array"]

    a = read_evoked(actual)
    e = read_evoked(expected)
    assert a.ch_names == e.ch_names, f"{label} channel names differ."
    assert np.allclose(a.times, e.times, atol=1e-12, rtol=0.0), f"{label} time vector differs."
    assert_allclose_arrays(a.data, e.data, f"{label}.data")
    assert_no_nans_array(a.data, f"{label}.data")


def _compare_metadata_hdf5(actual: Path, expected: Path, hdf5_helpers: dict, compare_helpers: dict, label: str) -> None:
    hdf5_keys = hdf5_helpers["hdf5_dataset_keys"]
    load_h5 = hdf5_helpers["load_hdf5_dataset"]
    assert_allclose_arrays = compare_helpers["assert_allclose_arrays"]

    a_keys = hdf5_keys(actual)
    e_keys = hdf5_keys(expected)
    assert a_keys == e_keys, f"{label} dataset keys differ: actual={sorted(a_keys)} expected={sorted(e_keys)}"

    for key in sorted(a_keys):
        a_val = load_h5(actual, key)
        e_val = load_h5(expected, key)
        if isinstance(a_val, np.ndarray) and np.issubdtype(a_val.dtype, np.number):
            assert_allclose_arrays(a_val.astype(float), e_val.astype(float), f"{label}.{key}")
        else:
            assert np.array_equal(a_val, e_val), f"{label}.{key} differs."

    a_times = load_h5(actual, "times").astype(float)
    e_times = load_h5(expected, "times").astype(float)
    assert np.allclose(a_times, e_times, atol=1e-12, rtol=0.0), f"{label}.times diverged."


def _compare_cluster_hdf5(actual: Path, expected: Path, hdf5_helpers: dict, compare_helpers: dict, label: str) -> None:
    hdf5_keys = hdf5_helpers["hdf5_dataset_keys"]
    load_h5 = hdf5_helpers["load_hdf5_dataset"]
    assert_allclose_arrays = compare_helpers["assert_allclose_arrays"]

    a_keys = hdf5_keys(actual)
    e_keys = hdf5_keys(expected)
    assert a_keys == e_keys, f"{label} HDF5 key set mismatch."

    a_t = load_h5(actual, "t-values").astype(float)
    e_t = load_h5(expected, "t-values").astype(float)
    assert np.array_equal(np.isnan(a_t), np.isnan(e_t)), (
        f"{label}.t-values NaN mask changed; this indicates changed statistical behavior."
    )
    assert_allclose_arrays(a_t, e_t, f"{label}.t-values")

    for key in ["p-values", "h0"]:
        a = load_h5(actual, key).astype(float)
        e = load_h5(expected, key).astype(float)
        assert_allclose_arrays(a, e, f"{label}.{key}")


def _compare_cluster_summary(actual: Path, expected: Path, compare_helpers: dict, label: str) -> None:
    assert_csv_equal = compare_helpers["assert_csv_equal"]
    assert_csv_equal(actual, expected, label, allow_nans=True)

    a = pd.read_csv(actual)
    e = pd.read_csv(expected)
    assert list(a.columns) == list(e.columns), f"{label} summary key columns changed."
    assert int(a["n_clusters"].iloc[0]) == int(e["n_clusters"].iloc[0]), (
        f"{label} number of clusters changed: actual={int(a['n_clusters'].iloc[0])} expected={int(e['n_clusters'].iloc[0])}."
    )


def test_tfr_pipeline_regression(
    runtime_config_path: Path,
    out_dir: Path,
    baseline_path,
    tmp_path: Path,
    run_snakemake,
    compare_helpers: dict,
    hdf5_helpers: dict,
):
    """Compares TFR outputs and TFR cluster outputs against scientific baselines."""
    run_snakemake("tfr_all", runtime_config_path, tmp_path)
    run_snakemake("tfr_cluster_all", runtime_config_path, tmp_path)

    assert_csv_equal = compare_helpers["assert_csv_equal"]
    assert_allclose_arrays = compare_helpers["assert_allclose_arrays"]
    assert_no_nans_array = compare_helpers["assert_no_nans_array"]

    contrast_to_conditions = {
        "duration": ("long", "short"),
        "latency": ("fast", "slow"),
    }

    for contrast, (cond_1, cond_2) in contrast_to_conditions.items():
        for band in ("alpha", "beta"):
            out_npy = out_dir / "tfr" / contrast / band / "induced-data.npy"
            exp_npy = baseline_path(f"tfr/{contrast}/{band}/induced-data.npy")
            _assert_file_exists(out_npy, f"TFR induced-data ({contrast}/{band})")
            out_arr = np.load(out_npy)
            exp_arr = np.load(exp_npy)
            assert_allclose_arrays(out_arr, exp_arr, f"tfr.{contrast}.{band}.induced-data")
            assert_no_nans_array(out_arr, f"tfr.{contrast}.{band}.induced-data")

            out_trials = out_dir / "tfr" / contrast / band / "n_trials.csv"
            exp_trials = baseline_path(f"tfr/{contrast}/{band}/n_trials.csv")
            _assert_file_exists(out_trials, f"TFR n_trials ({contrast}/{band})")
            assert_csv_equal(out_trials, exp_trials, f"tfr.{contrast}.{band}.n_trials")

            fif_pairs = [
                (out_dir / "tfr" / contrast / band / "difference_ave.fif", baseline_path(f"tfr/{contrast}/{band}/difference_ave.fif"), "difference"),
                (out_dir / "tfr" / contrast / band / f"{cond_1}_ave.fif", baseline_path(f"tfr/{contrast}/{band}/{cond_1}_ave.fif"), cond_1),
                (out_dir / "tfr" / contrast / band / f"{cond_2}_ave.fif", baseline_path(f"tfr/{contrast}/{band}/{cond_2}_ave.fif"), cond_2),
            ]
            for out_fif, exp_fif, name in fif_pairs:
                _assert_file_exists(out_fif, f"TFR FIF ({contrast}/{band}/{name})")
                _compare_evoked_fif(out_fif, exp_fif, compare_helpers, f"tfr.{contrast}.{band}.{name}")

            out_meta = out_dir / "tfr" / contrast / band / "metadata.hdf5"
            exp_meta = baseline_path(f"tfr/{contrast}/{band}/metadata.hdf5")
            _assert_file_exists(out_meta, f"TFR metadata ({contrast}/{band})")
            _compare_metadata_hdf5(out_meta, exp_meta, hdf5_helpers, compare_helpers, f"tfr.{contrast}.{band}.metadata")

            out_summary = out_dir / "stats" / "tfr" / contrast / band / "cluster_summary.csv"
            exp_summary = baseline_path(f"stats/tfr/{contrast}/{band}/cluster_summary.csv")
            _assert_file_exists(out_summary, f"TFR cluster summary ({contrast}/{band})")
            _compare_cluster_summary(out_summary, exp_summary, compare_helpers, f"tfr.{contrast}.{band}.cluster_summary")

            out_cluster = out_dir / "stats" / "tfr" / contrast / band / "cluster_results.hdf5"
            exp_cluster = baseline_path(f"stats/tfr/{contrast}/{band}/cluster_results.hdf5")
            _assert_file_exists(out_cluster, f"TFR cluster results ({contrast}/{band})")
            _compare_cluster_hdf5(out_cluster, exp_cluster, hdf5_helpers, compare_helpers, f"tfr.{contrast}.{band}.cluster_results")

            trials_df = pd.read_csv(out_trials)
            count_cols = [c for c in trials_df.columns if c != "subject"]
            assert int(trials_df[count_cols].iloc[0].sum()) > 0, (
                f"tfr.{contrast}.{band}: n_trials indicates zero observations."
            )
