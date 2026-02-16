
"""Tests for ERP I/O wrappers around the pure core."""

from pathlib import Path

import numpy as np
import pandas as pd

from turntaking.analysis.erp.io import load_epochs, run_erp_analysis


def _save_epochs_fixture(path: Path, sample_epochs) -> None:
    epochs = sample_epochs.copy()
    epochs.metadata = epochs.metadata.copy()
    epochs.metadata["subject"] = "sub-001"
    epochs.metadata["run"] = 1
    epochs.save(path, overwrite=True, verbose=False)


def test_load_epochs_returns_epoch_bundle_with_metadata_and_info(tmp_path, sample_epochs):
    epoch_path = tmp_path / "sub-001_task-conversation_run-1_epochs-epo.fif"
    _save_epochs_fixture(epoch_path, sample_epochs)

    bundle = load_epochs(str(epoch_path))
    assert bundle.metadata is not None
    assert "latency" in bundle.metadata.columns
    assert bundle.info["sfreq"] == sample_epochs.info["sfreq"]
    assert bundle.epochs.get_data(copy=True).shape == sample_epochs.get_data(copy=True).shape


def test_run_erp_analysis_matches_contrast_math_and_writes_outputs(tmp_path, sample_epochs):
    epoch_path = tmp_path / "sub-001_task-conversation_run-1_epochs-epo.fif"
    _save_epochs_fixture(epoch_path, sample_epochs)
    out_dir = tmp_path / "erp" / "duration"

    result = run_erp_analysis(
        str(epoch_path),
        {
            "contrast": "duration",
            "baseline": [-0.2, 0.0],
            "min_latency": 0.0,
            "max_latency": 1.0,
            "min_self_duration": 0.0,
            "summary_window": [0.0, 0.2],
        },
        save_path=str(out_dir),
    )

    assert set(result.keys()) == {"erp_condition1", "erp_condition2", "contrast", "times"}
    np.testing.assert_allclose(result["contrast"], result["erp_condition1"] - result["erp_condition2"])

    expected_files = [
        "long_ave.fif",
        "short_ave.fif",
        "difference_ave.fif",
        "evoked-data.npy",
        "n_trials.csv",
        "metadata.hdf5",
        "offsets.csv",
        "summary.csv",
    ]
    for name in expected_files:
        assert (out_dir / name).exists(), name

    evoked_data = np.load(out_dir / "evoked-data.npy")
    np.testing.assert_allclose(evoked_data[0, 0], result["erp_condition1"])
    np.testing.assert_allclose(evoked_data[0, 1], result["erp_condition2"])
    np.testing.assert_allclose(evoked_data[0, 2], result["contrast"])

    summary = pd.read_csv(out_dir / "summary.csv")
    assert set(summary.columns) == {"name", "mean_amplitude", "peak_latency", "peak_amplitude"}
    assert set(summary["name"]) == {"long", "short", "difference"}
