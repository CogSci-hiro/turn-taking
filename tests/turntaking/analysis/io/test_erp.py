
"""Tests for ERP output naming and file-contract writer."""

from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
import pytest

from turntaking.analysis.erp.outputs import get_erp_condition_names, write_erp_outputs


def _evoked(comment: str) -> mne.Evoked:
    info = mne.create_info(["Cz", "Pz"], sfreq=32.0, ch_types=["eeg", "eeg"])
    data = np.ones((2, 5), dtype=float)
    return mne.EvokedArray(data, info, tmin=-0.1, comment=comment)


def test_get_erp_condition_names_mapping_and_validation():
    """Confirms contrast names map to stable filename prefixes used by downstream rules."""
    assert get_erp_condition_names("duration").cond_1 == "long"
    assert get_erp_condition_names("duration").cond_2 == "short"
    assert get_erp_condition_names("latency").cond_1 == "fast"
    assert get_erp_condition_names("latency").cond_2 == "slow"
    with pytest.raises(ValueError, match="Unknown contrast"):
        get_erp_condition_names("invalid")


def test_write_erp_outputs_writes_expected_artifacts(tmp_path):
    """Validates ERP writer produces every required file for a successful pipeline stage."""
    out_dir = tmp_path / "erp" / "duration"
    ev1 = [_evoked("long")]
    ev2 = [_evoked("short")]
    evd = [_evoked("long-short")]
    evoked_data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    n_trials = pd.DataFrame({"subject": ["sub-001"], "long": [10], "short": [11]})
    offsets = pd.DataFrame({"latency": [0.1], "condition": ["long"]})

    write_erp_outputs(
        out_dir,
        contrast="duration",
        evokeds_cond_1=ev1,
        evokeds_cond_2=ev2,
        evokeds_difference=evd,
        evoked_data=evoked_data,
        n_trials=n_trials,
        results={"seed": 0, "kind": "erp"},
        offsets=offsets,
    )

    expected = [
        "long_ave.fif",
        "short_ave.fif",
        "difference_ave.fif",
        "evoked-data.npy",
        "n_trials.csv",
        "metadata.hdf5",
        "offsets.csv",
    ]
    for name in expected:
        assert (out_dir / name).exists(), name

    np.testing.assert_array_equal(np.load(out_dir / "evoked-data.npy"), evoked_data)
    pd.testing.assert_frame_equal(pd.read_csv(out_dir / "n_trials.csv"), n_trials)
    pd.testing.assert_frame_equal(pd.read_csv(out_dir / "offsets.csv"), offsets)
    with h5py.File(out_dir / "metadata.hdf5", "r") as h5:
        assert "seed" in h5
        assert "kind" in h5


def test_write_erp_outputs_validates_input_lengths(tmp_path):
    """Ensures mismatched per-subject lists fail fast before writing partial outputs."""
    out_dir = tmp_path / "erp" / "latency"
    ev = [_evoked("x")]
    with pytest.raises(ValueError, match="Cond lists must match in length"):
        write_erp_outputs(
            out_dir,
            contrast="latency",
            evokeds_cond_1=ev,
            evokeds_cond_2=[],
            evokeds_difference=ev,
            evoked_data=np.zeros((1, 1, 1)),
            n_trials=pd.DataFrame(),
            results={},
            offsets=pd.DataFrame(),
        )

    with pytest.raises(ValueError, match="Difference list must match subject count"):
        write_erp_outputs(
            out_dir,
            contrast="latency",
            evokeds_cond_1=ev,
            evokeds_cond_2=ev,
            evokeds_difference=[],
            evoked_data=np.zeros((1, 1, 1)),
            n_trials=pd.DataFrame(),
            results={},
            offsets=pd.DataFrame(),
        )


def test_write_erp_outputs_respects_overwrite_flag_for_metadata(tmp_path):
    """Checks overwrite safety on metadata artifact to avoid accidental destructive writes."""
    out_dir = tmp_path / "erp" / "duration"
    out_dir.mkdir(parents=True)
    (out_dir / "metadata.hdf5").write_text("already-there", encoding="utf-8")

    ev = [_evoked("x")]
    with pytest.raises(FileExistsError, match="Refusing to overwrite results"):
        write_erp_outputs(
            out_dir,
            contrast="duration",
            evokeds_cond_1=ev,
            evokeds_cond_2=ev,
            evokeds_difference=ev,
            evoked_data=np.zeros((1, 1, 1)),
            n_trials=pd.DataFrame({"a": [1]}),
            results={"x": 1},
            offsets=pd.DataFrame({"b": [1]}),
            overwrite=False,
        )
