
"""Tests for induced-TFR output naming and writer contract."""

import h5py
import mne
import numpy as np
import pandas as pd
import pytest

from turntaking.analysis.tfr.io import get_tfr_condition_names, write_tfr_outputs


def _evoked(comment: str) -> mne.Evoked:
    info = mne.create_info(["Cz", "Pz"], sfreq=64.0, ch_types=["eeg", "eeg"])
    data = np.full((2, 6), 0.5, dtype=float)
    return mne.EvokedArray(data, info, tmin=-0.2, comment=comment)


def test_get_tfr_condition_names_mapping_and_validation():
    """Confirms contrast labels map to deterministic condition names for filenames and metadata."""
    assert get_tfr_condition_names("duration").cond_1 == "long"
    assert get_tfr_condition_names("duration").cond_2 == "short"
    assert get_tfr_condition_names("latency").cond_1 == "fast"
    assert get_tfr_condition_names("latency").cond_2 == "slow"
    with pytest.raises(ValueError, match="Unknown contrast"):
        get_tfr_condition_names("oops")


def test_write_tfr_outputs_writes_files_and_enriched_metadata(tmp_path):
    """Validates TFR writer emits all required artifacts and enriches metadata defaults."""
    out_dir = tmp_path / "tfr" / "duration" / "alpha"
    ev1 = [_evoked("long")]
    ev2 = [_evoked("short")]
    evd = [_evoked("long-short")]
    induced_data = np.zeros((1, 3, 2, 6))
    n_trials = pd.DataFrame({"subject": ["sub-001"], "long": [12], "short": [12]})

    write_tfr_outputs(
        out_dir,
        contrast="duration",
        band="alpha",
        evokeds_cond_1=ev1,
        evokeds_cond_2=ev2,
        evokeds_difference=evd,
        induced_data=induced_data,
        n_trials=n_trials,
        metadata={"custom": "ok"},
    )

    for name in [
        "long_ave.fif",
        "short_ave.fif",
        "difference_ave.fif",
        "induced-data.npy",
        "n_trials.csv",
        "metadata.hdf5",
    ]:
        assert (out_dir / name).exists(), name

    np.testing.assert_array_equal(np.load(out_dir / "induced-data.npy"), induced_data)
    pd.testing.assert_frame_equal(pd.read_csv(out_dir / "n_trials.csv"), n_trials)
    with h5py.File(out_dir / "metadata.hdf5", "r") as h5:
        assert "custom" in h5
        assert "kind" in h5
        assert "contrast" in h5
        assert "band" in h5
        assert "induced_data_shape" in h5


def test_write_tfr_outputs_validates_shapes_and_lengths(tmp_path):
    """Prevents writing partial/invalid outputs when key input dimensions are inconsistent."""
    out_dir = tmp_path / "tfr" / "latency" / "beta"
    ev = [_evoked("x")]

    with pytest.raises(ValueError, match="Cond lists must match in length"):
        write_tfr_outputs(
            out_dir,
            contrast="latency",
            band="beta",
            evokeds_cond_1=ev,
            evokeds_cond_2=[],
            evokeds_difference=ev,
            induced_data=np.zeros((1, 3, 2, 6)),
            n_trials=pd.DataFrame(),
            metadata={},
        )

    with pytest.raises(ValueError, match="Difference list must match subject count"):
        write_tfr_outputs(
            out_dir,
            contrast="latency",
            band="beta",
            evokeds_cond_1=ev,
            evokeds_cond_2=ev,
            evokeds_difference=[],
            induced_data=np.zeros((1, 3, 2, 6)),
            n_trials=pd.DataFrame(),
            metadata={},
        )

    with pytest.raises(ValueError, match="induced_data looks wrong"):
        write_tfr_outputs(
            out_dir,
            contrast="latency",
            band="beta",
            evokeds_cond_1=ev,
            evokeds_cond_2=ev,
            evokeds_difference=ev,
            induced_data=np.zeros((2, 2)),
            n_trials=pd.DataFrame(),
            metadata={},
        )


def test_write_tfr_outputs_respects_overwrite_for_metadata(tmp_path):
    """Verifies overwrite=False blocks replacing existing metadata.hdf5."""
    out_dir = tmp_path / "tfr" / "duration" / "alpha"
    out_dir.mkdir(parents=True)
    (out_dir / "metadata.hdf5").write_text("existing", encoding="utf-8")

    ev = [_evoked("x")]
    with pytest.raises(FileExistsError, match="Refusing to overwrite metadata"):
        write_tfr_outputs(
            out_dir,
            contrast="duration",
            band="alpha",
            evokeds_cond_1=ev,
            evokeds_cond_2=ev,
            evokeds_difference=ev,
            induced_data=np.zeros((1, 3, 2, 6)),
            n_trials=pd.DataFrame({"a": [1]}),
            metadata={},
            overwrite=False,
        )
