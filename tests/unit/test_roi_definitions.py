
import mne
import pytest

from turntaking.analysis.constants import ANTERIOR, POSTERIOR
from turntaking.analysis.features.roi import pick_roi


def _epochs():
    info = mne.create_info(
        ch_names=["Fz", "AFz", "FCz", "F1", "F2", "POz", "Pz", "Oz", "PO4", "PO7"],
        sfreq=64.0,
        ch_types=["eeg"] * 10,
    )
    return type("E", (), {"ch_names": info["ch_names"]})()


def test_roi_lists_have_expected_channels_and_no_duplicates():
    assert "Fz" in ANTERIOR
    assert "Pz" in POSTERIOR
    assert len(set(ANTERIOR)) == len(ANTERIOR)
    assert len(set(POSTERIOR)) == len(POSTERIOR)


def test_roi_channels_are_valid_and_pickable():
    epochs = _epochs()
    ant_idx = pick_roi(epochs, ANTERIOR)
    post_idx = pick_roi(epochs, POSTERIOR)
    assert len(ant_idx) == len(ANTERIOR)
    assert len(post_idx) == len(POSTERIOR)


def test_empty_roi_raises():
    epochs = _epochs()
    with pytest.raises(ValueError, match="ROI picks are empty"):
        pick_roi(epochs, ["NotAChannel"])
