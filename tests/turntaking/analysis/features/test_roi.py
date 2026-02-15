
"""Tests for ROI channel index selection helper."""

import pytest

from turntaking.analysis.features.roi import pick_roi


class _Epochs:
    ch_names = ["Fz", "Cz", "Pz"]


def test_pick_roi_returns_indices_for_requested_channels():
    """Confirms ROI channel selection is mapped to integer indices expected by NumPy slicing."""
    out = pick_roi(_Epochs(), ["Fz", "Pz"])
    assert out == [0, 2]


def test_pick_roi_raises_when_roi_empty():
    """Prevents downstream averaging over empty channel sets."""
    with pytest.raises(ValueError, match="ROI picks are empty"):
        pick_roi(_Epochs(), ["Oz"])
