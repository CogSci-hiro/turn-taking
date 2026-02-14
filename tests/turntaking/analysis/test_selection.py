from __future__ import annotations

"""Tests for metadata-driven epoch selection and median splitting."""

import pandas as pd
import pytest

from turntaking.analysis.selection import SelectionParams, select_epochs, split_epochs_median


def test_select_epochs_filters_using_metadata_thresholds(sample_epochs):
    """Verifies epoch selection enforces latency and duration thresholds to prevent silent drift in inclusion logic."""
    params = SelectionParams(min_latency=0.25, max_latency=0.75, min_self_duration=0.15)
    out = select_epochs(sample_epochs, params)

    lat = out.metadata["latency"].to_numpy()
    dur = out.metadata["self_duration"].to_numpy()
    assert (lat > 0.25).all()
    assert (lat < 0.75).all()
    assert (dur > 0.15).all()
    assert len(out) == 4


def test_select_epochs_rejects_missing_metadata():
    """Guards against calling selection on epochs without metadata, which would otherwise produce invalid selections."""
    epochs = sample_epochs_no_metadata()
    params = SelectionParams(min_latency=0.0, max_latency=1.0, min_self_duration=0.0)

    with pytest.raises(ValueError, match="epochs.metadata is required"):
        select_epochs(epochs, params)


def test_select_epochs_rejects_missing_columns(sample_epochs):
    """Ensures required metadata columns are mandatory so later steps never run on incomplete behavioral annotations."""
    sample_epochs.metadata = pd.DataFrame({"latency": [0.1] * len(sample_epochs)})
    params = SelectionParams(min_latency=0.0, max_latency=1.0, min_self_duration=0.0)

    with pytest.raises(ValueError, match="missing required columns"):
        select_epochs(sample_epochs, params)


def test_split_epochs_median_latency_labels_and_balance(sample_epochs):
    """Checks latency split semantics and balanced class counts for stable downstream decoding/statistics."""
    cond_1, cond_2, labels = split_epochs_median(sample_epochs, contrast="latency")

    assert labels == {"cond_1": "fast", "cond_2": "slow"}
    assert len(cond_1) == len(cond_2)
    assert cond_1.metadata["latency"].max() < cond_2.metadata["latency"].min()


def test_split_epochs_median_duration_labels_and_order(sample_epochs):
    """Checks duration split semantics to confirm 'long' and 'short' condition assignment has not flipped."""
    cond_1, cond_2, labels = split_epochs_median(sample_epochs, contrast="duration")

    assert labels == {"cond_1": "long", "cond_2": "short"}
    assert len(cond_1) == len(cond_2)
    assert cond_1.metadata["self_duration"].min() > cond_2.metadata["self_duration"].max()


def sample_epochs_no_metadata():
    # Minimal helper used only for negative-path testing.
    import mne
    import numpy as np

    info = mne.create_info(ch_names=["Cz"], sfreq=16.0, ch_types=["eeg"])
    return mne.EpochsArray(np.zeros((2, 1, 4)), info, tmin=0.0, verbose=False)
