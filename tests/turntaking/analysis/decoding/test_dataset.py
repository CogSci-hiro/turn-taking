from __future__ import annotations

"""Tests for subject split and decoding dataset assembly in analysis.decoding.dataset."""

from pathlib import Path

import numpy as np
import pytest

from turntaking.analysis.decoding.dataset import DecodingDatasetParams, make_decoding_data, make_subject_split
from turntaking.analysis.selection import SelectionParams


class _FakeEpochs:
    def __init__(self, data: np.ndarray, times: np.ndarray):
        self._data = np.asarray(data)
        self.times = np.asarray(times)
        self.resample_calls: list[float] = []

    def copy(self):
        return _FakeEpochs(self._data.copy(), self.times.copy())

    def resample(self, sfreq, npad="auto"):
        self.resample_calls.append(float(sfreq))
        return self

    def get_data(self):
        return self._data


def _params() -> DecodingDatasetParams:
    return DecodingDatasetParams(
        contrast="duration",
        selection=SelectionParams(min_latency=0.1, max_latency=0.9, min_self_duration=0.1),
        sfreq_hz=64.0,
    )


def test_make_subject_split_calls_selection_resample_and_split(monkeypatch):
    """Validates preprocessing order for subject-level split creation."""
    epochs = _FakeEpochs(np.ones((4, 2, 3)), np.array([0.0, 0.1, 0.2]))
    called = {"select": 0, "split": 0}

    def fake_load(_subject, _epoch_dir):
        return epochs

    def fake_select(ep, params):
        called["select"] += 1
        assert ep is epochs
        assert params.min_latency == 0.1
        return ep

    def fake_split(ep, contrast):
        called["split"] += 1
        assert contrast == "duration"
        assert ep.resample_calls == [64.0]
        return "C1", "C2", {"cond_1": "long", "cond_2": "short"}

    monkeypatch.setattr("turntaking.analysis.decoding.dataset.select_epochs", fake_select)
    monkeypatch.setattr("turntaking.analysis.decoding.dataset.split_epochs_median", fake_split)

    c1, c2 = make_subject_split(
        subject="sub-001",
        epoch_dir=Path("/tmp"),
        params=_params(),
        load_subject_epochs_fn=fake_load,
    )
    assert (c1, c2) == ("C1", "C2")
    assert called == {"select": 1, "split": 1}


def test_make_decoding_data_stacks_trials_labels_and_times(monkeypatch):
    """Checks decoding matrix assembly and class label encoding semantics."""
    times = np.array([0.0, 0.1, 0.2], dtype=float)

    def fake_make_subject_split(**_kwargs):
        c1 = _FakeEpochs(np.ones((2, 2, 3)), times)
        c2 = _FakeEpochs(np.full((3, 2, 3), 2.0), times)
        return c1, c2

    monkeypatch.setattr("turntaking.analysis.decoding.dataset.make_subject_split", fake_make_subject_split)
    X, y, t = make_decoding_data(
        subject="sub-001",
        epoch_dir=Path("/tmp"),
        params=_params(),
        load_subject_epochs_fn=lambda _s, _d: None,
    )
    assert X.shape == (5, 2, 3)
    assert y.tolist() == [0, 0, 1, 1, 1]
    np.testing.assert_array_equal(t, times)


def test_make_decoding_data_rejects_time_mismatch(monkeypatch):
    """Prevents building invalid decoding matrices when split classes have different time axes."""
    t1 = np.array([0.0, 0.1, 0.2])
    t2 = np.array([0.0, 0.1, 0.25])

    def fake_make_subject_split(**_kwargs):
        return _FakeEpochs(np.ones((2, 1, 3)), t1), _FakeEpochs(np.ones((2, 1, 3)), t2)

    monkeypatch.setattr("turntaking.analysis.decoding.dataset.make_subject_split", fake_make_subject_split)
    with pytest.raises(ValueError, match="Time axis mismatch after split"):
        make_decoding_data(
            subject="sub-001",
            epoch_dir=Path("/tmp"),
            params=_params(),
            load_subject_epochs_fn=lambda _s, _d: None,
        )
