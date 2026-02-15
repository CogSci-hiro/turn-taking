
"""Tests for decoding dataset assembly from split epoch groups."""

from pathlib import Path

import numpy as np
import pytest

from turntaking.analysis.datasets.decoding_dataset import (
    DecodingDatasetParams,
    make_decoding_data,
)


class _FakeEpochs:
    # Small stand-in for mne.Epochs to keep this unit test pure and fast.
    def __init__(self, data: np.ndarray, times: np.ndarray):
        self._data = np.asarray(data)
        self.times = np.asarray(times)

    def get_data(self) -> np.ndarray:
        return self._data


def _params() -> DecodingDatasetParams:
    return DecodingDatasetParams(
        contrast="duration",
        min_latency_s=-1.0,
        max_latency_s=1.0,
        min_response_duration_s=0.05,
        sfreq_hz=64.0,
    )


def test_make_decoding_data_stacks_trials_and_labels():
    """Verifies dataset assembly preserves trial order and label semantics required for binary decoding."""
    times = np.array([-0.1, 0.0, 0.1], dtype=float)
    cond_1 = _FakeEpochs(np.ones((2, 2, 3)), times)
    cond_2 = _FakeEpochs(np.full((3, 2, 3), 2.0), times)

    def split_fn(*_args, **_kwargs):
        return cond_1, cond_2

    X, y, times_out = make_decoding_data(
        subject="sub-001",
        epoch_dir=Path("/tmp"),
        params=_params(),
        make_subject_split_fn=split_fn,
    )

    assert X.shape == (5, 2, 3)
    np.testing.assert_array_equal(y, np.array([0, 0, 1, 1, 1], dtype=np.int64))
    np.testing.assert_array_equal(times_out, times)


def test_make_decoding_data_rejects_mismatched_time_axes():
    """Ensures both classes share the exact same time axis so temporal generalization is well-defined."""
    cond_1 = _FakeEpochs(np.ones((2, 1, 3)), np.array([0.0, 0.1, 0.2]))
    cond_2 = _FakeEpochs(np.ones((2, 1, 3)), np.array([0.0, 0.1, 0.25]))

    def split_fn(*_args, **_kwargs):
        return cond_1, cond_2

    with pytest.raises(ValueError, match="Time axis mismatch"):
        make_decoding_data(
            subject="sub-001",
            epoch_dir=Path("/tmp"),
            params=_params(),
            make_subject_split_fn=split_fn,
        )
