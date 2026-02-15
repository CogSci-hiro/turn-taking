from __future__ import annotations

"""Tests for decoding execution and group orchestration utilities."""

from pathlib import Path

import numpy as np
import pytest

from turntaking.analysis.decoding.dataset import DecodingDatasetParams
from turntaking.analysis.decoding.run_decoding import (
    DecodingRunParams,
    decode_subject_temporal_generalization,
    run_group_decoding,
)
from turntaking.analysis.selection import SelectionParams


def _dataset_params() -> DecodingDatasetParams:
    return DecodingDatasetParams(
        contrast="latency",
        selection=SelectionParams(min_latency=0.1, max_latency=1.0, min_self_duration=0.1),
        sfreq_hz=64.0,
    )


def _run_params() -> DecodingRunParams:
    return DecodingRunParams(n_splits=3, seed=0, n_jobs=1)


def test_decode_subject_temporal_generalization_validates_input_shapes():
    """Guards against malformed trial/label matrices before expensive model fitting."""
    params = _run_params()
    with pytest.raises(ValueError, match="Expected X as"):
        decode_subject_temporal_generalization(np.zeros((3, 4)), np.zeros((3,)), params)
    with pytest.raises(ValueError, match="Expected y as"):
        decode_subject_temporal_generalization(np.zeros((3, 2, 4)), np.zeros((3, 1)), params)
    with pytest.raises(ValueError, match="Trial mismatch"):
        decode_subject_temporal_generalization(np.zeros((4, 2, 4)), np.zeros((3,)), params)


def test_decode_subject_temporal_generalization_returns_scores(monkeypatch):
    """Checks decoder delegates to cross-validation function and returns expected score cube."""
    params = _run_params()
    X = np.zeros((8, 2, 5))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    class _DummyGE:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    def fake_cross_val_multiscore(decoder, X_in, y_in, cv, n_jobs, verbose):
        assert isinstance(decoder, _DummyGE)
        assert X_in.shape == X.shape
        assert y_in.shape == y.shape
        assert n_jobs == 1
        assert verbose == "ERROR"
        return np.ones((3, 5, 5))

    monkeypatch.setattr("turntaking.analysis.decoding.run_decoding.GeneralizingEstimator", _DummyGE)
    monkeypatch.setattr("turntaking.analysis.decoding.run_decoding.cross_val_multiscore", fake_cross_val_multiscore)
    out = decode_subject_temporal_generalization(X, y, params)
    assert out.shape == (3, 5, 5)


def test_run_group_decoding_uses_cache_path_and_sorts_subjects(monkeypatch, tmp_path):
    """Ensures cached feature loading path is used and subject ordering is deterministic."""
    calls: list[str] = []

    def fake_load_cache(subject: str):
        calls.append(subject)
        X = np.zeros((6, 2, 4))
        y = np.array([0, 1, 0, 1, 0, 1])
        t = np.array([0.0, 0.1, 0.2, 0.3])
        return X, y, t

    def fake_decode(X, y, params):
        return np.full((params.n_splits, 4, 4), fill_value=float(X.shape[0]))

    monkeypatch.setattr("turntaking.analysis.decoding.run_decoding.decode_subject_temporal_generalization", fake_decode)

    scores, times = run_group_decoding(
        subjects=["sub-002", "sub-001"],
        epoch_dir=tmp_path,
        dataset_params=_dataset_params(),
        run_params=_run_params(),
        load_subject_epochs_fn=lambda _s, _d: None,
        load_cached_features_fn=fake_load_cache,
    )

    assert calls == ["sub-001", "sub-002"]
    assert scores.shape == (2, 3, 4, 4)
    np.testing.assert_array_equal(times, np.array([0.0, 0.1, 0.2, 0.3]))


def test_run_group_decoding_builds_and_saves_features_when_no_cache(monkeypatch, tmp_path):
    """Checks non-cache path computes features and calls save hook once per subject."""
    saved: list[tuple[str, tuple[int, ...]]] = []

    def fake_make_decoding_data(**kwargs):
        subj = kwargs["subject"]
        X = np.zeros((4, 2, 3)) + (1 if subj == "sub-001" else 2)
        y = np.array([0, 1, 0, 1])
        t = np.array([0.0, 0.1, 0.2])
        return X, y, t

    def fake_decode(X, y, params):
        return np.ones((params.n_splits, 3, 3))

    def fake_save(subject, X, y, times_s):
        saved.append((subject, X.shape))

    monkeypatch.setattr("turntaking.analysis.decoding.run_decoding.make_decoding_data", fake_make_decoding_data)
    monkeypatch.setattr("turntaking.analysis.decoding.run_decoding.decode_subject_temporal_generalization", fake_decode)

    scores, times = run_group_decoding(
        subjects=["sub-001"],
        epoch_dir=tmp_path,
        dataset_params=_dataset_params(),
        run_params=_run_params(),
        load_subject_epochs_fn=lambda _s, _d: None,
        save_cached_features_fn=fake_save,
    )
    assert scores.shape == (1, 3, 3, 3)
    np.testing.assert_array_equal(times, np.array([0.0, 0.1, 0.2]))
    assert saved == [("sub-001", (4, 2, 3))]


def test_run_group_decoding_rejects_time_axis_mismatch(monkeypatch, tmp_path):
    """Prevents stacking scores from subjects with incompatible time vectors."""
    seq = [
        (np.zeros((4, 2, 2)), np.array([0, 1, 0, 1]), np.array([0.0, 0.1])),
        (np.zeros((4, 2, 3)), np.array([0, 1, 0, 1]), np.array([0.0, 0.1, 0.2])),
    ]

    def fake_make_decoding_data(**_kwargs):
        return seq.pop(0)

    monkeypatch.setattr("turntaking.analysis.decoding.run_decoding.make_decoding_data", fake_make_decoding_data)
    monkeypatch.setattr(
        "turntaking.analysis.decoding.run_decoding.decode_subject_temporal_generalization",
        lambda X, y, params: np.zeros((params.n_splits, X.shape[2], X.shape[2])),
    )

    with pytest.raises(ValueError, match="Time axis mismatch"):
        run_group_decoding(
            subjects=["sub-001", "sub-002"],
            epoch_dir=tmp_path,
            dataset_params=_dataset_params(),
            run_params=_run_params(),
            load_subject_epochs_fn=lambda _s, _d: None,
        )


def test_run_group_decoding_validates_split_count_and_empty_subjects(monkeypatch, tmp_path):
    """Ensures returned score tensor shape is consistent with configured CV split count."""
    monkeypatch.setattr(
        "turntaking.analysis.decoding.run_decoding.make_decoding_data",
        lambda **_kwargs: (np.zeros((4, 2, 2)), np.array([0, 1, 0, 1]), np.array([0.0, 0.1])),
    )
    monkeypatch.setattr(
        "turntaking.analysis.decoding.run_decoding.decode_subject_temporal_generalization",
        lambda X, y, params: np.zeros((params.n_splits - 1, 2, 2)),
    )

    with pytest.raises(ValueError, match="Unexpected n_splits"):
        run_group_decoding(
            subjects=["sub-001"],
            epoch_dir=tmp_path,
            dataset_params=_dataset_params(),
            run_params=_run_params(),
            load_subject_epochs_fn=lambda _s, _d: None,
        )

    monkeypatch.setattr(
        "turntaking.analysis.decoding.run_decoding.decode_subject_temporal_generalization",
        lambda X, y, params: np.zeros((params.n_splits, 2, 2)),
    )
    with pytest.raises(RuntimeError, match="No subjects provided"):
        run_group_decoding(
            subjects=[],
            epoch_dir=tmp_path,
            dataset_params=_dataset_params(),
            run_params=_run_params(),
            load_subject_epochs_fn=lambda _s, _d: None,
        )
