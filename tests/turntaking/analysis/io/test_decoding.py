from __future__ import annotations

"""Tests for decoding score and feature-cache I/O contracts."""

import numpy as np
import pytest

from turntaking.analysis.io.decoding import (
    DecodingScorePaths,
    Hdf5CacheParams,
    get_decoding_out_dir,
    load_decoding_scores,
    load_subject_feature_cache_hdf5,
    save_decoding_scores,
    save_subject_feature_cache_hdf5,
)


def test_save_and_load_decoding_scores_round_trip(tmp_path):
    """Checks decoding score artifacts are persisted and validated with expected dimensions."""
    scores = np.zeros((2, 3, 4, 4), dtype=float)
    times = np.linspace(-0.2, 0.4, 4)
    scores_path, times_path = save_decoding_scores(
        out_dir=tmp_path, contrast="duration", scores=scores, times_s=times
    )

    paths = DecodingScorePaths(scores_npy=scores_path, times_npy=times_path)
    loaded_scores, loaded_times = load_decoding_scores(paths)

    np.testing.assert_array_equal(loaded_scores, scores)
    np.testing.assert_array_equal(loaded_times, times)


def test_load_decoding_scores_rejects_missing_files(tmp_path):
    """Ensures failures are explicit when expected decoding artifacts are absent."""
    paths = DecodingScorePaths.from_dir(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="scores file"):
        load_decoding_scores(paths)


def test_load_decoding_scores_rejects_invalid_shapes(tmp_path):
    """Prevents invalid data from silently propagating into cluster statistics."""
    d = get_decoding_out_dir(tmp_path, "latency")
    d.mkdir(parents=True)

    np.save(d / "scores.npy", np.zeros((2, 3, 4)))  # not 4D
    np.save(d / "times.npy", np.zeros((4,)))
    with pytest.raises(ValueError, match="Expected scores as 4D array"):
        load_decoding_scores(DecodingScorePaths.from_dir(d))

    np.save(d / "scores.npy", np.zeros((2, 3, 4, 4)))
    np.save(d / "times.npy", np.zeros((2, 2)))  # not 1D
    with pytest.raises(ValueError, match="Expected times as 1D array"):
        load_decoding_scores(DecodingScorePaths.from_dir(d))

    np.save(d / "scores.npy", np.zeros((2, 3, 4, 4)))
    np.save(d / "times.npy", np.zeros((5,)))
    with pytest.raises(ValueError, match="Time axis mismatch"):
        load_decoding_scores(DecodingScorePaths.from_dir(d))


def test_feature_cache_hdf5_round_trip_and_dtype_handling(tmp_path):
    """Verifies per-subject feature cache writes expected dtypes and loads data without information loss."""
    X = np.random.default_rng(0).normal(size=(5, 3, 7)).astype(np.float64)
    y = np.array([0, 1, 0, 1, 1], dtype=np.int64)
    times = np.linspace(-0.1, 0.3, 7)

    save_subject_feature_cache_hdf5(
        out_dir=tmp_path,
        contrast="duration",
        subject="sub-001",
        X=X,
        y=y,
        times_s=times,
        cache_params=Hdf5CacheParams(x_dtype="float32", compression="gzip", compression_level=1),
    )

    X2, y2, times2 = load_subject_feature_cache_hdf5(
        out_dir=tmp_path,
        contrast="duration",
        subject="sub-001",
    )

    assert X2.dtype == np.float32
    assert y2.dtype == np.int8
    assert times2.dtype == np.float64
    np.testing.assert_allclose(X2, X.astype(np.float32))
    np.testing.assert_array_equal(y2, y.astype(np.int8))
    np.testing.assert_array_equal(times2, times)


def test_feature_cache_hdf5_missing_file_raises(tmp_path):
    """Provides a clear error when a requested subject cache has not been generated yet."""
    with pytest.raises(FileNotFoundError, match="Feature cache not found"):
        load_subject_feature_cache_hdf5(
            out_dir=tmp_path,
            contrast="latency",
            subject="sub-404",
        )
