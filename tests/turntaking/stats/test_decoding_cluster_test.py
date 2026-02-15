from __future__ import annotations

"""Tests for decoding cluster-test orchestration and summary formatting."""

import numpy as np
import pandas as pd
import pytest

from turntaking.stats.decoding_cluster_test import (
    DecodingClusterTestParams,
    make_decoding_cluster_summary,
    run_decoding_cluster_test,
)


def _params() -> DecodingClusterTestParams:
    return DecodingClusterTestParams(threshold=None, n_permutations=10, tail=1, n_jobs=1, chance_level=0.5)


def test_run_decoding_cluster_test_validates_input_shape():
    """Rejects malformed decoding score tensors before costly permutation testing."""
    with pytest.raises(ValueError, match="Expected scores with ndim 3 or 4"):
        run_decoding_cluster_test(scores=np.zeros((2, 2)), params=_params())

    with pytest.raises(ValueError, match="Need >=2 subjects"):
        run_decoding_cluster_test(scores=np.zeros((1, 3, 3)), params=_params())

    with pytest.raises(ValueError, match="Expected square temporal generalization matrices"):
        run_decoding_cluster_test(scores=np.zeros((2, 3, 4)), params=_params())


def test_run_decoding_cluster_test_averages_splits_and_calls_mne(monkeypatch):
    """Checks 4D input is averaged across splits before passing into cluster routine."""
    captured = {}

    def fake_cluster_test(X, **kwargs):
        captured["X"] = X
        captured["kwargs"] = kwargs
        return np.zeros((4, 4)), [(np.array([0]), np.array([1]))], np.array([0.04]), np.array([0.1, 0.2])

    monkeypatch.setattr(
        "turntaking.stats.decoding_cluster_test.spatio_temporal_cluster_1samp_test",
        fake_cluster_test,
    )

    scores = np.ones((3, 2, 4, 4))
    t, clusters, p, h0 = run_decoding_cluster_test(scores=scores, params=_params())
    assert captured["X"].shape == (3, 4, 4)
    assert captured["kwargs"]["out_type"] == "indices"
    assert isinstance(clusters, list)
    np.testing.assert_array_equal(t, np.zeros((4, 4)))
    np.testing.assert_array_equal(p, np.array([0.04]))
    np.testing.assert_array_equal(h0, np.array([0.1, 0.2]))


def test_make_decoding_cluster_summary_formats_ranges_and_counts():
    """Verifies summary table encodes cluster extents and p-values for reporting."""
    clusters = [
        (np.array([0, 1, 2]), np.array([1, 2, 3])),
        (np.array([], dtype=int), np.array([], dtype=int)),
    ]
    pvals = np.array([0.03, 0.4])
    times = np.array([-0.1, 0.0, 0.1, 0.2])
    df = make_decoding_cluster_summary(clusters=clusters, p_values=pvals, times_s=times)

    assert isinstance(df, pd.DataFrame)
    assert list(df["cluster_id"]) == [0, 1]
    assert df.loc[0, "n_points"] == 3
    assert df.loc[0, "train_tmin_s"] == -0.1
    assert df.loc[0, "test_tmax_s"] == 0.2
    assert df.loc[1, "n_points"] == 0


def test_make_decoding_cluster_summary_requires_1d_times():
    """Avoids ambiguous indexing by enforcing 1D time vector input."""
    with pytest.raises(ValueError, match="Expected times_s as"):
        make_decoding_cluster_summary(
            clusters=[],
            p_values=np.array([]),
            times_s=np.zeros((2, 2)),
        )
