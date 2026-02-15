
"""Tests for generic spatio-temporal cluster permutation wrapper."""

import mne
import numpy as np
import pytest

from turntaking.stats.cluster_test import ClusterTestParams, run_cluster_1samp_spatiotemporal


def _info() -> mne.Info:
    return mne.create_info(ch_names=["Cz", "Pz"], sfreq=32.0, ch_types=["eeg", "eeg"])


def test_run_cluster_1samp_spatiotemporal_validates_shape_and_kind():
    """Ensures wrapper enforces documented input contracts before calling MNE stats."""
    params = ClusterTestParams(n_permutations=10)
    with pytest.raises(ValueError, match="X must be"):
        run_cluster_1samp_spatiotemporal(
            np.zeros((2, 3)),
            info=_info(),
            params=params,
            kind="erp",
            data_tmin_s=-0.2,
            sfreq_hz=32.0,
        )

    with pytest.raises(ValueError, match="kind must be"):
        run_cluster_1samp_spatiotemporal(
            np.zeros((2, 3, 2)),
            info=_info(),
            params=params,
            kind="bad",  # type: ignore[arg-type]
            data_tmin_s=-0.2,
            sfreq_hz=32.0,
        )


def test_run_cluster_1samp_spatiotemporal_passes_through_and_builds_metadata(monkeypatch):
    """Validates adjacency/test call wiring and result metadata completeness."""
    params = ClusterTestParams(
        n_permutations=20,
        threshold=2.0,
        tail=1,
        alpha=0.01,
        seed=42,
        n_jobs=2,
        ch_type="eeg",
    )
    captured = {}

    def fake_adj(info, ch_type):
        captured["adj_call"] = (info["sfreq"], ch_type)
        return np.eye(2)

    def fake_cluster(X, **kwargs):
        captured["cluster_X_shape"] = X.shape
        captured["cluster_kwargs"] = kwargs
        return np.zeros((4, 2)), [(np.array([0]), np.array([1]))], np.array([0.02]), np.array([1.0, 2.0])

    monkeypatch.setattr("turntaking.stats.cluster_test._channel_adjacency", fake_adj)
    monkeypatch.setattr("turntaking.stats.cluster_test.spatio_temporal_cluster_1samp_test", fake_cluster)

    out = run_cluster_1samp_spatiotemporal(
        np.zeros((3, 4, 2)),
        info=_info(),
        params=params,
        kind="erp",
        data_tmin_s=-0.2,
        sfreq_hz=32.0,
    )

    assert captured["adj_call"] == (32.0, "eeg")
    assert captured["cluster_X_shape"] == (3, 4, 2)
    assert captured["cluster_kwargs"]["n_permutations"] == 20
    assert out.metadata["kind"] == "erp"
    assert out.metadata["n_subjects"] == 3
    assert out.metadata["n_times"] == 4
    assert out.metadata["n_channels"] == 2
    assert out.metadata["threshold"] == 2.0
    np.testing.assert_array_equal(out.p_values, np.array([0.02]))
