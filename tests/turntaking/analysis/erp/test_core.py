
"""Unit tests for pure ERP computation helpers."""

import numpy as np
import pandas as pd
import pytest
import mne

from turntaking.analysis.datasets.evoked_dataset import EvokedDatasetRaw
from turntaking.analysis.erp.core import (
    apply_baseline,
    compute_contrast,
    compute_evoked_dataset_result,
    compute_erp_average,
    summarize_erp,
)


def test_compute_erp_average_applies_mask_and_means_trials():
    data = np.array(
        [
            [[1.0, 2.0], [10.0, 20.0]],
            [[3.0, 4.0], [30.0, 40.0]],
            [[5.0, 6.0], [50.0, 60.0]],
        ]
    )
    mask = np.array([True, False, True])
    out = compute_erp_average(data, mask)

    expected = np.array([[3.0, 4.0], [30.0, 40.0]])
    np.testing.assert_allclose(out, expected)


def test_compute_erp_average_validates_shapes_and_nonempty_selection():
    with pytest.raises(ValueError, match="must be 3D"):
        compute_erp_average(np.zeros((2, 3)), np.array([True, False]))

    with pytest.raises(ValueError, match="must match n_trials"):
        compute_erp_average(np.zeros((2, 3, 4)), np.array([True]))

    with pytest.raises(ValueError, match="selects zero trials"):
        compute_erp_average(np.zeros((2, 3, 4)), np.array([False, False]))


def test_compute_contrast_subtracts_condition_2_from_condition_1():
    c1 = np.array([[2.0, 3.0], [4.0, 5.0]])
    c2 = np.array([[1.0, 1.0], [0.5, 1.5]])
    out = compute_contrast(c1, c2)
    np.testing.assert_allclose(out, np.array([[1.0, 2.0], [3.5, 3.5]]))


def test_apply_baseline_subtracts_window_mean_per_channel():
    erp = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    times = np.array([-0.1, 0.0, 0.1])
    out = apply_baseline(erp, times, baseline=(-0.1, 0.0))

    # Baseline means are [1.5, 3.0]
    expected = np.array([[-0.5, 0.5, 1.5], [-1.0, 1.0, 3.0]])
    np.testing.assert_allclose(out, expected)

    unchanged = apply_baseline(erp, times, baseline=None)
    np.testing.assert_allclose(unchanged, erp)
    assert unchanged is not erp


def test_summarize_erp_returns_window_mean_and_peak_metrics():
    erp = np.array([[0.0, 1.0, 0.0], [0.0, 3.0, 2.0]])
    times = np.array([0.0, 0.1, 0.2])
    out = summarize_erp(erp, times, summary_window=(0.0, 0.2))

    assert out["mean_amplitude"] == pytest.approx(1.0)
    assert out["peak_latency"] == pytest.approx(0.1)
    assert out["peak_amplitude"] == pytest.approx(2.0)


def test_compute_evoked_dataset_result_builds_expected_erp_outputs():
    info = mne.create_info(ch_names=["Cz", "Pz"], sfreq=10.0, ch_types=["eeg", "eeg"])
    times = np.array([-0.1, 0.0, 0.1])
    cond1 = np.array(
        [
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
            [[3.0, 4.0, 5.0], [5.0, 6.0, 7.0]],
        ]
    )
    cond2 = np.array(
        [
            [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]],
            [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
        ]
    )
    raw = EvokedDatasetRaw(
        subject_ids=["sub-001"],
        cond1_epochs=[cond1],
        cond2_epochs=[cond2],
        cond1_metadata=[pd.DataFrame({"latency": [0.1, 0.2]})],
        cond2_metadata=[pd.DataFrame({"latency": [0.3, 0.4]})],
        times=times,
        ch_names=["Cz", "Pz"],
        labels={"cond_1": "long", "cond_2": "short"},
        infos=[info],
    )

    result = compute_evoked_dataset_result(raw, contrast="duration")

    assert result.evoked_data.shape == (1, 3, 2, 3)
    np.testing.assert_allclose(result.evoked_data[0, 0], cond1.mean(axis=0))
    np.testing.assert_allclose(result.evoked_data[0, 1], cond2.mean(axis=0))
    np.testing.assert_allclose(result.evoked_data[0, 2], result.evoked_data[0, 0] - result.evoked_data[0, 1])
    assert result.results["kind"] == "erp"
    assert result.results["contrast"] == "duration"
