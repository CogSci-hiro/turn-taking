import mne
import numpy as np
import pandas as pd

from turntaking.analysis.datasets.evoked_dataset import EvokedDatasetRaw
from turntaking.analysis.tfr.core import compute_induced_dataset_result


def test_compute_induced_dataset_result_returns_expected_contract():
    rng = np.random.default_rng(0)
    info = mne.create_info(ch_names=["Cz", "Pz"], sfreq=64.0, ch_types=["eeg", "eeg"])
    times = np.linspace(-0.5, 0.5, 64, endpoint=False)
    cond1 = rng.normal(size=(4, 2, 64))
    cond2 = rng.normal(size=(4, 2, 64))
    raw = EvokedDatasetRaw(
        subject_ids=["sub-001"],
        cond1_epochs=[cond1],
        cond2_epochs=[cond2],
        cond1_metadata=[pd.DataFrame({"latency": [0.1, 0.2, 0.3, 0.4]})],
        cond2_metadata=[pd.DataFrame({"latency": [0.5, 0.6, 0.7, 0.8]})],
        times=times,
        ch_names=["Cz", "Pz"],
        labels={"cond_1": "fast", "cond_2": "slow"},
        infos=[info],
    )

    result = compute_induced_dataset_result(raw, band="alpha", contrast="latency")

    assert result.evoked_data.shape == (1, 3, 2, 64)
    np.testing.assert_allclose(result.evoked_data[0, 2], result.evoked_data[0, 0] - result.evoked_data[0, 1])
    assert result.results["kind"] == "tfr"
    assert result.results["band"] == "alpha"
    assert result.results["contrast"] == "latency"
