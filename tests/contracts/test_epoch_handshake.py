"""Contract: raw evoked datasets preserve epoch/metadata handshake invariants."""

from pathlib import Path

import pytest

from turntaking.analysis.datasets.evoked_dataset import build_raw_evoked_dataset
from turntaking.analysis.selection import SelectionParams


@pytest.mark.contract
def test_raw_evoked_dataset_epoch_handshake_invariant() -> None:
    epoch_dir = Path("tests/fixtures/epochs")
    epoch_paths = sorted(epoch_dir.glob("sub-*_task-*_run-*_epo.fif")) + sorted(
        epoch_dir.glob("sub-*_task-*_run-*_epochs-epo.fif")
    )
    assert len(epoch_paths) > 0, "No fixture epochs found for handshake contract test."

    raw = build_raw_evoked_dataset(
        epoch_paths=epoch_paths,
        contrast="duration",
        selection_params=SelectionParams(
            min_latency=-1.0,
            max_latency=1.0,
            min_self_duration=0.01,
        ),
        sfreq=64.0,
    )

    n_subjects = len(raw.subject_ids)
    assert n_subjects == len(raw.cond1_epochs) == len(raw.cond2_epochs)
    assert n_subjects == len(raw.cond1_metadata) == len(raw.cond2_metadata) == len(raw.infos)

    n_channels = len(raw.ch_names)
    n_times = len(raw.times)
    for cond1_arr, cond2_arr, cond1_meta, cond2_meta in zip(
        raw.cond1_epochs,
        raw.cond2_epochs,
        raw.cond1_metadata,
        raw.cond2_metadata,
    ):
        assert cond1_arr.shape[0] == len(cond1_meta)
        assert cond2_arr.shape[0] == len(cond2_meta)
        assert cond1_arr.shape[1] == cond2_arr.shape[1] == n_channels
        assert cond1_arr.shape[2] == cond2_arr.shape[2] == n_times
