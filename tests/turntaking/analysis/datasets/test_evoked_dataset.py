from pathlib import Path

from turntaking.analysis.datasets.evoked_dataset import build_raw_evoked_dataset
from turntaking.analysis.selection import SelectionParams


def _save_epochs(path: Path, sample_epochs) -> None:
    epochs = sample_epochs.copy()
    epochs.metadata = epochs.metadata.copy()
    epochs.save(path, overwrite=True, verbose=False)


def test_build_raw_evoked_dataset_returns_split_epoch_arrays(sample_epochs, tmp_path):
    p1 = tmp_path / "sub-001_task-conversation_run-1_epochs-epo.fif"
    p2 = tmp_path / "sub-002_task-conversation_run-1_epochs-epo.fif"
    _save_epochs(p1, sample_epochs)
    _save_epochs(p2, sample_epochs)

    raw = build_raw_evoked_dataset(
        [p1, p2],
        contrast="duration",
        selection_params=SelectionParams(min_latency=0.0, max_latency=1.0, min_self_duration=0.0),
    )

    assert raw.subject_ids == ["sub-001", "sub-002"]
    assert raw.labels == {"cond_1": "long", "cond_2": "short"}
    assert raw.ch_names == ["Cz", "Pz"]
    assert len(raw.cond1_epochs) == 2
    assert raw.cond1_epochs[0].ndim == 3
    assert raw.cond2_epochs[0].shape == raw.cond1_epochs[0].shape
    assert len(raw.cond1_metadata) == 2
    assert len(raw.infos) == 2
