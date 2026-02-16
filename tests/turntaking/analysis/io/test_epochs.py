
"""Tests for epoch-path parsing and subject-level epoch loading."""

from pathlib import Path

import pytest

from turntaking.analysis.utils.epochs import EpochLoadParams, parse_epochs_filepath, load_epochs, load_subject_epochs


def test_parse_epochs_filepath_extracts_subject_and_run():
    """Ensures filename parsing extracts subject/run used by downstream grouping logic."""
    info = parse_epochs_filepath(Path("/x/sub-004_task-conversation_run-3_epochs-epo.fif"))
    assert info.subject == "sub-004"
    assert info.run == 3


def test_parse_epochs_filepath_rejects_unexpected_pattern():
    """Prevents silent mis-parsing when input filenames drift from required convention."""
    with pytest.raises(ValueError, match="Could not parse subject/run"):
        parse_epochs_filepath(Path("/x/not_an_epoch_file.fif"))


def test_load_epochs_delegates_to_mne_read_epochs(monkeypatch, tmp_path):
    """Checks the thin loader forwards path and preload parameters to MNE unchanged."""
    called: dict[str, object] = {}

    def fake_read_epochs(path, preload, verbose):
        called["path"] = path
        called["preload"] = preload
        called["verbose"] = verbose
        return "EPOCHS"

    monkeypatch.setattr("turntaking.analysis.utils.epochs.mne.read_epochs", fake_read_epochs)

    out = load_epochs(tmp_path / "sub-001_task-conversation_run-1_epochs-epo.fif", preload=True)
    assert out == "EPOCHS"
    assert called["preload"] is True
    assert called["verbose"] is False


def test_load_subject_epochs_rejects_missing_dir(tmp_path):
    """Ensures callers receive explicit missing-directory errors rather than empty results."""
    with pytest.raises(FileNotFoundError, match="epoch_dir does not exist"):
        load_subject_epochs("sub-001", tmp_path / "missing")


def test_load_subject_epochs_rejects_when_no_subject_files(tmp_path):
    """Guards against accidental subject mismatches in file discovery patterns."""
    epoch_dir = tmp_path / "epochs"
    epoch_dir.mkdir()
    (epoch_dir / "sub-999_task-conversation_run-1_epochs-epo.fif").write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="No epoch files found for subject=sub-001"):
        load_subject_epochs("sub-001", epoch_dir)


def test_load_subject_epochs_returns_single_file_without_concat(monkeypatch, tmp_path):
    """Verifies single-run subjects bypass concatenation and return raw loaded epochs."""
    epoch_dir = tmp_path / "epochs"
    epoch_dir.mkdir()
    f = epoch_dir / "sub-001_task-conversation_run-1_epochs-epo.fif"
    f.write_text("x", encoding="utf-8")

    def fake_read_epochs(path, preload, verbose):
        return {"path": str(path), "preload": preload, "verbose": verbose}

    monkeypatch.setattr("turntaking.analysis.utils.epochs.mne.read_epochs", fake_read_epochs)
    out = load_subject_epochs("sub-001", epoch_dir, EpochLoadParams(preload=False))
    assert out["path"] == str(f)
    assert out["preload"] is False


def test_load_subject_epochs_concatenates_multiple_runs(monkeypatch, tmp_path):
    """Checks deterministic ordering and concatenation for multi-run subjects."""
    epoch_dir = tmp_path / "epochs"
    epoch_dir.mkdir()
    f2 = epoch_dir / "sub-001_task-conversation_run-2_epochs-epo.fif"
    f1 = epoch_dir / "sub-001_task-conversation_run-1_epochs-epo.fif"
    f1.write_text("x", encoding="utf-8")
    f2.write_text("x", encoding="utf-8")

    loaded = []

    def fake_read_epochs(path, preload, verbose):
        loaded.append(Path(path).name)
        return Path(path).name

    def fake_concat(lst, verbose):
        return {"ordered": lst, "verbose": verbose}

    monkeypatch.setattr("turntaking.analysis.utils.epochs.mne.read_epochs", fake_read_epochs)
    monkeypatch.setattr("turntaking.analysis.utils.epochs.mne.concatenate_epochs", fake_concat)

    out = load_subject_epochs("sub-001", epoch_dir)
    assert loaded == [f1.name, f2.name]
    assert out["ordered"] == [f1.name, f2.name]
    assert out["verbose"] == "ERROR"
