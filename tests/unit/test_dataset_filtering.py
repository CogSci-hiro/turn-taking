
from pathlib import Path

from turntaking.cli.commands.erp import _expand_epoch_paths_from_config


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _cfg(epoch_dir: Path):
    return Obj(
        io=Obj(
            epoch_dir=str(epoch_dir),
            epoch_pattern="{subject}_task-{task}_run-{run}_epochs-epo.fif",
            out_dir=str(epoch_dir / "out"),
        ),
        dataset=Obj(
            subjects=Obj(mode="explicit", include=["sub-006", "sub-007"], exclude=["sub-007"]),
            tasks=["conversation"],
            runs=[1, 2],
            invalid_subject_run=[["sub-006", "2"]],
        ),
    )


def test_excluded_and_invalid_pairs_removed_deterministically(tmp_path):
    epoch_dir = tmp_path / "epochs"
    epoch_dir.mkdir()
    (epoch_dir / "sub-006_task-conversation_run-1_epochs-epo.fif").write_text("x", encoding="utf-8")
    (epoch_dir / "sub-006_task-conversation_run-2_epochs-epo.fif").write_text("x", encoding="utf-8")
    (epoch_dir / "sub-007_task-conversation_run-1_epochs-epo.fif").write_text("x", encoding="utf-8")

    cfg = _cfg(epoch_dir)
    paths = _expand_epoch_paths_from_config(cfg)
    assert paths == [epoch_dir / "sub-006_task-conversation_run-1_epochs-epo.fif"]


def test_missing_runs_are_ignored_not_crashing(tmp_path):
    epoch_dir = tmp_path / "epochs"
    epoch_dir.mkdir()
    (epoch_dir / "sub-006_task-conversation_run-1_epochs-epo.fif").write_text("x", encoding="utf-8")

    cfg = _cfg(epoch_dir)
    cfg.dataset.invalid_subject_run = []
    paths = _expand_epoch_paths_from_config(cfg)
    assert paths == [epoch_dir / "sub-006_task-conversation_run-1_epochs-epo.fif"]
