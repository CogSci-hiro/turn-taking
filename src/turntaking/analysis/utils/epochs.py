
"""Generic epoch path parsing and loading helpers shared across analysis domains."""

import re
from dataclasses import dataclass
from pathlib import Path

import mne

__all__ = [
    "EpochFileInfo",
    "EpochLoadParams",
    "parse_epochs_filepath",
    "load_epochs",
    "load_subject_epochs",
]


@dataclass(frozen=True)
class EpochFileInfo:
    subject: str
    run: int


@dataclass(frozen=True)
class EpochLoadParams:
    suffix: str = "_epochs-epo.fif"
    preload: bool = True


_EPOCHS_RE = re.compile(
    r".*[/\\](?P<subject>sub-\d+)_.*_run-(?P<run>\d+).*(_epochs-epo|_epo)\.fif$"
)


def parse_epochs_filepath(path: Path) -> EpochFileInfo:
    match = _EPOCHS_RE.match(str(path))
    if match is None:
        raise ValueError(f"Could not parse subject/run from epochs path: {path}")
    return EpochFileInfo(subject=match.group("subject"), run=int(match.group("run")))


def load_epochs(path: Path, *, preload: bool = False) -> mne.BaseEpochs:
    return mne.read_epochs(str(path), preload=preload, verbose=False)


def load_subject_epochs(
    subject: str,
    epoch_dir: Path,
    params: EpochLoadParams | None = None,
) -> mne.BaseEpochs:
    params = params or EpochLoadParams()
    epoch_dir = Path(epoch_dir)
    if not epoch_dir.exists():
        raise FileNotFoundError(f"epoch_dir does not exist: {epoch_dir}")
    files = _subject_epoch_files(subject, epoch_dir, params.suffix)
    if not files:
        raise FileNotFoundError(
            f"No epoch files found for subject={subject} in {epoch_dir} matching '*{params.suffix}'."
        )
    epochs_list = [mne.read_epochs(path, preload=params.preload, verbose="ERROR") for path in files]
    if len(epochs_list) == 1:
        return epochs_list[0]
    return mne.concatenate_epochs(epochs_list, verbose="ERROR")


def _subject_epoch_files(subject: str, epoch_dir: Path, suffix: str) -> list[Path]:
    files = [path for path in epoch_dir.rglob(f"{subject}_*{suffix}") if path.is_file()]
    return sorted(files, key=lambda path: str(path))
