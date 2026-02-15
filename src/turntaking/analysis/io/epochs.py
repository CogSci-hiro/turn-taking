import re
from dataclasses import dataclass
from pathlib import Path

import mne


@dataclass(frozen=True)
class EpochFileInfo:
    subject: str
    run: int


_EPOCHS_RE = re.compile(
    r".*[/\\](?P<subject>sub-\d+)_.*_run-(?P<run>\d+).*(_epochs-epo|_epo)\.fif$"
)


def parse_epochs_filepath(path: Path) -> EpochFileInfo:
    """Parse subject/run from an epochs filepath."""
    match = _EPOCHS_RE.match(str(path))
    if match is None:
        raise ValueError(f"Could not parse subject/run from epochs path: {path}")
    subject = match.group("subject")
    run = int(match.group("run"))
    return EpochFileInfo(subject=subject, run=run)


def load_epochs(path: Path, *, preload: bool = False) -> mne.BaseEpochs:
    """Load epochs from disk."""
    return mne.read_epochs(str(path), preload=preload, verbose=False)


@dataclass(frozen=True)
class EpochLoadParams:
    """
    Parameters controlling subject epoch discovery.

    Attributes
    ----------
    suffix
        Filename suffix identifying epoch files.
    preload
        Whether to preload epochs (recommended for decoding).
    """
    suffix: str = "_epochs-epo.fif"
    preload: bool = True


def load_subject_epochs(
    subject: str,
    epoch_dir: Path,
    params: EpochLoadParams | None = None,
) -> mne.BaseEpochs:
    """
    Load and concatenate all runs for a subject.

    Expected filename pattern:
        {subject}_..._epochs-epo.fif

    Example:
        sub-004_task-conversation_run-3_epochs-epo.fif
    """
    if params is None:
        params = EpochLoadParams()

    epoch_dir = Path(epoch_dir)
    if not epoch_dir.exists():
        raise FileNotFoundError(f"epoch_dir does not exist: {epoch_dir}")

    # Strict match: subject prefix + required suffix
    files = sorted(
        [
            f
            for f in epoch_dir.rglob(f"{subject}_*{params.suffix}")
            if f.is_file()
        ],
        key=lambda p: str(p),
    )

    if not files:
        raise FileNotFoundError(
            f"No epoch files found for subject={subject} in {epoch_dir} "
            f"matching '*{params.suffix}'."
        )

    epochs_list = [
        mne.read_epochs(f, preload=params.preload, verbose="ERROR")
        for f in files
    ]

    if len(epochs_list) == 1:
        return epochs_list[0]

    return mne.concatenate_epochs(epochs_list, verbose="ERROR")
