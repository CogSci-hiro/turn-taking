import re
from dataclasses import dataclass
from pathlib import Path

import mne


@dataclass(frozen=True)
class EpochFileInfo:
    subject: str
    run: int


_EPOCHS_RE = re.compile(
    r".*[/\\](?P<subject>sub-\d+)_.*_run-(?P<run>\d+)_epochs-epo\.fif$"
)


def parse_epochs_filepath(path: Path) -> EpochFileInfo:
    """Parse subject/run from an epochs filepath."""
    match = _EPOCHS_RE.match(str(path))
    if match is None:
        raise ValueError(f"Could not parse subject/run from epochs path: {path}")
    subject = match.group("subject").replace("sub-", "")
    run = int(match.group("run"))
    return EpochFileInfo(subject=subject, run=run)


def load_epochs(path: Path, *, preload: bool = False) -> mne.BaseEpochs:
    """Load epochs from disk."""
    return mne.read_epochs(str(path), preload=preload, verbose=False)
