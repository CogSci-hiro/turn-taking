
from pathlib import Path

import pytest


def expand_epoch_paths(
    *,
    epoch_dir: Path,
    pattern: str,
    subjects: list[str],
    tasks: list[str],
    runs: list[str],
) -> list[Path]:
    out: list[Path] = []
    for subject in subjects:
        for task in tasks:
            for run in runs:
                fname = pattern.format(subject=subject, task=task, run=run)
                out.append(epoch_dir / fname)
    return out


def test_pattern_expands_expected_paths():
    epoch_dir = Path("data/epochs")
    pattern = "{subject}_{task}_run-{run}_epo.fif"
    paths = expand_epoch_paths(
        epoch_dir=epoch_dir,
        pattern=pattern,
        subjects=["sub-006"],
        tasks=["diapix"],
        runs=["1"],
    )
    assert paths == [Path("data/epochs/sub-006_diapix_run-1_epo.fif")]


def test_invalid_pattern_fails_loudly():
    with pytest.raises(KeyError):
        expand_epoch_paths(
            epoch_dir=Path("data/epochs"),
            pattern="{subject}_{task}_run-{missing}_epo.fif",
            subjects=["sub-006"],
            tasks=["diapix"],
            runs=["1"],
        )
