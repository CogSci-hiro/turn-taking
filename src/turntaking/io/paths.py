
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from turntaking.config.schema import ProjectConfig


@dataclass(frozen=True)
class EpochsPath:
    path: Path


def epochs_fif_path(
    cfg: ProjectConfig,
    subject: str,
    run: str,
) -> EpochsPath:
    out_dir = cfg.paths.derived_root / "epochs"
    fname = f"{subject}_task-conversation_{run}_epochs-epo.fif"
    return EpochsPath(path=out_dir / fname)
