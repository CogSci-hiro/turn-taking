
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PathsConfig:
    raw_root: Path
    derived_root: Path

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PathsConfig":
        return PathsConfig(
            raw_root=Path(d["raw_root"]),
            derived_root=Path(d["derived_root"]),
        )


@dataclass(frozen=True)
class EpochingConfig:
    tmin_s: float
    tmax_s: float
    event_name: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EpochingConfig":
        return EpochingConfig(
            tmin_s=float(d["tmin_s"]),
            tmax_s=float(d["tmax_s"]),
            event_name=str(d.get("event_name", "onset")),
        )


@dataclass(frozen=True)
class ProjectConfig:
    paths: PathsConfig
    epoching: EpochingConfig

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProjectConfig":
        return ProjectConfig(
            paths=PathsConfig.from_dict(d["paths"]),
            epoching=EpochingConfig.from_dict(d["epoching"]),
        )
