from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

@dataclass(frozen=True)
class ProjectSection:
    workdir: Path

@dataclass(frozen=True)
class IoSection:
    epoch_dir: Path
    epoch_pattern: str
    out_dir: Path

@dataclass(frozen=True)
class SubjectsSection:
    mode: Literal["from_epochs"]
    exclude: list[str]
    include: list[str]

@dataclass(frozen=True)
class DatasetSection:
    subjects: SubjectsSection
    tasks: list[str]
    runs: list[int]
    invalid_subject_run: list[list[str]]  # keep as-is for now

@dataclass(frozen=True)
class ConstraintsSection:
    min_latency: float
    max_latency: float
    min_response_duration: float

@dataclass(frozen=True)
class AnalysisErpSection:
    left_margin: float
    right_margin: float
    sfreq: int
    n_permutations: int
    threshold: Any | None

@dataclass(frozen=True)
class AnalysisSection:
    contrasts: list[Literal["duration", "latency"]]
    bands: list[str]
    erp: AnalysisErpSection
    # tfr/mixed/decoding can be added similarly

@dataclass(frozen=True)
class ExecutionSection:
    threads_light: int
    threads_heavy: int
    mem_mb_light: int
    mem_mb_heavy: int

@dataclass(frozen=True)
class TurntakingConfig:
    project: ProjectSection
    io: IoSection
    dataset: DatasetSection
    constraints: ConstraintsSection
    analysis: AnalysisSection
    execution: ExecutionSection

    @staticmethod
    def from_dict(d: dict) -> "TurntakingConfig":
        # implement properly (I can write this for you once I see your current loader style)
        raise NotImplementedError
