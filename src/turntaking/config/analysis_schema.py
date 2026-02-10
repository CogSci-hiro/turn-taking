from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional


def _require_mapping(d: Any, where: str) -> dict[str, Any]:
    if not isinstance(d, dict):
        raise ValueError(f"Expected mapping at {where}, got {type(d).__name__}.")
    return d


def _require_key(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key '{key}' at {where}.")
    return d[key]


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
    mode: Literal["from_epochs", "explicit"]
    exclude: list[str]
    include: list[str]


@dataclass(frozen=True)
class DatasetSection:
    subjects: SubjectsSection
    tasks: list[str]
    runs: list[int]
    invalid_subject_run: list[tuple[str, str]]


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
class AnalysisTfrSection:
    left_margin: float
    right_margin: float
    method: str
    sfreq: int
    n_permutations: int
    threshold: Any | None


@dataclass(frozen=True)
class AnalysisMixedSection:
    tw1: list[float]
    tw2: list[float]
    baseline: list[float]


@dataclass(frozen=True)
class AnalysisDecodingSection:
    sfreq: int
    n_splits: int
    n_permutations: int
    threshold: Any | None
    left_margin: float
    right_margin: float


@dataclass(frozen=True)
class AnalysisSection:
    contrasts: list[Literal["duration", "latency"]]
    bands: list[str]
    erp: AnalysisErpSection
    tfr: AnalysisTfrSection
    mixed: AnalysisMixedSection
    decoding: AnalysisDecodingSection


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
    def from_dict(d: dict[str, Any]) -> "TurntakingConfig":
        d = _require_mapping(d, "root")

        # ----------------------------- project -----------------------------
        project_d = _require_mapping(_require_key(d, "project", "root"), "project")
        project = ProjectSection(
            workdir=Path(_require_key(project_d, "workdir", "project")),
        )

        # -------------------------------- io ------------------------------
        io_d = _require_mapping(_require_key(d, "io", "root"), "io")
        io = IoSection(
            epoch_dir=Path(_require_key(io_d, "epoch_dir", "io")),
            epoch_pattern=str(_require_key(io_d, "epoch_pattern", "io")),
            out_dir=Path(_require_key(io_d, "out_dir", "io")),
        )

        # ----------------------------- dataset ----------------------------
        dataset_d = _require_mapping(_require_key(d, "dataset", "root"), "dataset")

        subjects_d = _require_mapping(_require_key(dataset_d, "subjects", "dataset"), "dataset.subjects")
        mode = str(_require_key(subjects_d, "mode", "dataset.subjects"))
        if mode not in {"from_epochs", "explicit"}:
            raise ValueError("dataset.subjects.mode must be 'from_epochs' or 'explicit'.")

        exclude = list(subjects_d.get("exclude", [])) or []
        include = list(subjects_d.get("include", [])) or []
        subjects = SubjectsSection(
            mode=mode,  # type: ignore[arg-type]
            exclude=[str(x) for x in exclude],
            include=[str(x) for x in include],
        )

        tasks = [str(x) for x in _require_key(dataset_d, "tasks", "dataset")]
        runs_raw = _require_key(dataset_d, "runs", "dataset")
        runs = [int(x) for x in runs_raw]

        invalid_raw = dataset_d.get("invalid_subject_run", []) or []
        invalid_pairs: list[tuple[str, str]] = []
        for pair in invalid_raw:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise ValueError("dataset.invalid_subject_run must be a list of [subject, run] pairs.")
            invalid_pairs.append((str(pair[0]), str(pair[1])))

        dataset = DatasetSection(
            subjects=subjects,
            tasks=tasks,
            runs=runs,
            invalid_subject_run=invalid_pairs,
        )

        # ---------------------------- constraints --------------------------
        constraints_d = _require_mapping(_require_key(d, "constraints", "root"), "constraints")
        constraints = ConstraintsSection(
            min_latency=float(_require_key(constraints_d, "min_latency", "constraints")),
            max_latency=float(_require_key(constraints_d, "max_latency", "constraints")),
            min_response_duration=float(_require_key(constraints_d, "min_response_duration", "constraints")),
        )

        # ------------------------------ analysis ---------------------------
        analysis_d = _require_mapping(_require_key(d, "analysis", "root"), "analysis")

        contrasts_raw = _require_key(analysis_d, "contrasts", "analysis")
        contrasts = [str(x) for x in contrasts_raw]
        for c in contrasts:
            if c not in {"duration", "latency"}:
                raise ValueError(f"analysis.contrasts contains unsupported value: {c!r}")

        bands = [str(x) for x in _require_key(analysis_d, "bands", "analysis")]

        erp_d = _require_mapping(_require_key(analysis_d, "erp", "analysis"), "analysis.erp")
        erp = AnalysisErpSection(
            left_margin=float(_require_key(erp_d, "left_margin", "analysis.erp")),
            right_margin=float(_require_key(erp_d, "right_margin", "analysis.erp")),
            sfreq=int(_require_key(erp_d, "sfreq", "analysis.erp")),
            n_permutations=int(_require_key(erp_d, "n_permutations", "analysis.erp")),
            threshold=erp_d.get("threshold"),
        )

        tfr_d = _require_mapping(_require_key(analysis_d, "tfr", "analysis"), "analysis.tfr")
        tfr = AnalysisTfrSection(
            left_margin=float(_require_key(tfr_d, "left_margin", "analysis.tfr")),
            right_margin=float(_require_key(tfr_d, "right_margin", "analysis.tfr")),
            method=str(_require_key(tfr_d, "method", "analysis.tfr")),
            sfreq=int(_require_key(tfr_d, "sfreq", "analysis.tfr")),
            n_permutations=int(_require_key(tfr_d, "n_permutations", "analysis.tfr")),
            threshold=tfr_d.get("threshold"),
        )

        mixed_d = _require_mapping(_require_key(analysis_d, "mixed", "analysis"), "analysis.mixed")
        mixed = AnalysisMixedSection(
            tw1=[float(x) for x in _require_key(mixed_d, "tw1", "analysis.mixed")],
            tw2=[float(x) for x in _require_key(mixed_d, "tw2", "analysis.mixed")],
            baseline=[float(x) for x in _require_key(mixed_d, "baseline", "analysis.mixed")],
        )

        decoding_d = _require_mapping(_require_key(analysis_d, "decoding", "analysis"), "analysis.decoding")
        decoding = AnalysisDecodingSection(
            sfreq=int(_require_key(decoding_d, "sfreq", "analysis.decoding")),
            n_splits=int(_require_key(decoding_d, "n_splits", "analysis.decoding")),
            n_permutations=int(_require_key(decoding_d, "n_permutations", "analysis.decoding")),
            threshold=decoding_d.get("threshold"),
            left_margin=float(_require_key(decoding_d, "left_margin", "analysis.decoding")),
            right_margin=float(_require_key(decoding_d, "right_margin", "analysis.decoding")),
        )

        analysis = AnalysisSection(
            contrasts=contrasts,  # type: ignore[arg-type]
            bands=bands,
            erp=erp,
            tfr=tfr,
            mixed=mixed,
            decoding=decoding,
        )

        # ----------------------------- execution ---------------------------
        execution_d = _require_mapping(_require_key(d, "execution", "root"), "execution")
        execution = ExecutionSection(
            threads_light=int(_require_key(execution_d, "threads_light", "execution")),
            threads_heavy=int(_require_key(execution_d, "threads_heavy", "execution")),
            mem_mb_light=int(_require_key(execution_d, "mem_mb_light", "execution")),
            mem_mb_heavy=int(_require_key(execution_d, "mem_mb_heavy", "execution")),
        )

        return TurntakingConfig(
            project=project,
            io=io,
            dataset=dataset,
            constraints=constraints,
            analysis=analysis,
            execution=execution,
        )
