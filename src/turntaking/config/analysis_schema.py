from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple


def _require_mapping(d: Any, where: str) -> dict[str, Any]:
    if not isinstance(d, dict):
        raise ValueError(f"Expected mapping at {where}, got {type(d).__name__}.")
    return d


def _require_key(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key '{key}' at {where}.")
    return d[key]


@dataclass(frozen=True)
class VizErpHistSection:
    duration_long_fif: Path
    duration_short_fif: Path
    latency_fast_fif: Path
    latency_slow_fif: Path

    hist_table_csv: Path  # NEW, required

    out_base: Path
    xlim_ms: Tuple[float, float]
    ylim_uv: Tuple[float, float]

    @classmethod
    def from_dict(cls, raw: dict) -> "VizErpHistSection":
        return cls(
            duration_long_fif=Path(raw["duration_long_fif"]),
            duration_short_fif=Path(raw["duration_short_fif"]),
            latency_fast_fif=Path(raw["latency_fast_fif"]),
            latency_slow_fif=Path(raw["latency_slow_fif"]),
            hist_table_csv=Path(raw["hist_table_csv"]),  # NEW
            out_base=Path(raw["out_base"]),
            xlim_ms=(float(raw["xlim_ms"][0]), float(raw["xlim_ms"][1])),
            ylim_uv=(float(raw["ylim_uv"][0]), float(raw["ylim_uv"][1])),
        )



@dataclass(frozen=True)
class VizErpTopomapsSection:
    template_svg: Path

    # Optional: Snakemake/CLI can override these
    out_svg: Optional[Path] = None
    parts_dir: Optional[Path] = None

    info_source_fif: Path = Path(".")
    duration_cluster_hdf5: Path = Path(".")
    latency_cluster_hdf5: Path = Path(".")

    p_threshold: float = 0.05
    n_duration_maps: int = 2
    n_latency_maps: int = 3


@dataclass(frozen=True)
class VizTfrToposSection:
    alpha_duration_cluster_hdf5: Path
    alpha_latency_cluster_hdf5: Path
    beta_duration_cluster_hdf5: Path
    beta_latency_cluster_hdf5: Path
    info_source_fif: Path

    out_alpha_duration: Path
    out_alpha_latency: Path
    out_beta_duration: Path
    out_beta_latency: Path

    tmin_s: float
    tmax_s: float
    step_ms: float
    max_cols: int = 10
    p_threshold: float = 0.01

    @classmethod
    def from_dict(cls, raw: dict) -> "VizTfrToposSection":
        return cls(
            alpha_duration_cluster_hdf5=Path(raw["alpha_duration_cluster_hdf5"]),
            alpha_latency_cluster_hdf5=Path(raw["alpha_latency_cluster_hdf5"]),
            beta_duration_cluster_hdf5=Path(raw["beta_duration_cluster_hdf5"]),
            beta_latency_cluster_hdf5=Path(raw["beta_latency_cluster_hdf5"]),
            info_source_fif=Path(raw["info_source_fif"]),
            out_alpha_duration=Path(raw["out_alpha_duration"]),
            out_alpha_latency=Path(raw["out_alpha_latency"]),
            out_beta_duration=Path(raw["out_beta_duration"]),
            out_beta_latency=Path(raw["out_beta_latency"]),
            tmin_s=float(raw["tmin_s"]),
            tmax_s=float(raw["tmax_s"]),
            step_ms=float(raw["step_ms"]),
            max_cols=int(raw.get("max_cols", 10)),
            p_threshold=float(raw.get("p_threshold", 0.01)),
        )


@dataclass(frozen=True)
class VizTfrTopomapsSection:
    """
    Config for the composed TFR topomap figure assembled from SVG parts.

    Usage example
    -------------
        viz:
          tfr_topomaps:
            template_svg: "workflow/templates/EF-timeline.svg"
            info_source_fif: "/path/to/tfr/.../difference_ave.fif"
            alpha_cluster_hdf5: "/path/to/.../alpha/cluster_results.hdf5"
            beta_cluster_hdf5: "/path/to/.../beta/cluster_results.hdf5"
            p_threshold: 0.05
            n_duration_maps: 2
            n_latency_maps: 3
    """
    template_svg: Path

    # Optional outputs if you want config fallback (Snakemake/CLI should override)
    out_svg: Optional[Path] = None
    parts_dir: Optional[Path] = None

    info_source_fif: Path = Path(".")
    alpha_cluster_hdf5: Path = Path(".")
    beta_cluster_hdf5: Path = Path(".")

    p_threshold: float = 0.05
    n_duration_maps: int = 2
    n_latency_maps: int = 3


@dataclass(frozen=True)
class VizErpTimecourseSection:
    duration_long_fif: Path
    duration_short_fif: Path
    latency_fast_fif: Path
    latency_slow_fif: Path
    out_base: Path
    xlim_ms: Tuple[float, float]
    ylim_uv: Tuple[float, float]

    @classmethod
    def from_dict(cls, raw: dict) -> "VizErpTimecourseSection":
        return cls(
            duration_long_fif=Path(raw["duration_long_fif"]),
            duration_short_fif=Path(raw["duration_short_fif"]),
            latency_fast_fif=Path(raw["latency_fast_fif"]),
            latency_slow_fif=Path(raw["latency_slow_fif"]),
            out_base=Path(raw["out_base"]),
            xlim_ms=tuple(float(x) for x in raw["xlim_ms"]),
            ylim_uv=tuple(float(y) for y in raw["ylim_uv"]),
        )


@dataclass(frozen=True)
class VizErpTopoSection:
    duration_cluster_hdf5: Path
    latency_cluster_hdf5: Path
    info_source_fif: Path

    out_duration: Path
    out_latency: Path

    tmin_s: float
    tmax_s: float
    step_ms: float
    max_cols: int = 10
    p_threshold: float = 0.01

    @classmethod
    def from_dict(cls, raw: dict) -> "VizErpTopoSection":
        return cls(
            duration_cluster_hdf5=Path(raw["duration_cluster_hdf5"]),
            latency_cluster_hdf5=Path(raw["latency_cluster_hdf5"]),
            info_source_fif=Path(raw["info_source_fif"]),
            out_duration=Path(raw["out_duration"]),
            out_latency=Path(raw["out_latency"]),
            tmin_s=float(raw["tmin_s"]),
            tmax_s=float(raw["tmax_s"]),
            step_ms=float(raw["step_ms"]),
            max_cols=int(raw.get("max_cols", 10)),
            p_threshold=float(raw.get("p_threshold", 0.01)),
        )

@dataclass(frozen=True)
class VizBehaviorSection:
    duration_offsets_csv: str
    latency_offsets_csv: str
    turn_table_csv: str
    out_base: str
    n_bins: int = 100

@dataclass(frozen=True)
class VizSection:
    erp_timecourse: VizErpTimecourseSection
    erp_topo: VizErpTopoSection
    tfr_topos: VizTfrToposSection
    behavior: VizBehaviorSection
    erp_topomaps: VizErpTopomapsSection
    tfr_topomaps: VizTfrTopomapsSection
    erp_hist: VizErpHistSection

    @classmethod
    def from_dict(cls, raw: dict) -> "VizSection":
        return cls(
            erp_timecourse=VizErpTimecourseSection.from_dict(raw["erp_timecourse"])
        )



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
class MixedSelectionSection:
    min_latency: float
    max_latency: float
    min_self_duration: float


@dataclass(frozen=True)
class AnalysisMixedSection:
    tw1: list[float]
    tw2: list[float]
    baseline: list[float]
    selection: MixedSelectionSection


@dataclass(frozen=True)
class AnalysisDecodingSection:
    sfreq: int
    n_splits: int

    # used in stats stage (but harmless to exist now)
    n_permutations: int = 0
    threshold: Any | None = None
    left_margin: float = 0.0
    right_margin: float = 0.0


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
    io: IoSection
    dataset: DatasetSection
    constraints: ConstraintsSection
    analysis: AnalysisSection
    execution: ExecutionSection
    viz: VizSection

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TurntakingConfig":
        d = _require_mapping(d, "root")

        # -------------------------------- io ------------------------------
        if "io" in d:
            io_d = _require_mapping(_require_key(d, "io", "root"), "io")
        elif "paths" in d:
            io_d = _require_mapping(_require_key(d, "paths", "root"), "paths")
        else:
            raise KeyError("Missing required key 'io' (or fallback 'paths') in root.")
        io = IoSection(
            epoch_dir=Path(_require_key(io_d, "epoch_dir", "io/paths")),
            epoch_pattern=str(_require_key(io_d, "epoch_pattern", "io/paths")),
            out_dir=Path(_require_key(io_d, "out_dir", "io/paths")),
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
            min_response_duration=float(constraints_d.get("min_response_duration", 0.0)),
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
            left_margin=float(erp_d.get("left_margin", 0.0)),
            right_margin=float(erp_d.get("right_margin", 0.0)),
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

        # ------------------------------ mixed ------------------------------
        mixed_d = _require_mapping(_require_key(analysis_d, "mixed", "analysis"), "analysis.mixed")

        sel_d = _require_mapping(_require_key(mixed_d, "selection", "analysis.mixed"), "analysis.mixed.selection")
        selection = MixedSelectionSection(
            min_latency=float(_require_key(sel_d, "min_latency", "analysis.mixed.selection")),
            max_latency=float(_require_key(sel_d, "max_latency", "analysis.mixed.selection")),
            min_self_duration=float(_require_key(sel_d, "min_self_duration", "analysis.mixed.selection")),
        )

        mixed = AnalysisMixedSection(
            tw1=[float(x) for x in _require_key(mixed_d, "tw1", "analysis.mixed")],
            tw2=[float(x) for x in _require_key(mixed_d, "tw2", "analysis.mixed")],
            baseline=[float(x) for x in _require_key(mixed_d, "baseline", "analysis.mixed")],
            selection=selection,
        )

        # ----------------------------- decoding ----------------------------
        decoding_d = _require_mapping(_require_key(analysis_d, "decoding", "analysis"), "analysis.decoding")
        decoding = AnalysisDecodingSection(
            sfreq=int(_require_key(decoding_d, "sfreq", "analysis.decoding")),
            n_splits=int(_require_key(decoding_d, "n_splits", "analysis.decoding")),
            n_permutations=int(decoding_d.get("n_permutations", 0)),
            threshold=decoding_d.get("threshold", None),
            left_margin=float(decoding_d.get("left_margin", 0.0)),
            right_margin=float(decoding_d.get("right_margin", 0.0)),
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

        # ------------------------------- viz -------------------------------
        viz_d = _require_mapping(_require_key(d, "viz", "root"), "viz")
        erp_tc_d = _require_mapping(_require_key(viz_d, "erp_timecourse", "viz"), "viz.erp_timecourse")

        erp_timecourse = VizErpTimecourseSection(
            duration_long_fif=Path(_require_key(erp_tc_d, "duration_long_fif", "viz.erp_timecourse")),
            duration_short_fif=Path(_require_key(erp_tc_d, "duration_short_fif", "viz.erp_timecourse")),
            latency_fast_fif=Path(_require_key(erp_tc_d, "latency_fast_fif", "viz.erp_timecourse")),
            latency_slow_fif=Path(_require_key(erp_tc_d, "latency_slow_fif", "viz.erp_timecourse")),
            out_base=Path(_require_key(erp_tc_d, "out_base", "viz.erp_timecourse")),
            xlim_ms=[float(x) for x in _require_key(erp_tc_d, "xlim_ms", "viz.erp_timecourse")],
            ylim_uv=[float(x) for x in _require_key(erp_tc_d, "ylim_uv", "viz.erp_timecourse")],
        )

        behavior_d = _require_mapping(_require_key(viz_d, "behavior", "viz"), "viz.behavior")
        behavior = VizBehaviorSection(
            duration_offsets_csv=Path(_require_key(behavior_d, "duration_offsets_csv", "viz.behavior")),
            latency_offsets_csv=Path(_require_key(behavior_d, "latency_offsets_csv", "viz.behavior")),
            turn_table_csv=Path(_require_key(behavior_d, "turn_table_csv", "viz.behavior")),
            out_base=Path(_require_key(behavior_d, "out_base", "viz.behavior")),
            n_bins=int(_require_key(behavior_d, "n_bins", "viz.behavior")),
        )

        erp_topo_d = _require_mapping(
            _require_key(viz_d, "erp_topo", "viz"),
            "viz.erp_topo",
        )

        erp_topomaps_d = _require_mapping(
            _require_key(viz_d, "erp_topomaps", "viz"),
            "viz.erp_topomaps",
        )

        erp_topomaps = VizErpTopomapsSection(
            template_svg=Path(_require_key(erp_topomaps_d, "template_svg", "viz.erp_topomaps")),
            out_svg=Path(erp_topomaps_d["out_svg"]) if "out_svg" in erp_topomaps_d else None,
            parts_dir=Path(erp_topomaps_d["parts_dir"]) if "parts_dir" in erp_topomaps_d else None,
            info_source_fif=Path(_require_key(erp_topomaps_d, "info_source_fif", "viz.erp_topomaps")),
            duration_cluster_hdf5=Path(_require_key(erp_topomaps_d, "duration_cluster_hdf5", "viz.erp_topomaps")),
            latency_cluster_hdf5=Path(_require_key(erp_topomaps_d, "latency_cluster_hdf5", "viz.erp_topomaps")),
            p_threshold=float(erp_topomaps_d.get("p_threshold", 0.05)),
            n_duration_maps=int(erp_topomaps_d.get("n_duration_maps", 2)),
            n_latency_maps=int(erp_topomaps_d.get("n_latency_maps", 3)),
        )

        erp_topo = VizErpTopoSection(
            duration_cluster_hdf5=Path(_require_key(erp_topo_d, "duration_cluster_hdf5", "viz.erp_topo")),
            latency_cluster_hdf5=Path(_require_key(erp_topo_d, "latency_cluster_hdf5", "viz.erp_topo")),
            info_source_fif=Path(_require_key(erp_topo_d, "info_source_fif", "viz.erp_topo")),
            out_duration=Path(_require_key(erp_topo_d, "out_duration", "viz.erp_topo")),
            out_latency=Path(_require_key(erp_topo_d, "out_latency", "viz.erp_topo")),
            tmin_s=float(_require_key(erp_topo_d, "tmin_s", "viz.erp_topo")),
            tmax_s=float(_require_key(erp_topo_d, "tmax_s", "viz.erp_topo")),
            step_ms=float(_require_key(erp_topo_d, "step_ms", "viz.erp_topo")),
            max_cols=int(erp_topo_d.get("max_cols", 10)),
            p_threshold=float(erp_topo_d.get("p_threshold", 0.01)),
        )

        tfr_topomaps_d = _require_mapping(
            _require_key(viz_d, "tfr_topomaps", "viz"),
            "viz.tfr_topomaps",
        )

        tfr_topomaps = VizTfrTopomapsSection(
            template_svg=Path(_require_key(tfr_topomaps_d, "template_svg", "viz.tfr_topomaps")),
            out_svg=Path(tfr_topomaps_d["out_svg"]) if "out_svg" in tfr_topomaps_d else None,
            parts_dir=Path(tfr_topomaps_d["parts_dir"]) if "parts_dir" in tfr_topomaps_d else None,
            info_source_fif=Path(_require_key(tfr_topomaps_d, "info_source_fif", "viz.tfr_topomaps")),
            alpha_cluster_hdf5=Path(_require_key(tfr_topomaps_d, "alpha_cluster_hdf5", "viz.tfr_topomaps")),
            beta_cluster_hdf5=Path(_require_key(tfr_topomaps_d, "beta_cluster_hdf5", "viz.tfr_topomaps")),
            p_threshold=float(tfr_topomaps_d.get("p_threshold", 0.05)),
            n_duration_maps=int(tfr_topomaps_d.get("n_duration_maps", 2)),
            n_latency_maps=int(tfr_topomaps_d.get("n_latency_maps", 3)),
        )

        tfr_topos_d = _require_mapping(
            _require_key(viz_d, "tfr_topos", "viz"),
            "viz.tfr_topos",
        )

        tfr_topos = VizTfrToposSection(
            alpha_duration_cluster_hdf5=Path(_require_key(tfr_topos_d, "alpha_duration_cluster_hdf5", "viz.tfr_topos")),
            alpha_latency_cluster_hdf5=Path(_require_key(tfr_topos_d, "alpha_latency_cluster_hdf5", "viz.tfr_topos")),
            beta_duration_cluster_hdf5=Path(_require_key(tfr_topos_d, "beta_duration_cluster_hdf5", "viz.tfr_topos")),
            beta_latency_cluster_hdf5=Path(_require_key(tfr_topos_d, "beta_latency_cluster_hdf5", "viz.tfr_topos")),
            info_source_fif=Path(_require_key(tfr_topos_d, "info_source_fif", "viz.tfr_topos")),
            out_alpha_duration=Path(_require_key(tfr_topos_d, "out_alpha_duration", "viz.tfr_topos")),
            out_alpha_latency=Path(_require_key(tfr_topos_d, "out_alpha_latency", "viz.tfr_topos")),
            out_beta_duration=Path(_require_key(tfr_topos_d, "out_beta_duration", "viz.tfr_topos")),
            out_beta_latency=Path(_require_key(tfr_topos_d, "out_beta_latency", "viz.tfr_topos")),
            tmin_s=float(_require_key(tfr_topos_d, "tmin_s", "viz.tfr_topos")),
            tmax_s=float(_require_key(tfr_topos_d, "tmax_s", "viz.tfr_topos")),
            step_ms=float(_require_key(tfr_topos_d, "step_ms", "viz.tfr_topos")),
            max_cols=int(tfr_topos_d.get("max_cols", 10)),
            p_threshold=float(tfr_topos_d.get("p_threshold", 0.01)),
        )

        erp_hist_d = _require_mapping(_require_key(viz_d, "erp_hist", "viz"), "viz.erp_hist")
        erp_hist = VizErpHistSection.from_dict(erp_hist_d)

        viz = VizSection(
            erp_timecourse=erp_timecourse,
            erp_topo=erp_topo,
            behavior=behavior,
            erp_topomaps=erp_topomaps,
            tfr_topomaps=tfr_topomaps,
            tfr_topos=tfr_topos,
            erp_hist=erp_hist
        )

        return TurntakingConfig(
            io=io,
            dataset=dataset,
            constraints=constraints,
            analysis=analysis,
            execution=execution,
            viz=viz,
        )
