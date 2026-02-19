"""
Typed configuration schema for the turn-taking workflow.

This module turns a user-provided YAML mapping (see ``workflow/config.yaml``)
into a nested set of frozen dataclasses with basic validation and a few
backward-compatible defaults.

Why dataclasses?
---------------
- Call-sites get attribute access with type hints instead of raw dicts.
- Validation happens once at load time, not sprinkled across the codebase.
- Sphinx API docs can render the schema in a discoverable way.

The canonical loader is ``turntaking.config.loader.load_config``.
"""

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


def _resolve_viz_path(base_out_dir: Path, value: Any) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_out_dir / path


def _optional_mapping(d: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    raw = d.get(key, {})
    if raw is None:
        raw = {}
    return _require_mapping(raw, f"{where}.{key}")


def _resolve_viz_default_path(
    section: dict[str, Any],
    key: str,
    *,
    base_out_dir: Path,
    default: str | Path,
) -> Path:
    del section, key
    return _resolve_viz_path(base_out_dir, default)


def _float_pair(
    section: dict[str, Any],
    key: str,
    *,
    default: tuple[float, float],
    where: str,
) -> tuple[float, float]:
    raw = section.get(key, default)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{where}.{key} must be a two-value list/tuple.")
    return float(raw[0]), float(raw[1])


def _normalize_cluster_threshold(value: Any, where: str) -> Any | None:
    """
    Normalize cluster threshold config into values accepted by MNE.

    Supported forms:
    - ``null`` / ``None`` -> ``None`` (automatic threshold)
    - number -> float
    - ``{value: ...}`` -> float or ``None``
    - ``{type: automatic}`` -> ``None``
    - ``{start: x, step: y}`` -> dict (TFCE-style threshold)
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, dict):
        raise ValueError(
            f"{where} must be null, a number, or mapping. Got {type(value).__name__}."
        )

    if "type" in value:
        mode = str(value["type"]).strip().lower()
        if mode in {"automatic", "auto", "none", "null"}:
            return None
        if mode == "value":
            value = {"value": value.get("value", None)}

    if "value" in value:
        inner = value.get("value")
        return None if inner is None else float(inner)

    if "start" in value and "step" in value:
        normalized = dict(value)
        normalized["start"] = float(value["start"])
        normalized["step"] = float(value["step"])
        return normalized

    raise ValueError(
        f"{where} mapping must contain either "
        "'value', 'type: automatic', or both 'start' and 'step'."
    )


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

    p_threshold: float = 0.01
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
            p_threshold: 0.01
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
    duration_offsets_csv: Path
    latency_offsets_csv: Path
    turn_table_csv: Path
    out_base: Path
    n_bins: int = 100


@dataclass(frozen=True)
class VizDecodingSection:
    duration_scores_npy: Path
    duration_times_npy: Path
    latency_scores_npy: Path
    latency_times_npy: Path
    duration_cluster_hdf5: Path
    latency_cluster_hdf5: Path
    out_base: Path
    p_threshold: float = 0.05
    figure_profile: str = "jneuro_2col"
    ymax: float = 0.65
    lim: float = 0.04


@dataclass(frozen=True)
class VizSection:
    """Paths and parameters for figure rendering (main + supplementary)."""

    base_out_dir: Path
    erp_timecourse: VizErpTimecourseSection
    erp_topo: VizErpTopoSection
    tfr_topos: VizTfrToposSection
    behavior: VizBehaviorSection
    erp_topomaps: VizErpTopomapsSection
    tfr_topomaps: VizTfrTopomapsSection
    erp_hist: VizErpHistSection
    decoding: VizDecodingSection

    @classmethod
    def from_dict(cls, raw: dict) -> "VizSection":
        viz_d = _require_mapping(raw, "viz")
        return _parse_viz_section(viz_d, io_out_dir=Path("."))


def _parse_viz_erp_timecourse(viz_d: dict[str, Any], *, base_out_dir: Path) -> VizErpTimecourseSection:
    section = _optional_mapping(viz_d, "erp_timecourse", "viz")
    return VizErpTimecourseSection(
        duration_long_fif=_resolve_viz_default_path(section, "duration_long_fif", base_out_dir=base_out_dir, default="erp/duration/long_ave.fif"),
        duration_short_fif=_resolve_viz_default_path(section, "duration_short_fif", base_out_dir=base_out_dir, default="erp/duration/short_ave.fif"),
        latency_fast_fif=_resolve_viz_default_path(section, "latency_fast_fif", base_out_dir=base_out_dir, default="erp/latency/fast_ave.fif"),
        latency_slow_fif=_resolve_viz_default_path(section, "latency_slow_fif", base_out_dir=base_out_dir, default="erp/latency/slow_ave.fif"),
        out_base=_resolve_viz_default_path(section, "out_base", base_out_dir=base_out_dir, default="figures/main/F_erp_timecourse"),
        xlim_ms=_float_pair(section, "xlim_ms", default=(-1500.0, 500.0), where="viz.erp_timecourse"),
        ylim_uv=_float_pair(section, "ylim_uv", default=(-2.8, 1.9), where="viz.erp_timecourse"),
    )


def _parse_viz_behavior(viz_d: dict[str, Any], *, base_out_dir: Path) -> VizBehaviorSection:
    section = _optional_mapping(viz_d, "behavior", "viz")
    return VizBehaviorSection(
        duration_offsets_csv=_resolve_viz_default_path(section, "duration_offsets_csv", base_out_dir=base_out_dir, default="erp/duration/offsets.csv"),
        latency_offsets_csv=_resolve_viz_default_path(section, "latency_offsets_csv", base_out_dir=base_out_dir, default="erp/latency/offsets.csv"),
        turn_table_csv=_resolve_viz_default_path(section, "turn_table_csv", base_out_dir=base_out_dir, default="beh/turn_table.csv"),
        out_base=_resolve_viz_default_path(section, "out_base", base_out_dir=base_out_dir, default="figures/F_behavior"),
        n_bins=int(section.get("n_bins", 100)),
    )


def _parse_viz_erp_topo(viz_d: dict[str, Any], *, base_out_dir: Path) -> VizErpTopoSection:
    section = _optional_mapping(viz_d, "erp_topo", "viz")
    return VizErpTopoSection(
        duration_cluster_hdf5=_resolve_viz_default_path(section, "duration_cluster_hdf5", base_out_dir=base_out_dir, default="stats/erp/duration/cluster_results.hdf5"),
        latency_cluster_hdf5=_resolve_viz_default_path(section, "latency_cluster_hdf5", base_out_dir=base_out_dir, default="stats/erp/latency/cluster_results.hdf5"),
        info_source_fif=_resolve_viz_default_path(section, "info_source_fif", base_out_dir=base_out_dir, default="erp/duration/difference_ave.fif"),
        out_duration=_resolve_viz_default_path(section, "out_duration", base_out_dir=base_out_dir, default="figures/supp/F_erp_topo_duration"),
        out_latency=_resolve_viz_default_path(section, "out_latency", base_out_dir=base_out_dir, default="figures/supp/F_erp_topo_latency"),
        tmin_s=float(section.get("tmin_s", -2.0)),
        tmax_s=float(section.get("tmax_s", 0.0)),
        step_ms=float(section.get("step_ms", 100)),
        max_cols=int(section.get("max_cols", 10)),
        p_threshold=float(section.get("p_threshold", 0.01)),
    )


def _parse_viz_erp_topomaps(viz_d: dict[str, Any], *, base_out_dir: Path) -> VizErpTopomapsSection:
    section = _optional_mapping(viz_d, "erp_topomaps", "viz")
    return VizErpTopomapsSection(
        template_svg=Path(section.get("template_svg", "workflow/templates/ERP-timeline.svg")),
        out_svg=_resolve_viz_default_path(section, "out_svg", base_out_dir=base_out_dir, default="figures/main/F_erp_topomap.svg"),
        parts_dir=_resolve_viz_default_path(section, "parts_dir", base_out_dir=base_out_dir, default="figures/main/parts_erp_topomap"),
        info_source_fif=_resolve_viz_default_path(section, "info_source_fif", base_out_dir=base_out_dir, default="erp/duration/difference_ave.fif"),
        duration_cluster_hdf5=_resolve_viz_default_path(section, "duration_cluster_hdf5", base_out_dir=base_out_dir, default="stats/erp/duration/cluster_results.hdf5"),
        latency_cluster_hdf5=_resolve_viz_default_path(section, "latency_cluster_hdf5", base_out_dir=base_out_dir, default="stats/erp/latency/cluster_results.hdf5"),
        p_threshold=float(section.get("p_threshold", 0.01)),
        n_duration_maps=int(section.get("n_duration_maps", 2)),
        n_latency_maps=int(section.get("n_latency_maps", 3)),
    )


def _parse_viz_tfr_topomaps(viz_d: dict[str, Any], *, base_out_dir: Path) -> VizTfrTopomapsSection:
    section = _optional_mapping(viz_d, "tfr_topomaps", "viz")
    return VizTfrTopomapsSection(
        template_svg=Path(section.get("template_svg", "workflow/templates/TF-timeline.svg")),
        out_svg=_resolve_viz_default_path(section, "out_svg", base_out_dir=base_out_dir, default="figures/main/F_tfr_topomap.svg"),
        parts_dir=_resolve_viz_default_path(section, "parts_dir", base_out_dir=base_out_dir, default="figures/main/parts_tfr_topomap"),
        info_source_fif=_resolve_viz_default_path(section, "info_source_fif", base_out_dir=base_out_dir, default="tfr/duration/alpha/difference_ave.fif"),
        alpha_cluster_hdf5=_resolve_viz_default_path(section, "alpha_cluster_hdf5", base_out_dir=base_out_dir, default="stats/tfr/duration/alpha/cluster_results.hdf5"),
        beta_cluster_hdf5=_resolve_viz_default_path(section, "beta_cluster_hdf5", base_out_dir=base_out_dir, default="stats/tfr/duration/beta/cluster_results.hdf5"),
        p_threshold=float(section.get("p_threshold", 0.05)),
        n_duration_maps=int(section.get("n_duration_maps", 2)),
        n_latency_maps=int(section.get("n_latency_maps", 3)),
    )


def _parse_viz_tfr_topos(viz_d: dict[str, Any], *, base_out_dir: Path) -> VizTfrToposSection:
    section = _optional_mapping(viz_d, "tfr_topos", "viz")
    return VizTfrToposSection(
        alpha_duration_cluster_hdf5=_resolve_viz_default_path(section, "alpha_duration_cluster_hdf5", base_out_dir=base_out_dir, default="stats/tfr/duration/alpha/cluster_results.hdf5"),
        alpha_latency_cluster_hdf5=_resolve_viz_default_path(section, "alpha_latency_cluster_hdf5", base_out_dir=base_out_dir, default="stats/tfr/latency/alpha/cluster_results.hdf5"),
        beta_duration_cluster_hdf5=_resolve_viz_default_path(section, "beta_duration_cluster_hdf5", base_out_dir=base_out_dir, default="stats/tfr/duration/beta/cluster_results.hdf5"),
        beta_latency_cluster_hdf5=_resolve_viz_default_path(section, "beta_latency_cluster_hdf5", base_out_dir=base_out_dir, default="stats/tfr/latency/beta/cluster_results.hdf5"),
        info_source_fif=_resolve_viz_default_path(section, "info_source_fif", base_out_dir=base_out_dir, default="erp/duration/difference_ave.fif"),
        out_alpha_duration=_resolve_viz_default_path(section, "out_alpha_duration", base_out_dir=base_out_dir, default="figures/supp/F_tfr_topo_alpha_duration"),
        out_alpha_latency=_resolve_viz_default_path(section, "out_alpha_latency", base_out_dir=base_out_dir, default="figures/supp/F_tfr_topo_alpha_latency"),
        out_beta_duration=_resolve_viz_default_path(section, "out_beta_duration", base_out_dir=base_out_dir, default="figures/supp/F_tfr_topo_beta_duration"),
        out_beta_latency=_resolve_viz_default_path(section, "out_beta_latency", base_out_dir=base_out_dir, default="figures/supp/F_tfr_topo_beta_latency"),
        tmin_s=float(section.get("tmin_s", -2.0)),
        tmax_s=float(section.get("tmax_s", 0.0)),
        step_ms=float(section.get("step_ms", 100)),
        max_cols=int(section.get("max_cols", 10)),
        p_threshold=float(section.get("p_threshold", 0.01)),
    )


def _parse_viz_erp_hist(viz_d: dict[str, Any], *, base_out_dir: Path) -> VizErpHistSection:
    section = _optional_mapping(viz_d, "erp_hist", "viz")
    return VizErpHistSection(
        duration_long_fif=_resolve_viz_default_path(section, "duration_long_fif", base_out_dir=base_out_dir, default="erp/duration/long_ave.fif"),
        duration_short_fif=_resolve_viz_default_path(section, "duration_short_fif", base_out_dir=base_out_dir, default="erp/duration/short_ave.fif"),
        latency_fast_fif=_resolve_viz_default_path(section, "latency_fast_fif", base_out_dir=base_out_dir, default="erp/latency/fast_ave.fif"),
        latency_slow_fif=_resolve_viz_default_path(section, "latency_slow_fif", base_out_dir=base_out_dir, default="erp/latency/slow_ave.fif"),
        hist_table_csv=_resolve_viz_default_path(section, "hist_table_csv", base_out_dir=base_out_dir, default="mixed_effect/table.csv"),
        out_base=_resolve_viz_default_path(section, "out_base", base_out_dir=base_out_dir, default="figures/main/F_erp_timecourse_hist"),
        xlim_ms=_float_pair(section, "xlim_ms", default=(-1500.0, 500.0), where="viz.erp_hist"),
        ylim_uv=_float_pair(section, "ylim_uv", default=(-2.8, 1.9), where="viz.erp_hist"),
    )


def _parse_viz_decoding(viz_d: dict[str, Any], *, base_out_dir: Path) -> VizDecodingSection:
    section = _optional_mapping(viz_d, "decoding", "viz")
    return VizDecodingSection(
        duration_scores_npy=_resolve_viz_default_path(section, "duration_scores_npy", base_out_dir=base_out_dir, default="decoding/erp/duration/scores.npy"),
        duration_times_npy=_resolve_viz_default_path(section, "duration_times_npy", base_out_dir=base_out_dir, default="decoding/erp/duration/times.npy"),
        latency_scores_npy=_resolve_viz_default_path(section, "latency_scores_npy", base_out_dir=base_out_dir, default="decoding/erp/latency/scores.npy"),
        latency_times_npy=_resolve_viz_default_path(section, "latency_times_npy", base_out_dir=base_out_dir, default="decoding/erp/latency/times.npy"),
        duration_cluster_hdf5=_resolve_viz_default_path(section, "duration_cluster_hdf5", base_out_dir=base_out_dir, default="stats/decoding/erp/duration/cluster_results.hdf5"),
        latency_cluster_hdf5=_resolve_viz_default_path(section, "latency_cluster_hdf5", base_out_dir=base_out_dir, default="stats/decoding/erp/latency/cluster_results.hdf5"),
        out_base=_resolve_viz_default_path(section, "out_base", base_out_dir=base_out_dir, default="figures/main/F_decoding"),
        p_threshold=float(section.get("p_threshold", 0.05)),
        figure_profile=str(section.get("figure_profile", "jneuro_2col")),
        ymax=float(section.get("ymax", 0.65)),
        lim=float(section.get("lim", 0.04)),
    )


def _parse_viz_section(viz_d: dict[str, Any], *, io_out_dir: Path) -> VizSection:
    base_out_dir = io_out_dir
    return VizSection(
        base_out_dir=base_out_dir,
        erp_timecourse=_parse_viz_erp_timecourse(viz_d, base_out_dir=base_out_dir),
        erp_topo=_parse_viz_erp_topo(viz_d, base_out_dir=base_out_dir),
        tfr_topos=_parse_viz_tfr_topos(viz_d, base_out_dir=base_out_dir),
        behavior=_parse_viz_behavior(viz_d, base_out_dir=base_out_dir),
        erp_topomaps=_parse_viz_erp_topomaps(viz_d, base_out_dir=base_out_dir),
        tfr_topomaps=_parse_viz_tfr_topomaps(viz_d, base_out_dir=base_out_dir),
        erp_hist=_parse_viz_erp_hist(viz_d, base_out_dir=base_out_dir),
        decoding=_parse_viz_decoding(viz_d, base_out_dir=base_out_dir),
    )



@dataclass(frozen=True)
class IoSection:
    """Filesystem layout needed to locate epochs and write analysis outputs."""

    epoch_dir: Path
    epoch_pattern: str
    out_dir: Path


@dataclass(frozen=True)
class SubjectsSection:
    """How to determine which subjects participate in the analysis."""

    mode: Literal["from_epochs", "explicit"]
    exclude: list[str]
    include: list[str]


@dataclass(frozen=True)
class DatasetSection:
    """Dataset expansion parameters used to discover epoch files on disk."""

    subjects: SubjectsSection
    tasks: list[str]
    runs: list[int]
    invalid_subject_run: list[tuple[str, str]]


@dataclass(frozen=True)
class ConstraintsSection:
    """Global selection thresholds shared by multiple analysis domains."""

    min_latency: float
    max_latency: float
    min_response_duration: float


@dataclass(frozen=True)
class AnalysisErpSection:
    @dataclass(frozen=True)
    class Artifacts:
        @dataclass(frozen=True)
        class Duration:
            long: str
            short: str

        @dataclass(frozen=True)
        class Latency:
            fast: str
            slow: str

        duration: Duration
        latency: Latency

    left_margin: float
    right_margin: float
    baseline: list[float]
    sfreq: int
    n_permutations: int
    threshold: Any | None
    artifacts: Artifacts


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
    """Analysis parameter bundle (ERP, TFR, decoding, mixed-effect)."""

    contrasts: list[Literal["duration", "latency"]]
    bands: list[str]
    erp: AnalysisErpSection
    tfr: AnalysisTfrSection
    mixed: AnalysisMixedSection
    decoding: AnalysisDecodingSection


@dataclass(frozen=True)
class ExecutionSection:
    """Resource settings used by Snakemake rules (threads/memory)."""

    threads_light: int
    threads_heavy: int
    mem_mb_light: int
    mem_mb_heavy: int


@dataclass(frozen=True)
class TurntakingConfig:
    """
    Top-level configuration object for the workflow.

    This object is produced by ``TurntakingConfig.from_dict`` (usually via
    ``turntaking.config.loader.load_config``) and passed to CLI and library
    entrypoints.
    """

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
        artifacts_raw = erp_d.get("artifacts", {})
        if artifacts_raw is None:
            artifacts_raw = {}
        artifacts_d = _require_mapping(artifacts_raw, "analysis.erp.artifacts")
        duration_d = _require_mapping(artifacts_d.get("duration", {}), "analysis.erp.artifacts.duration")
        latency_d = _require_mapping(artifacts_d.get("latency", {}), "analysis.erp.artifacts.latency")
        erp = AnalysisErpSection(
            left_margin=float(erp_d.get("left_margin", 0.0)),
            right_margin=float(erp_d.get("right_margin", 0.0)),
            baseline=[float(x) for x in _require_key(erp_d, "baseline", "analysis.erp")],
            sfreq=int(_require_key(erp_d, "sfreq", "analysis.erp")),
            n_permutations=int(_require_key(erp_d, "n_permutations", "analysis.erp")),
            threshold=_normalize_cluster_threshold(erp_d.get("threshold"), "analysis.erp.threshold"),
            artifacts=AnalysisErpSection.Artifacts(
                duration=AnalysisErpSection.Artifacts.Duration(
                    long=str(duration_d.get("long", "erp/duration/long_ave.fif")),
                    short=str(duration_d.get("short", "erp/duration/short_ave.fif")),
                ),
                latency=AnalysisErpSection.Artifacts.Latency(
                    fast=str(latency_d.get("fast", "erp/latency/fast_ave.fif")),
                    slow=str(latency_d.get("slow", "erp/latency/slow_ave.fif")),
                ),
            ),
        )

        tfr_d = _require_mapping(_require_key(analysis_d, "tfr", "analysis"), "analysis.tfr")
        tfr = AnalysisTfrSection(
            left_margin=float(_require_key(tfr_d, "left_margin", "analysis.tfr")),
            right_margin=float(_require_key(tfr_d, "right_margin", "analysis.tfr")),
            method=str(_require_key(tfr_d, "method", "analysis.tfr")),
            sfreq=int(_require_key(tfr_d, "sfreq", "analysis.tfr")),
            n_permutations=int(_require_key(tfr_d, "n_permutations", "analysis.tfr")),
            threshold=_normalize_cluster_threshold(tfr_d.get("threshold"), "analysis.tfr.threshold"),
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
            threshold=_normalize_cluster_threshold(
                decoding_d.get("threshold", None),
                "analysis.decoding.threshold",
            ),
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
        viz = _parse_viz_section(viz_d, io_out_dir=io.out_dir)

        return TurntakingConfig(
            io=io,
            dataset=dataset,
            constraints=constraints,
            analysis=analysis,
            execution=execution,
            viz=viz,
        )
