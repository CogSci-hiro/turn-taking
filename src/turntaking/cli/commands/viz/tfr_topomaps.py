import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import h5py
import numpy as np
import mne

from turntaking.viz.svg_pipeline import (
    ClusterOverlay,
    compose_svg_from_template,
    export_colorbar_svg,
    export_topomap_svg,
)
from turntaking.analysis.io.cluster import read_cluster_outputs
from turntaking.viz.svg_pipeline import (
    ClusterOverlay,
    compose_svg_from_template,
    export_colorbar_svg,
    export_topomap_svg,
    export_ptext_svg,
    read_template_slot_bboxes,
    _svg_units_to_inches,
)


# =============================================================================
#                     ########################################
#                     #            CLI REGISTRATION          #
#                     ########################################
# =============================================================================


@dataclass(frozen=True)
class _ClusterResult:
    t_vals: np.ndarray          # (n_times, n_channels)
    clusters: np.ndarray        # (n_clusters, n_times, n_channels) bool
    p_vals: np.ndarray          # (n_clusters,)
    meta: dict

def _read_meta_json(h5: h5py.File) -> dict:
    raw = h5["meta"]["json"][()]
    # h5py may return bytes, numpy scalar, or str depending on how it was saved
    if isinstance(raw, (bytes, bytearray)):
        s = raw.decode("utf-8")
    else:
        s = str(raw)
    return json.loads(s)

def _read_cluster_hdf5(path: Path) -> _ClusterResult:
    with h5py.File(path, "r") as h5:
        t_vals = np.asarray(h5["t-values"])
        clusters = np.asarray(h5["clusters"]).astype(bool)
        p_vals = np.asarray(h5["p-values"]).astype(float)
        meta = _read_meta_json(h5)

    return _ClusterResult(t_vals=t_vals, clusters=clusters, p_vals=p_vals, meta=meta)

def _times_from_metadata(metadata: dict) -> np.ndarray:
    """
    Reconstruct time axis for the stats array.

    Assumes:
    - data_tmin is already the tmin of the array used in stats
    - n_times matches t_values.shape[0]
    """
    sfreq_hz = float(metadata.get("crop_sfreq_used", metadata["sfreq_hz"]))
    n_times = int(metadata["n_times"])
    tmin_s = float(metadata["data_tmin"])  # already cropped tmin

    return tmin_s + (np.arange(n_times, dtype=float) / sfreq_hz)

def _assert_time_consistent(meta: dict) -> None:
    n_times = int(meta["n_times"])
    crop_start = int(meta["crop_start_idx"])
    crop_end = int(meta["crop_end_idx"])
    expected = (crop_end - crop_start) + 1
    if expected != n_times:
        raise ValueError(f"meta mismatch: n_times={n_times} but crop window implies {expected} samples.")


def _ensure_time_by_channel(x: np.ndarray, n_times: int, n_channels: int, name: str) -> np.ndarray:
    if x.shape == (n_times, n_channels):
        return x
    if x.shape == (n_channels, n_times):
        return x.T
    raise ValueError(f"{name} has shape {x.shape}, expected {(n_times, n_channels)} or {(n_channels, n_times)}.")


def _ensure_clusters_shape(x: np.ndarray, n_times: int, n_channels: int) -> np.ndarray:
    # expected: (n_clusters, n_times, n_channels)
    if x.ndim != 3:
        raise ValueError(f"clusters must be 3D, got {x.shape}")
    if x.shape[1:] == (n_times, n_channels):
        return x.astype(bool)
    if x.shape[1:] == (n_channels, n_times):
        return np.transpose(x, (0, 2, 1)).astype(bool)
    raise ValueError(f"clusters has shape {x.shape}, expected (*, {n_times}, {n_channels}) or (*, {n_channels}, {n_times}).")



def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Register the `viz-topomaps` command.

    Usage example
    -------------
        python -m turntaking.cli.main viz-topomaps \
            --config workflow/config.yaml \
            --template workflow/templates/ERP-timeline.svg \
            --parts-dir workflow/results/parts_erp_topomap \
            --out-svg workflow/results/F_erp_topomap.svg
    """
    p = subparsers.add_parser(
        "viz-tfr-topomaps",
        help="Generate TFR (alpha/beta) topomap SVG parts and compose into a template SVG.",
    )

    # Required by cli/main.py (even if unused for smoke test)
    p.add_argument("--config", type=str, required=True, help="Path to project config YAML.")

    p.add_argument("--template", type=Path, required=True, help="Template SVG containing slot_* anchors.")
    p.add_argument("--parts-dir", type=Path, required=True, help="Directory to write part SVGs.")
    p.add_argument("--out-svg", type=Path, required=True, help="Output composed SVG.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for dummy data.")
    p.add_argument("--n-channels", type=int, default=64, help="Number of channels from standard_1020.")

    p.set_defaults(func=run)


# =============================================================================
#                     ########################################
#                     #                 RUN                 #
#                     ########################################
# =============================================================================
def run(args: argparse.Namespace, cfg: Any) -> None:
    """
    Generate TFR topomap SVG parts from cluster-test outputs (alpha + beta) and
    compose them into a template SVG.

    Expected slots in template
    --------------------------
    Alpha:
        slot_alpha_tw1
        slot_alpha_tw2
        slot_alpha_tw3
        slot_alpha_cluster_ptext

    Beta:
        slot_beta_tw1
        slot_beta_tw2
        slot_beta_cluster_ptext

    Also:
        slot_colorbar

    Notes
    -----
    - Cluster outputs are assumed to be ERP-like: t_values is (n_times, n_channels)
      and clusters index (time_inds, channel_inds).
    - Export-time sizing is template-driven (slot bbox -> figsize_in).
    - Compose step does placement only.
    """
    from turntaking.analysis.io.cluster import read_cluster_outputs
    from turntaking.viz.svg_pipeline import (
        compose_svg_from_template,
        export_colorbar_svg,
        export_ptext_svg,
        export_topomap_svg,
        read_template_slot_bboxes,
        _svg_units_to_inches,
    )

    # -------------------------------------------------------------------------
    # Slot timepoints (ms)
    # -------------------------------------------------------------------------
    # Keep these aligned with your template layout.
    # You can later move them into config if you want full flexibility.
    alpha_label_times_ms: list[int] = [-1000, -700, -300]   # 3 alpha slots
    beta_label_times_ms: list[int] = [-700, -100]          # 2 beta slots

    section = cfg.viz.tfr_topomaps

    template_svg_path: Path = Path(section.template_svg)
    parts_directory: Path
    output_svg_path: Path

    # Prefer CLI outputs if present (Snakemake passes these)
    if getattr(args, "parts_dir", None):
        parts_directory = Path(args.parts_dir)
    else:
        # Optional config fallback if you kept it
        parts_directory = Path(getattr(section, "parts_dir"))

    if getattr(args, "out_svg", None):
        output_svg_path = Path(args.out_svg)
    else:
        output_svg_path = Path(getattr(section, "out_svg"))

    parts_directory.mkdir(parents=True, exist_ok=True)
    output_svg_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Local helpers
    # -------------------------------------------------------------------------
    def _nearest_time_index(times_s: np.ndarray, target_ms: float) -> int:
        target_s = float(target_ms) / 1000.0
        return int(np.argmin(np.abs(times_s - target_s)))

    def _fig_size_from_slot(
        slot_bboxes: dict[str, tuple[float, float, float, float]],
        slot_id: str,
    ) -> tuple[float, float]:
        _x, _y, w, h = slot_bboxes[slot_id]
        return (_svg_units_to_inches(w), _svg_units_to_inches(h))

    def _format_p(p: float | None) -> str:
        if p is None:
            return "n.s."
        if p < 0.001:
            return "p < 0.001"
        return f"p = {p:.3f}"

    def _min_p_overall(p_values: np.ndarray, p_threshold: float) -> float | None:
        candidates = [float(p) for p in np.asarray(p_values).ravel() if float(p) <= float(p_threshold)]
        if len(candidates) == 0:
            return None
        return float(min(candidates))

    def _get_mask(
        t_values: np.ndarray,
        p_values: np.ndarray,
        cluster_list: list[tuple],
        p_threshold: float,
    ) -> np.ndarray:
        mask = np.zeros_like(t_values, dtype=bool)
        for idx, cluster in enumerate(cluster_list):
            if float(p_values[idx]) > p_threshold:
                continue
            mask[cluster] = True
        return mask

    # -------------------------------------------------------------------------
    # Load Info (must match channel set/order used for stats)
    # -------------------------------------------------------------------------
    info_source_path: Path = Path(section.info_source_fif)
    evoked = mne.read_evokeds(info_source_path, condition=0, verbose="ERROR")
    info = evoked.info

    # -------------------------------------------------------------------------
    # Read cluster outputs (alpha + beta)
    # -------------------------------------------------------------------------
    alpha_results_path: Path = Path(section.alpha_cluster_hdf5)
    beta_results_path: Path = Path(section.beta_cluster_hdf5)

    alpha_result = read_cluster_outputs(alpha_results_path)
    beta_result = read_cluster_outputs(beta_results_path)

    meta_alpha: dict = dict(alpha_result.metadata or {})
    meta_beta: dict = dict(beta_result.metadata or {})

    # -------------------------------------------------------------------------
    # Time axes from metadata (assumes same helper works for TFR)
    # -------------------------------------------------------------------------
    times_alpha_s = _times_from_metadata(meta_alpha)
    times_beta_s = _times_from_metadata(meta_beta)

    # -------------------------------------------------------------------------
    # Channel compatibility
    # -------------------------------------------------------------------------
    n_channels_alpha = int(meta_alpha["n_channels"])
    n_channels_beta = int(meta_beta["n_channels"])
    if len(info.ch_names) != n_channels_alpha or len(info.ch_names) != n_channels_beta:
        raise ValueError(
            "Channel mismatch between info_source_fif and TFR cluster results.\n"
            f"info_source_fif channels: {len(info.ch_names)}\n"
            f"alpha n_channels (meta): {n_channels_alpha}\n"
            f"beta  n_channels (meta): {n_channels_beta}\n"
            "Use an info_source_fif with the exact same channel set/order used for stats."
        )

    # -------------------------------------------------------------------------
    # Arrays
    # -------------------------------------------------------------------------
    p_value_threshold: float = float(getattr(section, "p_threshold", 0.05))

    t_values_alpha = np.asarray(alpha_result.t_values, dtype=float)  # (n_times, n_channels)
    t_values_beta = np.asarray(beta_result.t_values, dtype=float)

    p_values_alpha = np.asarray(alpha_result.p_values, dtype=float)
    p_values_beta = np.asarray(beta_result.p_values, dtype=float)

    clusters_alpha = list(alpha_result.clusters)
    clusters_beta = list(beta_result.clusters)

    sig_mask_alpha = _get_mask(
        t_values=t_values_alpha,
        p_values=p_values_alpha,
        cluster_list=clusters_alpha,
        p_threshold=p_value_threshold,
    )
    sig_mask_beta = _get_mask(
        t_values=t_values_beta,
        p_values=p_values_beta,
        cluster_list=clusters_beta,
        p_threshold=p_value_threshold,
    )

    # -------------------------------------------------------------------------
    # Build slot->data/mask/title for ALPHA and BETA separately
    # -------------------------------------------------------------------------
    topomap_by_slot: dict[str, np.ndarray] = {}
    mask_by_slot: dict[str, np.ndarray] = {}
    title_by_slot: dict[str, str] = {}

    # Alpha slots (3)
    for slot_number, label_time_ms in enumerate(alpha_label_times_ms, start=1):
        slot_id = f"slot_alpha_tw{slot_number}"
        time_index = _nearest_time_index(times_alpha_s, label_time_ms)

        topomap_by_slot[slot_id] = t_values_alpha[time_index]
        mask_by_slot[slot_id] = sig_mask_alpha[time_index]
        title_by_slot[slot_id] = f"{label_time_ms:+d} ms"

    # Beta slots (2)
    for slot_number, label_time_ms in enumerate(beta_label_times_ms, start=1):
        slot_id = f"slot_beta_tw{slot_number}"
        time_index = _nearest_time_index(times_beta_s, label_time_ms)

        topomap_by_slot[slot_id] = t_values_beta[time_index]
        mask_by_slot[slot_id] = sig_mask_beta[time_index]
        title_by_slot[slot_id] = f"{label_time_ms:+d} ms"

    # -------------------------------------------------------------------------
    # Shared symmetric color scale across ALL exported maps (alpha+beta)
    # -------------------------------------------------------------------------
    absolute_max_value = float(np.max([np.max(np.abs(v)) for v in topomap_by_slot.values()]))
    vlim = (-absolute_max_value, absolute_max_value)

    # -------------------------------------------------------------------------
    # Template-driven sizing
    # -------------------------------------------------------------------------
    slot_bboxes = read_template_slot_bboxes(template_svg_path)

    # -------------------------------------------------------------------------
    # Export topomap parts
    # -------------------------------------------------------------------------
    for slot_id, topo_vector in topomap_by_slot.items():
        export_topomap_svg(
            data=topo_vector,
            info=info,
            out_svg=parts_directory / f"{slot_id}.svg",
            vlim=vlim,
            mask=mask_by_slot.get(slot_id, None),
            contours=6,
            show_sensors=False,
            title=title_by_slot.get(slot_id, None),
            fig_size_in=_fig_size_from_slot(slot_bboxes, slot_id),
        )

    # -------------------------------------------------------------------------
    # Export shared colorbar
    # -------------------------------------------------------------------------
    export_colorbar_svg(
        out_svg=parts_directory / "colorbar.svg",
        vlim=vlim,
        label="t value",
        fig_size_in=_fig_size_from_slot(slot_bboxes, "slot_colorbar"),
    )

    # -------------------------------------------------------------------------
    # Export band-level ptext (2 slots total)
    # -------------------------------------------------------------------------
    alpha_p = _min_p_overall(p_values_alpha, p_value_threshold)
    beta_p = _min_p_overall(p_values_beta, p_value_threshold)

    ptext_by_slot: dict[str, str] = {
        "slot_alpha_cluster_ptext": _format_p(alpha_p),
        "slot_beta_cluster_ptext": _format_p(beta_p),
    }

    for slot_id, text in ptext_by_slot.items():
        export_ptext_svg(
            text=text,
            out_svg=parts_directory / f"{slot_id}.svg",
            fig_size_in=_fig_size_from_slot(slot_bboxes, slot_id),
            fontsize_pt=10.0,
            fontweight="normal",
        )

    # -------------------------------------------------------------------------
    # Compose final SVG
    # -------------------------------------------------------------------------
    slot_to_snippet: dict[str, Path] = {slot: parts_directory / f"{slot}.svg" for slot in topomap_by_slot.keys()}
    slot_to_snippet["slot_colorbar"] = parts_directory / "colorbar.svg"
    slot_to_snippet.update({slot: parts_directory / f"{slot}.svg" for slot in ptext_by_slot.keys()})

    compose_svg_from_template(
        template_svg=template_svg_path,
        slot_to_snippet=slot_to_snippet,
        out_svg=output_svg_path,
    )

def _fig_size_from_slot(slot_bboxes: dict[str, tuple[float, float, float, float]], slot_id: str) -> tuple[float, float]:
    _x, _y, w, h = slot_bboxes[slot_id]
    return (_svg_units_to_inches(w), _svg_units_to_inches(h))


def _format_p(p: float | None) -> str:
    if p is None:
        return "n.s."
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _min_p_overall(p_values: np.ndarray, p_threshold: float) -> float | None:
    candidates = [float(p) for p in np.asarray(p_values).ravel() if float(p) <= float(p_threshold)]
    if len(candidates) == 0:
        return None
    return float(min(candidates))
