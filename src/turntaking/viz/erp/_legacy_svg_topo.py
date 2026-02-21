import argparse
from dataclasses import dataclass
from pathlib import Path

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
from turntaking.analysis.erp.io import read_cluster_outputs
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
        "viz-topomaps",
        help="Smoke-test: generate dummy topomap SVG parts and compose them into a template SVG.",
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
def run(args: argparse.Namespace, cfg) -> None:
    """
    Generate ERP topomaps from cluster-test outputs and compose into a template SVG.

    Layout (fixed timestamps):
    - Duration: -700 ms, -100 ms  -> slot_dur_tw1, slot_dur_tw2
    - Latency:  -1000 ms, -700 ms, -300 ms -> slot_lat_tw1..3

    Also exports p-text snippets into rect-anchored slots:
    - slot_dur_cluster_ptext_1
    - slot_dur_cluster_ptext_2
    - slot_lat_cluster_ptext

    Key ideas
    ---------
    - Build a global (time x channel) significance mask from all clusters with p <= threshold.
    - Slice both t-values and mask at desired timestamps (synchronized).
    - Export-time sizing is template-driven (slot bbox -> figsize_in) so typography is correct.
    - Compose step does placement only (no visual scaling logic).
    """
    from turntaking.analysis.erp.io import read_cluster_outputs
    from turntaking.viz.svg_pipeline import (
        compose_svg_from_template,
        export_colorbar_svg,
        export_ptext_svg,
        export_topomap_svg,
        read_template_slot_bboxes,
        _svg_units_to_inches,
    )

    duration_label_times_ms: list[int] = [-700, -100]
    latency_label_times_ms: list[int] = [-1000, -700, -300]

    topomaps_config = cfg.viz.erp_topomaps

    template_svg_path: Path = Path(args.template) if getattr(args, "template", None) else Path(
        topomaps_config.template_svg)

    parts_directory: Path
    if getattr(args, "parts_dir", None):
        parts_directory = Path(args.parts_dir)
    elif topomaps_config.parts_dir is not None:
        parts_directory = Path(topomaps_config.parts_dir)
    else:
        raise ValueError("No parts_dir provided. Pass --parts-dir or set viz.erp_topomaps.parts_dir in config.")

    output_svg_path: Path
    if getattr(args, "out_svg", None):
        output_svg_path = Path(args.out_svg)
    elif topomaps_config.out_svg is not None:
        output_svg_path = Path(topomaps_config.out_svg)
    else:
        raise ValueError("No out_svg provided. Pass --out-svg or set viz.erp_topomaps.out_svg in config.")

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

    # -------------------------------------------------------------------------
    # Load Info (must match the channel set/order used for cluster stats)
    # -------------------------------------------------------------------------
    info_source_path: Path = Path(topomaps_config.info_source_fif)
    evoked = mne.read_evokeds(info_source_path, condition=0, verbose="ERROR")
    info = evoked.info

    # -------------------------------------------------------------------------
    # Read cluster outputs
    # -------------------------------------------------------------------------
    duration_results_path: Path = Path(topomaps_config.duration_cluster_hdf5)
    latency_results_path: Path = Path(topomaps_config.latency_cluster_hdf5)

    duration_result = read_cluster_outputs(duration_results_path)
    latency_result = read_cluster_outputs(latency_results_path)

    metadata_duration: dict = dict(duration_result.metadata or {})
    metadata_latency: dict = dict(latency_result.metadata or {})

    # -------------------------------------------------------------------------
    # Time axes reconstructed from metadata
    # -------------------------------------------------------------------------
    times_duration_s = _times_from_metadata(metadata_duration)
    times_latency_s = _times_from_metadata(metadata_latency)

    # -------------------------------------------------------------------------
    # Validate channel compatibility
    # -------------------------------------------------------------------------
    n_channels_duration = int(metadata_duration["n_channels"])
    n_channels_latency = int(metadata_latency["n_channels"])

    if len(info.ch_names) != n_channels_duration or len(info.ch_names) != n_channels_latency:
        raise ValueError(
            "Channel mismatch between info_source_fif and cluster results.\n"
            f"info_source_fif channels: {len(info.ch_names)}\n"
            f"duration n_channels (meta): {n_channels_duration}\n"
            f"latency n_channels (meta): {n_channels_latency}\n"
            "Use an info_source_fif with the exact same channel set/order used for stats."
        )

    # -------------------------------------------------------------------------
    # Arrays
    # -------------------------------------------------------------------------
    p_value_threshold: float = float(topomaps_config.p_threshold)

    t_values_duration = np.asarray(duration_result.t_values, dtype=float)  # (n_times, n_channels)
    t_values_latency = np.asarray(latency_result.t_values, dtype=float)

    p_values_duration = np.asarray(duration_result.p_values, dtype=float)  # (n_clusters,)
    p_values_latency = np.asarray(latency_result.p_values, dtype=float)

    clusters_duration = list(duration_result.clusters)
    clusters_latency = list(latency_result.clusters)

    # -------------------------------------------------------------------------
    # Build global significance masks (time x channel) from all significant clusters
    # -------------------------------------------------------------------------
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
            # cluster is expected to be a tuple like (time_inds, channel_inds)
            mask[cluster] = True
        return mask

    sig_mask_duration = _get_mask(
        t_values=t_values_duration,
        p_values=p_values_duration,
        cluster_list=clusters_duration,
        p_threshold=p_value_threshold,
    )
    sig_mask_latency = _get_mask(
        t_values=t_values_latency,
        p_values=p_values_latency,
        cluster_list=clusters_latency,
        p_threshold=p_value_threshold,
    )

    # -------------------------------------------------------------------------
    # Minimum p among clusters active at a given time index
    # -------------------------------------------------------------------------
    def _min_p_at_time_index(
        cluster_list: list[tuple[np.ndarray, ...]],
        p_values: np.ndarray,
        time_index: int,
        p_threshold: float,
    ) -> float | None:
        candidates: list[float] = []
        for idx, cluster in enumerate(cluster_list):
            if float(p_values[idx]) > p_threshold:
                continue
            time_inds = np.asarray(cluster[0], dtype=int)
            if time_inds.size == 0:
                continue
            if np.any(time_inds == int(time_index)):
                candidates.append(float(p_values[idx]))
        if len(candidates) == 0:
            return None
        return float(min(candidates))

    # -------------------------------------------------------------------------
    # Build slot->data, slot->mask, slot->title (fixed timestamps)
    # -------------------------------------------------------------------------
    topomap_by_slot: dict[str, np.ndarray] = {}
    mask_by_slot: dict[str, np.ndarray] = {}
    title_by_slot: dict[str, str] = {}

    # Duration slots
    for slot_number, label_time_ms in enumerate(duration_label_times_ms, start=1):
        slot_id = f"slot_dur_tw{slot_number}"
        time_index = _nearest_time_index(times_duration_s, label_time_ms)

        topo_vector = t_values_duration[time_index]
        mask_vector = sig_mask_duration[time_index]

        topomap_by_slot[slot_id] = topo_vector
        mask_by_slot[slot_id] = mask_vector
        title_by_slot[slot_id] = f"{label_time_ms:+d} ms"

    # Latency slots
    for slot_number, label_time_ms in enumerate(latency_label_times_ms, start=1):
        slot_id = f"slot_lat_tw{slot_number}"
        time_index = _nearest_time_index(times_latency_s, label_time_ms)

        topo_vector = t_values_latency[time_index]
        mask_vector = sig_mask_latency[time_index]

        topomap_by_slot[slot_id] = topo_vector
        mask_by_slot[slot_id] = mask_vector
        title_by_slot[slot_id] = f"{label_time_ms:+d} ms"

    # -------------------------------------------------------------------------
    # Shared symmetric color scale across all exported maps
    # -------------------------------------------------------------------------
    absolute_max_value = float(np.max([np.max(np.abs(v)) for v in topomap_by_slot.values()]))
    vlim = (-absolute_max_value, absolute_max_value)

    # -------------------------------------------------------------------------
    # Template-driven sizing
    # -------------------------------------------------------------------------
    slot_bboxes = read_template_slot_bboxes(template_svg_path)

    # -------------------------------------------------------------------------
    # Export TOPOMAP SVG parts (export-time sizing from template slot)
    # -------------------------------------------------------------------------
    for slot_id, topo_vector in topomap_by_slot.items():
        fig_size_in = _fig_size_from_slot(slot_bboxes, slot_id)

        export_topomap_svg(
            data=topo_vector,
            info=info,
            out_svg=parts_directory / f"{slot_id}.svg",
            vlim=vlim,
            mask=mask_by_slot.get(slot_id, None),
            contours=6,
            show_sensors=False,
            title=title_by_slot.get(slot_id, None),
            fig_size_in=fig_size_in,
        )

    # -------------------------------------------------------------------------
    # Export COLORBAR SVG part (export-time sizing from template slot)
    # -------------------------------------------------------------------------
    export_colorbar_svg(
        out_svg=parts_directory / "colorbar.svg",
        vlim=vlim,
        label="t value",
        fig_size_in=_fig_size_from_slot(slot_bboxes, "slot_colorbar"),
    )

    # -------------------------------------------------------------------------
    # Export P-TEXT SVG parts (rect slots, export-time sizing)
    # -------------------------------------------------------------------------
    dur_tw1_time_index = _nearest_time_index(times_duration_s, duration_label_times_ms[0])
    dur_tw2_time_index = _nearest_time_index(times_duration_s, duration_label_times_ms[1])

    dur_p_1 = _min_p_at_time_index(
        cluster_list=clusters_duration,
        p_values=p_values_duration,
        time_index=dur_tw1_time_index,
        p_threshold=p_value_threshold,
    )
    dur_p_2 = _min_p_at_time_index(
        cluster_list=clusters_duration,
        p_values=p_values_duration,
        time_index=dur_tw2_time_index,
        p_threshold=p_value_threshold,
    )

    lat_p = _min_p_overall(p_values_latency, p_value_threshold)

    ptext_by_slot: dict[str, str] = {
        "slot_dur_cluster_ptext_1": _format_p(dur_p_1),
        "slot_dur_cluster_ptext_2": _format_p(dur_p_2),
        "slot_lat_cluster_ptext": _format_p(lat_p),
    }

    for slot_id, text in ptext_by_slot.items():
        fig_size_in = _fig_size_from_slot(slot_bboxes, slot_id)

        export_ptext_svg(
            text=text,
            out_svg=parts_directory / f"{slot_id}.svg",
            fig_size_in=fig_size_in,
            fontsize_pt=10.0,
            fontweight="normal",
        )

    # -------------------------------------------------------------------------
    # Compose final SVG (placement only; no scaling logic)
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
