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
from turntaking.analysis.io.cluster import read_cluster_outputs


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

    This implements the "B" layout:
    - Duration: 2 topomaps (best 2 clusters by p-value, thresholded then fallback)
    - Latency:  3 topomaps (best 3 clusters by p-value, thresholded then fallback)

    The command reads all paths/settings from the config section `viz.erp_topomaps`.
    """
    # Local imports to keep CLI startup light and to avoid import cycles.
    from turntaking.analysis.io.cluster import read_cluster_outputs

    duration_label_times_ms = [-700, -100]
    latency_label_times_ms = [-1000, -700, -300]

    topomaps_config = cfg.viz.erp_topomaps

    template_svg_path: Path = Path(topomaps_config.template_svg)
    parts_directory: Path = Path(topomaps_config.parts_dir)
    output_svg_path: Path = Path(topomaps_config.out_svg)

    parts_directory.mkdir(parents=True, exist_ok=True)
    output_svg_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load real Info (must match the channel set/order used for cluster stats)
    # -------------------------------------------------------------------------
    info_source_path: Path = Path(topomaps_config.info_source_fif)
    evoked = mne.read_evokeds(info_source_path, condition=0, verbose="ERROR")
    info = evoked.info


    # -------------------------------------------------------------------------
    # Read cluster outputs (canonical reader matches write_cluster_outputs)
    # -------------------------------------------------------------------------
    duration_results_path: Path = Path(topomaps_config.duration_cluster_hdf5)
    latency_results_path: Path = Path(topomaps_config.latency_cluster_hdf5)

    duration_result = read_cluster_outputs(duration_results_path)
    latency_result = read_cluster_outputs(latency_results_path)

    metadata_duration: dict = dict(duration_result.metadata or {})
    metadata_latency: dict = dict(latency_result.metadata or {})

    # -------------------------------------------------------------------------
    # Reconstruct time axis from metadata (no explicit times stored in HDF5)
    # ---------------------------------------------------------------

    def _nearest_time_index(times_s: np.ndarray, target_ms: float) -> int:
        target_s = float(target_ms) / 1000.0
        return int(np.argmin(np.abs(times_s - target_s)))

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
    # Convenience: choose clusters by p-threshold with fallback to best p-values
    # -------------------------------------------------------------------------
    p_value_threshold: float = float(topomaps_config.p_threshold)
    n_duration_maps: int = int(topomaps_config.n_duration_maps)
    n_latency_maps: int = int(topomaps_config.n_latency_maps)

    def _pick_cluster_indices(p_values: np.ndarray, n_keep: int) -> list[int]:
        sorted_indices = list(np.argsort(p_values))
        passing_indices = [i for i in sorted_indices if float(p_values[i]) <= p_value_threshold]
        if len(passing_indices) >= n_keep:
            return passing_indices[:n_keep]
        return sorted_indices[:n_keep]

    p_values_duration = np.asarray(duration_result.p_values, dtype=float)
    p_values_latency = np.asarray(latency_result.p_values, dtype=float)

    duration_cluster_indices = _pick_cluster_indices(p_values_duration, n_duration_maps)
    latency_cluster_indices = _pick_cluster_indices(p_values_latency, n_latency_maps)

    # -------------------------------------------------------------------------
    # Compute topomap vectors + overlay channel markers for a single cluster
    # -------------------------------------------------------------------------
    t_values_duration = np.asarray(duration_result.t_values, dtype=float)
    t_values_latency = np.asarray(latency_result.t_values, dtype=float)

    marker_cycle = ["o", "s", "^", "D", "P", "X"]  # distinguish different clusters

    def _summarize_cluster(
            t_values: np.ndarray,
            times_s: np.ndarray,
            cluster: tuple[np.ndarray, ...],
            *,
            topo_time_ms: float | None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float], int]:
        """
        Convert cluster (time inds, channel inds) to:
        - topo vector (n_channels,) at `topo_time_ms` if provided, else cluster-mean
        - channel mask (n_channels,) bool (cluster channels)
        - time window (tmin, tmax) of the cluster (for debugging)
        - topo_time_index used for the topo vector
        """
        if len(cluster) < 2:
            raise ValueError(f"Expected spatiotemporal cluster with >=2 dims, got {len(cluster)} dims.")

        time_indices = np.asarray(cluster[0], dtype=int)
        channel_indices = np.asarray(cluster[1], dtype=int)

        mask = np.zeros(len(info.ch_names), dtype=bool)

        if time_indices.size == 0:
            raise ValueError("Cluster has no time indices.")
        if channel_indices.size == 0:
            raise ValueError("Cluster has no channel indices.")

        unique_time_indices = np.unique(time_indices)
        unique_channel_indices = np.unique(channel_indices)
        mask[unique_channel_indices] = True

        tmin_s = float(times_s[unique_time_indices[0]])
        tmax_s = float(times_s[unique_time_indices[-1]])

        if topo_time_ms is None:
            topo_vector = t_values[unique_time_indices].mean(axis=0)
            topo_time_index = int(unique_time_indices[len(unique_time_indices) // 2])
            return topo_vector, mask, (tmin_s, tmax_s), topo_time_index

        topo_time_index = _nearest_time_index(times_s, topo_time_ms)
        topo_vector = t_values[topo_time_index]
        return topo_vector, mask, (tmin_s, tmax_s), topo_time_index

    # -------------------------------------------------------------------------
    # Build slot->data, slot->overlays, slot->titles (B layout)
    # -------------------------------------------------------------------------
    topomap_by_slot: dict[str, np.ndarray] = {}
    mask_by_slot: dict[str, np.ndarray] = {}
    title_by_slot: dict[str, str] = {}

    # Duration slots: slot_dur_tw1, slot_dur_tw2
    for slot_number, cluster_index in enumerate(duration_cluster_indices, start=1):
        slot_id = f"slot_dur_tw{slot_number}"

        label_time_ms = duration_label_times_ms[slot_number - 1]

        topo_vector, cluster_mask, (tmin_s, tmax_s), topo_time_index = _summarize_cluster(
            t_values=t_values_duration,
            times_s=times_duration_s,
            cluster=duration_result.clusters[cluster_index],
            topo_time_ms=label_time_ms,
        )

        cluster_p_value = float(p_values_duration[cluster_index])

        topomap_by_slot[slot_id] = topo_vector
        mask_by_slot[slot_id] = cluster_mask
        label_time_ms = duration_label_times_ms[slot_number - 1]
        title_by_slot[slot_id] = f"{label_time_ms:+d} ms (p={cluster_p_value:.3g})"

    # Latency slots: slot_lat_tw1, slot_lat_tw2, slot_lat_tw3
    for slot_number, cluster_index in enumerate(latency_cluster_indices, start=1):
        slot_id = f"slot_lat_tw{slot_number}"

        label_time_ms = latency_label_times_ms[slot_number - 1]

        topo_vector, cluster_mask, (tmin_s, tmax_s), topo_time_index = _summarize_cluster(
            t_values=t_values_latency,
            times_s=times_latency_s,
            cluster=latency_result.clusters[cluster_index],
            topo_time_ms=label_time_ms,
        )

        cluster_p_value = float(p_values_latency[cluster_index])

        topomap_by_slot[slot_id] = topo_vector
        mask_by_slot[slot_id] = cluster_mask
        label_time_ms = latency_label_times_ms[slot_number - 1]
        title_by_slot[slot_id] = f"{label_time_ms:+d} ms (p={cluster_p_value:.3g})"

    # -------------------------------------------------------------------------
    # Shared symmetric color scale across all exported maps
    # -------------------------------------------------------------------------
    absolute_max_value = float(
        np.max([np.max(np.abs(values)) for values in topomap_by_slot.values()])
    )
    vlim = (-absolute_max_value, absolute_max_value)

    # -------------------------------------------------------------------------
    # Export SVG parts
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
        )

    export_colorbar_svg(
        out_svg=parts_directory / "colorbar.svg",
        vlim=vlim,
        label="t value",
    )

    # -------------------------------------------------------------------------
    # Compose final SVG
    # -------------------------------------------------------------------------
    slot_to_snippet = {slot: parts_directory / f"{slot}.svg" for slot in topomap_by_slot.keys()}
    slot_to_snippet["slot_colorbar"] = parts_directory / "colorbar.svg"

    compose_svg_from_template(
        template_svg=template_svg_path,
        slot_to_snippet=slot_to_snippet,
        out_svg=output_svg_path,
    )

