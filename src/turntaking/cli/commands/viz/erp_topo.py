import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np

from turntaking.analysis.io.cluster import read_cluster_outputs

# =============================================================================
#                     ########################################
#                     #         ERP TOPO VIZ COMMAND         #
#                     ########################################
# =============================================================================


@dataclass(frozen=True)
class ErpTopoVizConfig:
    duration_cluster_hdf5: Path
    latency_cluster_hdf5: Path
    info_source_fif: Path

    out_duration: Path
    out_latency: Path

    tmin_s: float
    tmax_s: float
    step_ms: float
    max_cols: int
    p_threshold: float


def _load_info_from_evoked(path: Path) -> mne.Info:
    """
    Load an MNE Info object from an evoked FIF file.

    Parameters
    ----------
    path
        Path to a FIF file that contains at least one Evoked.

    Returns
    -------
    info
        MNE Info for topomap plotting.

    Usage example
    -------------
        info = _load_info_from_evoked(Path("erp/duration/difference_ave.fif"))
    """
    if not path.exists():
        raise FileNotFoundError(f"Evoked FIF not found: {path}")
    evokeds = mne.read_evokeds(path, condition=None, verbose="ERROR")
    if len(evokeds) == 0:
        raise ValueError(f"No Evoked objects found in: {path}")
    return evokeds[0].info


def _require_float(meta: dict[str, Any], key: str, context: str) -> float:
    if key not in meta:
        raise KeyError(
            f"Missing metadata key {key!r} in {context}. "
            f"Available keys: {sorted(meta.keys())}"
        )
    return float(meta[key])


def _load_cluster_outputs(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, ...]], float]:
    """
    Load cluster results as a simple tuple for plotting.

    Returns
    -------
    t_values
        Array of t-values.
    p_values
        Array of cluster p-values.
    clusters
        List of cluster index tuples.
    data_tmin
        Start time (seconds) for the t-value time axis.
    """
    out = read_cluster_outputs(path)

    t_values = np.asarray(out.t_values, dtype=float)
    p_values = np.asarray(out.p_values, dtype=float)
    clusters = list(out.clusters)

    # Make this strict to avoid silent wrong time alignment:
    data_tmin = _require_float(out.metadata, "data_tmin", context=str(path))

    return t_values, p_values, clusters, data_tmin


def _run_impl(cfg: ErpTopoVizConfig) -> None:
    from turntaking.viz.figures.erp import plot_erp_topo
    from matplotlib import pyplot as plt

    info = _load_info_from_evoked(cfg.info_source_fif)

    duration_t, duration_p, duration_clusters, data_tmin = _load_cluster_outputs(cfg.duration_cluster_hdf5)
    latency_t, latency_p, latency_clusters, data_tmin_2 = _load_cluster_outputs(cfg.latency_cluster_hdf5)

    # Sanity check: both stats should align to same data_tmin if produced similarly
    if abs(data_tmin - data_tmin_2) > 1e-9:
        raise ValueError(f"duration data_tmin={data_tmin} != latency data_tmin={data_tmin_2}")

    fig = plot_erp_topo(
        duration_t=duration_t,
        latency_t=latency_t,
        duration_p=duration_p,
        latency_p=latency_p,
        duration_cluster=duration_clusters,
        latency_cluster=latency_clusters,
        info=info,
        data_tmin=data_tmin,
        tmin=cfg.tmin_s,
        tmax=cfg.tmax_s,
        n_topo=cfg.n_topo,
        p_threshold=cfg.p_threshold,
        figure_profile="jneuro_2col",
        save_basepath=cfg.out_base,
    )

    plt.close(fig)


def run(args: argparse.Namespace, cfg: Any) -> None:
    section = cfg.viz.erp_topo

    viz_cfg = ErpTopoVizConfig(
        duration_cluster_hdf5=Path(section.duration_cluster_hdf5),
        latency_cluster_hdf5=Path(section.latency_cluster_hdf5),
        info_source_fif=Path(section.info_source_fif),
        out_base=Path(section.out_base),
        tmin_s=float(section.tmin_s),
        tmax_s=float(section.tmax_s),
        n_topo=int(section.n_topo),
        p_threshold=float(getattr(section, "p_threshold", 0.01)),
    )

    _run_impl(viz_cfg)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "viz-erp-topo",
        help="Plot ERP t-value topomaps for duration and latency cluster stats.",
    )
    parser.add_argument("--config", required=True)
