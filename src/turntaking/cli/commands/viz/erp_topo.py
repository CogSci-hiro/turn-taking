import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np

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
    out_base: Path
    tmin_ms: float
    tmax_ms: float
    n_topo: int
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


def _load_cluster_outputs(path: Path) -> tuple[np.ndarray, np.ndarray, list[tuple], float]:
    """
    Load cluster test outputs saved by the pipeline.

    Notes
    -----
    This assumes your project already writes cluster results in the format that
    `turntaking.analysis.io.cluster` knows how to read.

    Returns
    -------
    t
        t-stat array (time x channels) or (channels x time) depending on writer.
    p
        p-value array, same shape as t.
    clusters
        Cluster list (as returned by MNE permutation cluster test).
    data_tmin
        The tmin in seconds for the data arrays.

    Usage example
    -------------
        t, p, clusters, data_tmin = _load_cluster_outputs(Path(".../cluster_results.hdf5"))
    """
    if not path.exists():
        raise FileNotFoundError(f"Cluster results not found: {path}")

    # Prefer project-native loader to avoid guessing HDF5 key names.
    try:
        from turntaking.analysis.io.cluster import read_cluster_outputs  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import turntaking.analysis.io.cluster.read_cluster_outputs. "
            "Either add it, or implement an HDF5 reader here."
        ) from e

    out = read_cluster_outputs(path)

    # Be forgiving about exact return type: support dict-like or tuple-like.
    if isinstance(out, dict):
        t = np.asarray(out["t"])
        p = np.asarray(out["p"])
        clusters = list(out["clusters"])
        data_tmin = float(out["tmin"])
        return t, p, clusters, data_tmin

    # Otherwise assume tuple ordering
    t, p, clusters, data_tmin = out
    return np.asarray(t), np.asarray(p), list(clusters), float(data_tmin)


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
        tmin=cfg.tmin_ms,
        tmax=cfg.tmax_ms,
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
        tmin_ms=float(section.tmin_ms),
        tmax_ms=float(section.tmax_ms),
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
