from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np

from turntaking.analysis.io.cluster import read_cluster_outputs


# =============================================================================
#                     ########################################
#                     #         TFR TOPO VIZ COMMAND         #
#                     ########################################
# =============================================================================


@dataclass(frozen=True)
class TfrToposVizConfig:
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


def _run_impl(cfg: TfrToposVizConfig) -> None:
    from matplotlib import pyplot as plt
    from turntaking.viz._style import save_figure
    from turntaking.viz.figures.tfr import (
        plot_tfr_topo_alpha_duration,
        plot_tfr_topo_alpha_latency,
        plot_tfr_topo_beta_duration,
        plot_tfr_topo_beta_latency,
    )

    info = _load_info_from_evoked(cfg.info_source_fif)

    a_d_t, a_d_p, a_d_clusters, data_tmin_1 = _load_cluster_outputs(cfg.alpha_duration_cluster_hdf5)
    a_l_t, a_l_p, a_l_clusters, data_tmin_2 = _load_cluster_outputs(cfg.alpha_latency_cluster_hdf5)
    b_d_t, b_d_p, b_d_clusters, data_tmin_3 = _load_cluster_outputs(cfg.beta_duration_cluster_hdf5)
    b_l_t, b_l_p, b_l_clusters, data_tmin_4 = _load_cluster_outputs(cfg.beta_latency_cluster_hdf5)

    data_tmins = [data_tmin_1, data_tmin_2, data_tmin_3, data_tmin_4]
    if max(data_tmins) - min(data_tmins) > 1e-9:
        raise ValueError(
            "TFR topo inputs have mismatched data_tmin values: "
            + ", ".join(f"{v:.12g}" for v in data_tmins)
        )
    data_tmin = data_tmin_1

    for out_path in (
        cfg.out_alpha_duration,
        cfg.out_alpha_latency,
        cfg.out_beta_duration,
        cfg.out_beta_latency,
    ):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    out_alpha_duration_base = cfg.out_alpha_duration.with_suffix("")
    out_alpha_latency_base = cfg.out_alpha_latency.with_suffix("")
    out_beta_duration_base = cfg.out_beta_duration.with_suffix("")
    out_beta_latency_base = cfg.out_beta_latency.with_suffix("")

    fig_a_d = plot_tfr_topo_alpha_duration(
        t_values=a_d_t,
        p_values=a_d_p,
        clusters=a_d_clusters,
        info=info,
        data_tmin=data_tmin,
        tmin=cfg.tmin_s,
        tmax=cfg.tmax_s,
        step_ms=cfg.step_ms,
        p_threshold=cfg.p_threshold,
        max_cols=cfg.max_cols,
    )
    save_figure(fig_a_d, save_basepath=out_alpha_duration_base, profile_name="jneuro_2col")
    plt.close(fig_a_d)

    fig_a_l = plot_tfr_topo_alpha_latency(
        t_values=a_l_t,
        p_values=a_l_p,
        clusters=a_l_clusters,
        info=info,
        data_tmin=data_tmin,
        tmin=cfg.tmin_s,
        tmax=cfg.tmax_s,
        step_ms=cfg.step_ms,
        p_threshold=cfg.p_threshold,
        max_cols=cfg.max_cols,
    )
    save_figure(fig_a_l, save_basepath=out_alpha_latency_base, profile_name="jneuro_2col")
    plt.close(fig_a_l)

    fig_b_d = plot_tfr_topo_beta_duration(
        t_values=b_d_t,
        p_values=b_d_p,
        clusters=b_d_clusters,
        info=info,
        data_tmin=data_tmin,
        tmin=cfg.tmin_s,
        tmax=cfg.tmax_s,
        step_ms=cfg.step_ms,
        p_threshold=cfg.p_threshold,
        max_cols=cfg.max_cols,
    )
    save_figure(fig_b_d, save_basepath=out_beta_duration_base, profile_name="jneuro_2col")
    plt.close(fig_b_d)

    fig_b_l = plot_tfr_topo_beta_latency(
        t_values=b_l_t,
        p_values=b_l_p,
        clusters=b_l_clusters,
        info=info,
        data_tmin=data_tmin,
        tmin=cfg.tmin_s,
        tmax=cfg.tmax_s,
        step_ms=cfg.step_ms,
        p_threshold=cfg.p_threshold,
        max_cols=cfg.max_cols,
    )
    save_figure(fig_b_l, save_basepath=out_beta_latency_base, profile_name="jneuro_2col")
    plt.close(fig_b_l)


def run(args: argparse.Namespace, cfg: Any) -> None:
    section = cfg.viz.tfr_topos

    viz_cfg = TfrToposVizConfig(
        alpha_duration_cluster_hdf5=Path(section.alpha_duration_cluster_hdf5),
        alpha_latency_cluster_hdf5=Path(section.alpha_latency_cluster_hdf5),
        beta_duration_cluster_hdf5=Path(section.beta_duration_cluster_hdf5),
        beta_latency_cluster_hdf5=Path(section.beta_latency_cluster_hdf5),
        info_source_fif=Path(section.info_source_fif),
        out_alpha_duration=Path(section.out_alpha_duration),
        out_alpha_latency=Path(section.out_alpha_latency),
        out_beta_duration=Path(section.out_beta_duration),
        out_beta_latency=Path(section.out_beta_latency),
        tmin_s=float(section.tmin_s),
        tmax_s=float(section.tmax_s),
        step_ms=float(section.step_ms),
        max_cols=int(getattr(section, "max_cols", 10)),
        p_threshold=float(getattr(section, "p_threshold", 0.01)),
    )

    for path in (
        viz_cfg.alpha_duration_cluster_hdf5,
        viz_cfg.alpha_latency_cluster_hdf5,
        viz_cfg.beta_duration_cluster_hdf5,
        viz_cfg.beta_latency_cluster_hdf5,
        viz_cfg.info_source_fif,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input file: {path}")

    _run_impl(viz_cfg)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "viz-tfr-topos",
        help="Plot TFR t-value topomaps as four figures (alpha/beta x duration/latency).",
    )
    parser.add_argument("--config", required=True)
