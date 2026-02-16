"""Decoding visualization renderer."""


from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from turntaking.analysis.decoding.io import ContrastName, DecodingScorePaths, get_decoding_out_dir, load_decoding_scores
from turntaking.viz._style import apply_style, save_figure
from turntaking.viz.figures.decoding import plot_decoding
from turntaking.viz.utils import cfg_get_optional, out_dir, resolve_from_out_dir


@dataclass(frozen=True)
class _ClusterResults:
    clusters: list[tuple[np.ndarray, np.ndarray]]
    p_values: np.ndarray


def _cluster_results(analysis_out_dir: Any, contrast: ContrastName) -> _ClusterResults:
    path = out_dir(analysis_out_dir) / "stats" / "decoding" / "erp" / contrast / "cluster_results.hdf5"
    if not path.exists():
        raise FileNotFoundError(f"Missing decoding cluster results: {path}")

    clusters: list[tuple[np.ndarray, np.ndarray]] = []
    with h5py.File(path, "r") as handle:
        p_values = np.asarray(handle["p-values"], dtype=float)
        index = 0
        while f"clusters/train-{index}" in handle and f"clusters/test-{index}" in handle:
            train_idx = np.asarray(handle[f"clusters/train-{index}"], dtype=int)
            test_idx = np.asarray(handle[f"clusters/test-{index}"], dtype=int)
            clusters.append((train_idx, test_idx))
            index += 1
    return _ClusterResults(clusters=clusters, p_values=p_values)


def _scores(analysis_out_dir: Any, contrast: ContrastName) -> tuple[np.ndarray, np.ndarray]:
    directory = get_decoding_out_dir(out_dir(analysis_out_dir), contrast)
    return load_decoding_scores(DecodingScorePaths.from_dir(directory))


def render(cfg: Any) -> None:
    apply_style("jneuro_2col")
    duration_scores, duration_times = _scores(cfg, "duration")
    latency_scores, latency_times = _scores(cfg, "latency")
    if duration_times.shape != latency_times.shape or not np.allclose(duration_times, latency_times):
        raise ValueError("Duration and latency decoding times differ; cannot render joint figure.")

    duration_stats = _cluster_results(cfg, "duration")
    latency_stats = _cluster_results(cfg, "latency")
    p_threshold = float(cfg_get_optional(cfg, "viz", "decoding", "p_threshold", default=0.05))
    ymax = float(cfg_get_optional(cfg, "viz", "decoding", "ymax", default=0.65))
    profile = str(cfg_get_optional(cfg, "viz", "decoding", "figure_profile", default="jneuro_2col"))
    save_basepath = resolve_from_out_dir(cfg, "figures/main/F_decoding")

    fig = plot_decoding(
        tmin=float(duration_times.min()),
        tmax=float(duration_times.max()),
        duration_scores=duration_scores,
        latency_scores=latency_scores,
        duration_clusters=duration_stats.clusters,
        latency_clusters=latency_stats.clusters,
        duration_p=duration_stats.p_values,
        latency_p=latency_stats.p_values,
        p_threshold=p_threshold,
        ymax=ymax,
    )
    save_figure(fig, save_basepath=save_basepath, profile_name=profile)
