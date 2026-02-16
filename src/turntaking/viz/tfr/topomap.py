"""TFR topomap renderer with static/SVG format switching."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from turntaking.viz._style import apply_style
from turntaking.viz.tfr._legacy_static_topo import TfrToposVizConfig, _run_impl as run_static_topomap
from turntaking.viz.tfr._legacy_svg_topo import run as run_svg_topomap
from turntaking.viz.utils import cfg_get_optional, resolve_from_out_dir


def _format(cfg: Any, explicit: str | None) -> str:
    if explicit:
        return explicit
    return str(cfg_get_optional(cfg, "viz", "tfr", "topomap", "format", default="static"))


def _static_params(cfg: Any) -> TfrToposVizConfig:
    return TfrToposVizConfig(
        alpha_duration_cluster_hdf5=resolve_from_out_dir(cfg, "stats/tfr/duration/alpha/cluster_results.hdf5"),
        alpha_latency_cluster_hdf5=resolve_from_out_dir(cfg, "stats/tfr/latency/alpha/cluster_results.hdf5"),
        beta_duration_cluster_hdf5=resolve_from_out_dir(cfg, "stats/tfr/duration/beta/cluster_results.hdf5"),
        beta_latency_cluster_hdf5=resolve_from_out_dir(cfg, "stats/tfr/latency/beta/cluster_results.hdf5"),
        info_source_fif=resolve_from_out_dir(cfg, "erp/duration/difference_ave.fif"),
        out_alpha_duration=resolve_from_out_dir(cfg, "figures/supp/F_tfr_topo_alpha_duration"),
        out_alpha_latency=resolve_from_out_dir(cfg, "figures/supp/F_tfr_topo_alpha_latency"),
        out_beta_duration=resolve_from_out_dir(cfg, "figures/supp/F_tfr_topo_beta_duration"),
        out_beta_latency=resolve_from_out_dir(cfg, "figures/supp/F_tfr_topo_beta_latency"),
        tmin_s=float(cfg_get_optional(cfg, "viz", "tfr_topos", "tmin_s", default=-2.0)),
        tmax_s=float(cfg_get_optional(cfg, "viz", "tfr_topos", "tmax_s", default=0.0)),
        step_ms=float(cfg_get_optional(cfg, "viz", "tfr_topos", "step_ms", default=100.0)),
        max_cols=int(cfg_get_optional(cfg, "viz", "tfr_topos", "max_cols", default=10)),
        p_threshold=float(cfg_get_optional(cfg, "viz", "tfr_topos", "p_threshold", default=0.01)),
    )


def _svg_cfg(cfg: Any) -> tuple[argparse.Namespace, Any]:
    template = Path(cfg_get_optional(cfg, "viz", "tfr", "topomap", "template", default="workflow/templates/TF-timeline.svg"))
    parts_dir = resolve_from_out_dir(cfg, "figures/main/parts_tfr_topomap")
    out_svg = resolve_from_out_dir(cfg, "figures/main/F_tfr_topomap.svg")
    section = SimpleNamespace(
        template_svg=template,
        parts_dir=parts_dir,
        out_svg=out_svg,
        info_source_fif=resolve_from_out_dir(cfg, "tfr/duration/alpha/difference_ave.fif"),
        alpha_cluster_hdf5=resolve_from_out_dir(cfg, "stats/tfr/duration/alpha/cluster_results.hdf5"),
        beta_cluster_hdf5=resolve_from_out_dir(cfg, "stats/tfr/duration/beta/cluster_results.hdf5"),
        p_threshold=float(cfg_get_optional(cfg, "viz", "tfr_topomaps", "p_threshold", default=0.05)),
        n_duration_maps=int(cfg_get_optional(cfg, "viz", "tfr_topomaps", "n_duration_maps", default=2)),
        n_latency_maps=int(cfg_get_optional(cfg, "viz", "tfr_topomaps", "n_latency_maps", default=3)),
    )
    args = argparse.Namespace(config=None, template=template, parts_dir=parts_dir, out_svg=out_svg)
    legacy_cfg = SimpleNamespace(viz=SimpleNamespace(tfr_topomaps=section))
    return args, legacy_cfg


def render(cfg: Any, *, format_name: str | None = None) -> None:
    apply_style("jneuro_2col")
    chosen = _format(cfg, format_name)
    if chosen == "static":
        run_static_topomap(_static_params(cfg))
        return
    if chosen == "svg":
        args, legacy_cfg = _svg_cfg(cfg)
        run_svg_topomap(args, legacy_cfg)
        return
    raise ValueError(f"Unsupported TFR topomap format: {chosen!r}")
