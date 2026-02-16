"""ERP topomap renderer with static/SVG format switching."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from turntaking.viz._style import apply_style
from turntaking.viz.erp._legacy_static_topo import ErpTopoVizConfig, _run_impl as run_static_topomap
from turntaking.viz.erp._legacy_svg_topo import run as run_svg_topomap
from turntaking.viz.utils import cfg_get_optional, resolve_from_out_dir


def _format(cfg: Any, explicit: str | None) -> str:
    if explicit:
        return explicit
    return str(cfg_get_optional(cfg, "viz", "erp", "topomap", "format", default="static"))


def _static_params(cfg: Any) -> ErpTopoVizConfig:
    return ErpTopoVizConfig(
        duration_cluster_hdf5=resolve_from_out_dir(cfg, "stats/erp/duration/cluster_results.hdf5"),
        latency_cluster_hdf5=resolve_from_out_dir(cfg, "stats/erp/latency/cluster_results.hdf5"),
        info_source_fif=resolve_from_out_dir(cfg, "erp/duration/difference_ave.fif"),
        out_duration=resolve_from_out_dir(cfg, "figures/supp/F_erp_topo_duration"),
        out_latency=resolve_from_out_dir(cfg, "figures/supp/F_erp_topo_latency"),
        tmin_s=float(cfg_get_optional(cfg, "viz", "erp_topo", "tmin_s", default=-2.0)),
        tmax_s=float(cfg_get_optional(cfg, "viz", "erp_topo", "tmax_s", default=0.0)),
        step_ms=float(cfg_get_optional(cfg, "viz", "erp_topo", "step_ms", default=100.0)),
        max_cols=int(cfg_get_optional(cfg, "viz", "erp_topo", "max_cols", default=10)),
        p_threshold=float(cfg_get_optional(cfg, "viz", "erp_topo", "p_threshold", default=0.01)),
    )


def _svg_cfg(cfg: Any) -> tuple[argparse.Namespace, Any]:
    template = Path(cfg_get_optional(cfg, "viz", "erp", "topomap", "template", default="workflow/templates/ERP-timeline.svg"))
    parts_dir = resolve_from_out_dir(cfg, "figures/main/parts_erp_topomap")
    out_svg = resolve_from_out_dir(cfg, "figures/main/F_erp_topomap.svg")
    section = SimpleNamespace(
        template_svg=template,
        parts_dir=parts_dir,
        out_svg=out_svg,
        info_source_fif=resolve_from_out_dir(cfg, "erp/duration/difference_ave.fif"),
        duration_cluster_hdf5=resolve_from_out_dir(cfg, "stats/erp/duration/cluster_results.hdf5"),
        latency_cluster_hdf5=resolve_from_out_dir(cfg, "stats/erp/latency/cluster_results.hdf5"),
        p_threshold=float(cfg_get_optional(cfg, "viz", "erp_topomaps", "p_threshold", default=0.05)),
        n_duration_maps=int(cfg_get_optional(cfg, "viz", "erp_topomaps", "n_duration_maps", default=2)),
        n_latency_maps=int(cfg_get_optional(cfg, "viz", "erp_topomaps", "n_latency_maps", default=3)),
    )
    args = argparse.Namespace(config=None, template=template, parts_dir=parts_dir, out_svg=out_svg)
    legacy_cfg = SimpleNamespace(viz=SimpleNamespace(erp_topomaps=section))
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
    raise ValueError(f"Unsupported ERP topomap format: {chosen!r}")
