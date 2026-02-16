"""ERP visualization entrypoint."""

from __future__ import annotations

from typing import Any

from turntaking.viz.utils import viz_mode


def run(cfg: Any, *, mode: str | None = None, topomap_format: str | None = None) -> None:
    selected_mode = mode or viz_mode(cfg, "erp", default="timecourse")
    if selected_mode == "timecourse":
        from turntaking.viz.erp.timecourse import render

        render(cfg)
        return
    if selected_mode == "hist":
        from turntaking.viz.erp.hist import render

        render(cfg)
        return
    if selected_mode == "topomap":
        from turntaking.viz.erp.topomap import render

        render(cfg, format_name=topomap_format)
        return
    raise ValueError(f"Unsupported ERP viz mode: {selected_mode!r}")

