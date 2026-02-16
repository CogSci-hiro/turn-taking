"""Decoding visualization entrypoint."""


from typing import Any

from turntaking.viz.utils import viz_mode


def run(cfg: Any, *, mode: str | None = None) -> None:
    selected_mode = mode or viz_mode(cfg, "decoding", default="figure")
    if selected_mode == "figure":
        from turntaking.viz.decoding.figure import render

        render(cfg)
        return
    raise ValueError(f"Unsupported decoding viz mode: {selected_mode!r}")

