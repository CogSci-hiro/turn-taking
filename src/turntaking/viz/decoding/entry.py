"""Decoding visualization entrypoint."""


from typing import Any

from turntaking.viz.utils import viz_mode


def run(cfg: Any, *, mode: str | None = None) -> None:
    """
    Run a decoding visualization mode.

    Parameters
    ----------
    cfg
        Loaded ``TurntakingConfig`` (or a compatible mapping-like object).
    mode
        Visualization mode override. Defaults to ``viz_mode(cfg, "decoding")``.
        Currently supported value: ``"figure"``.
    """
    selected_mode = mode or viz_mode(cfg, "decoding", default="figure")
    if selected_mode == "figure":
        from turntaking.viz.decoding.figure import render

        render(cfg)
        return
    raise ValueError(f"Unsupported decoding viz mode: {selected_mode!r}")
