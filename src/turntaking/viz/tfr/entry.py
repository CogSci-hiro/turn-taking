"""TFR visualization entrypoint."""


from typing import Any

from turntaking.viz.utils import viz_mode


def run(cfg: Any, *, mode: str | None = None, topomap_format: str | None = None) -> None:
    """
    Run a TFR visualization mode.

    Parameters
    ----------
    cfg
        Loaded ``TurntakingConfig`` (or a compatible mapping-like object).
    mode
        Visualization mode override. Defaults to ``viz_mode(cfg, "tfr")``.
    topomap_format
        Optional format name for topomap rendering (e.g. ``"static"`` or ``"svg"``).
    """
    selected_mode = mode or viz_mode(cfg, "tfr", default="topomap")
    if selected_mode == "topomap":
        from turntaking.viz.tfr.topomap import render

        render(cfg, format_name=topomap_format)
        return
    raise ValueError(f"Unsupported TFR viz mode: {selected_mode!r}")
