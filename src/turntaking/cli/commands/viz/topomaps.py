from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import mne

from turntaking.viz.svg_pipeline import (
    ClusterOverlay,
    compose_svg_from_template,
    export_colorbar_svg,
    export_topomap_svg,
)

# =============================================================================
#                     ########################################
#                     #            CLI REGISTRATION          #
#                     ########################################
# =============================================================================
def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Register the `viz-topomaps` command.

    Usage example
    -------------
        python -m turntaking.cli.main viz-topomaps \
            --config workflow/config.yaml \
            --template workflow/templates/ERP-timeline.svg \
            --parts-dir workflow/results/parts_erp_topomap \
            --out-svg workflow/results/F_erp_topomap.svg
    """
    p = subparsers.add_parser(
        "viz-topomaps",
        help="Smoke-test: generate dummy topomap SVG parts and compose them into a template SVG.",
    )

    # Required by cli/main.py (even if unused for smoke test)
    p.add_argument("--config", type=str, required=True, help="Path to project config YAML.")

    p.add_argument("--template", type=Path, required=True, help="Template SVG containing slot_* anchors.")
    p.add_argument("--parts-dir", type=Path, required=True, help="Directory to write part SVGs.")
    p.add_argument("--out-svg", type=Path, required=True, help="Output composed SVG.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for dummy data.")
    p.add_argument("--n-channels", type=int, default=64, help="Number of channels from standard_1020.")

    p.set_defaults(func=run)


# =============================================================================
#                     ########################################
#                     #                 RUN                 #
#                     ########################################
# =============================================================================
def run(args: argparse.Namespace, cfg) -> None:
    """
    Generate dummy topomap SVG parts and compose into the template SVG.

    Notes
    -----
    This is intentionally minimal. Later you replace dummy maps with real
    ERP/TFR-derived arrays but keep the same slot/compose machinery.
    """
    template_svg: Path = args.template
    parts_dir: Path = args.parts_dir
    out_svg: Path = args.out_svg

    parts_dir.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    # Minimal Info for testing
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = montage.ch_names[: int(args.n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types="eeg")
    info.set_montage(montage)

    rng = np.random.default_rng(int(args.seed))

    slot_ids = [
        "slot_dur_tw1",
        "slot_dur_tw2",
        "slot_lat_tw1",
        "slot_lat_tw2",
        "slot_lat_tw3",
    ]

    maps = {slot: rng.normal(size=len(ch_names)) for slot in slot_ids}

    vmax = max(float(np.max(np.abs(v))) for v in maps.values())
    vlim = (-vmax, vmax)

    overlays = [
        ClusterOverlay(
            name="C1",
            ch_names=("Fz", "FCz", "Cz", "Pz"),
            marker="s",
            markersize=9.0,
            markeredgecolor="black",
            markerfacecolor="none",
            markeredgewidth=1.5,
        )
    ]

    # Export parts
    for slot_id, data in maps.items():
        export_topomap_svg(
            data=data,
            info=info,
            out_svg=parts_dir / f"{slot_id}.svg",
            vlim=vlim,
            overlays=overlays,
            contours=0,
            show_sensors=False,
            title=None,
        )

    export_colorbar_svg(
        out_svg=parts_dir / "colorbar.svg",
        vlim=vlim,
        label="t value",
    )

    # Compose
    slot_to_snippet = {slot: parts_dir / f"{slot}.svg" for slot in slot_ids}
    slot_to_snippet["slot_colorbar"] = parts_dir / "colorbar.svg"

    compose_svg_from_template(
        template_svg=template_svg,
        slot_to_snippet=slot_to_snippet,
        out_svg=out_svg,
    )
