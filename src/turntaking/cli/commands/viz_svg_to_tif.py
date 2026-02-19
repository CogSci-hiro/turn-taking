import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# =============================================================================
#                     ########################################
#                     #        SVG -> TIFF VIZ COMMAND       #
#                     ########################################
# =============================================================================

_DEFAULT_DPI: float = 300.0
_DEFAULT_COMPRESSION: str = "tiff_lzw"


@dataclass(frozen=True)
class SvgToTiffVizConfig:
    """
    Config for converting an SVG figure into a TIFF.

    Notes
    -----
    - This uses CairoSVG to rasterize SVG -> PNG at a specified DPI, then Pillow to
      save PNG -> TIFF with compression.
    - Keep SVG as the "source of truth" (deterministic layout), and derive TIFF.

    Usage example
    -------------
        cfg = SvgToTiffVizConfig(
            in_svg=Path("figures/main/F_erp_topomap.svg"),
            out_tif=Path("figures/main/F_erp_topomap.tif"),
            dpi=300.0,
            compression="tiff_lzw",
            keep_intermediate_png=False,
        )
    """

    in_svg: Path
    out_tif: Path
    out_png: Path | None = None
    out_eps: Path | None = None
    dpi: float = _DEFAULT_DPI
    compression: str = _DEFAULT_COMPRESSION
    keep_intermediate_png: bool = False


def _require_exists(path: Path, context: str) -> None:
    """
    Raise a clear error if a required file does not exist.

    Parameters
    ----------
    path
        Path to validate.
    context
        String describing where/why this file is required.

    Usage example
    -------------
        _require_exists(Path("figure.svg"), context="SVG input")
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {context}: {path}")


def _ensure_parent_dir(path: Path) -> None:
    """
    Ensure the parent directory of a path exists.

    Parameters
    ----------
    path
        Output path.

    Usage example
    -------------
        _ensure_parent_dir(Path("out/figure.tif"))
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _svg_to_png_bytes(in_svg: Path, dpi: float) -> bytes:
    """
    Rasterize an SVG file to PNG bytes at a given DPI.

    Parameters
    ----------
    in_svg
        Input SVG file.
    dpi
        Dots per inch used by CairoSVG.

    Returns
    -------
    png_bytes
        PNG image bytes.

    Usage example
    -------------
        png_bytes = _svg_to_png_bytes(Path("figure.svg"), dpi=300.0)
    """
    try:
        import cairosvg
    except ImportError as e:
        raise ImportError(
            "CairoSVG is required for SVG->PNG conversion. Install it with:\n"
            "  pip install cairosvg"
        ) from e

    # CairoSVG accepts file paths via url=
    return cairosvg.svg2png(url=str(in_svg), dpi=float(dpi))


def _svg_to_eps_bytes(in_svg: Path, dpi: float) -> bytes:
    """
    Convert an SVG file to EPS bytes.

    Parameters
    ----------
    in_svg
        Input SVG file.
    dpi
        Dots per inch used by CairoSVG.

    Returns
    -------
    eps_bytes
        EPS bytes.
    """
    try:
        import cairosvg
    except ImportError as e:
        raise ImportError(
            "CairoSVG is required for SVG->EPS conversion. Install it with:\n"
            "  pip install cairosvg"
        ) from e

    return cairosvg.svg2ps(url=str(in_svg), dpi=float(dpi))


def _write_tiff_from_png_bytes(
    png_bytes: bytes,
    out_tif: Path,
    compression: str,
) -> None:
    """
    Save PNG bytes to a TIFF file using Pillow.

    Parameters
    ----------
    png_bytes
        PNG bytes in memory.
    out_tif
        TIFF output path.
    compression
        Pillow TIFF compression string (e.g., "tiff_lzw", "tiff_adobe_deflate", "raw").

    Usage example
    -------------
        _write_tiff_from_png_bytes(png_bytes, Path("out.tif"), compression="tiff_lzw")
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Pillow is required for PNG->TIFF conversion. Install it with:\n"
            "  pip install pillow"
        ) from e

    from io import BytesIO

    with Image.open(BytesIO(png_bytes)) as im:
        # Ensure a consistent mode for saving
        im_converted = im.convert("RGBA") if im.mode not in {"RGB", "RGBA"} else im
        im_converted.save(out_tif, format="TIFF", compression=compression)


def _run_impl(cfg: SvgToTiffVizConfig) -> None:
    """
    Convert SVG -> TIFF via SVG -> PNG (CairoSVG) and PNG -> TIFF (Pillow).

    Parameters
    ----------
    cfg
        Conversion config.

    Usage example
    -------------
        _run_impl(SvgToTiffVizConfig(
            in_svg=Path("fig.svg"),
            out_tif=Path("fig.tif"),
            dpi=300.0,
        ))
    """
    _require_exists(cfg.in_svg, context="input SVG")
    _ensure_parent_dir(cfg.out_tif)
    if cfg.out_png is not None:
        _ensure_parent_dir(cfg.out_png)
    if cfg.out_eps is not None:
        _ensure_parent_dir(cfg.out_eps)

    png_bytes = _svg_to_png_bytes(cfg.in_svg, dpi=cfg.dpi)
    _write_tiff_from_png_bytes(
        png_bytes=png_bytes,
        out_tif=cfg.out_tif,
        compression=cfg.compression,
    )

    if cfg.out_png is not None:
        cfg.out_png.write_bytes(png_bytes)

    if cfg.out_eps is not None:
        eps_bytes = _svg_to_eps_bytes(cfg.in_svg, dpi=cfg.dpi)
        cfg.out_eps.write_bytes(eps_bytes)

    if cfg.keep_intermediate_png:
        png_path = cfg.out_tif.with_suffix(".png")
        png_path.write_bytes(png_bytes)


def run(args: argparse.Namespace, cfg: Any) -> None:
    """
    CLI entrypoint: convert an SVG file to a TIFF file.

    Notes
    -----
    - This command is intentionally config-light: it primarily follows CLI args,
      which plays nicely with Snakemake (outputs are workflow-defined).

    Usage example
    -------------
        python -m turntaking.cli.main viz-svg-to-tiff \
          --config workflow/config.yaml \
          --in-svg /path/to/F_erp_topomap.svg \
          --out-tif /path/to/F_erp_topomap.tif \
          --dpi 300
    """
    in_svg = Path(args.in_svg)
    out_tif = Path(args.out_tif)
    out_png = Path(args.out_png) if args.out_png else None
    out_eps = Path(args.out_eps) if args.out_eps else None

    viz_cfg = SvgToTiffVizConfig(
        in_svg=in_svg,
        out_tif=out_tif,
        out_png=out_png,
        out_eps=out_eps,
        dpi=float(args.dpi),
        compression=str(args.compression),
        keep_intermediate_png=bool(args.keep_png),
    )

    _run_impl(viz_cfg)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Register the `viz-svg-to-tiff` command.

    Usage example
    -------------
        python -m turntaking.cli.main viz-svg-to-tiff \
          --config workflow/config.yaml \
          --in-svg figures/main/F_erp_topomap.svg \
          --out-tif figures/main/F_erp_topomap.tif \
          --dpi 300
    """
    p = subparsers.add_parser(
        "viz-svg-to-tiff",
        help="Convert an SVG figure into a TIFF (SVG->PNG via CairoSVG, PNG->TIFF via Pillow).",
    )
    p.add_argument("--config", type=str, required=True, help="Path to project config YAML (required by CLI harness).")
    p.add_argument("--in-svg", type=str, required=True, help="Input SVG path.")
    p.add_argument("--out-tif", type=str, required=True, help="Output TIFF path.")
    p.add_argument("--out-png", type=str, default=None, help="Optional PNG output path.")
    p.add_argument("--out-eps", type=str, default=None, help="Optional EPS output path.")
    p.add_argument("--dpi", type=float, default=_DEFAULT_DPI, help=f"Rasterization DPI (default: {_DEFAULT_DPI}).")
    p.add_argument(
        "--compression",
        type=str,
        default=_DEFAULT_COMPRESSION,
        help=f"TIFF compression (default: {_DEFAULT_COMPRESSION}).",
    )
    p.add_argument(
        "--keep-png",
        action="store_true",
        help="Also write an intermediate PNG next to the TIFF (same stem).",
    )
    p.set_defaults(func=run)
