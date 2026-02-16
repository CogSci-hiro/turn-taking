"""Shared style and constants for turn-taking visualizations.

This module intentionally contains only *stable* plotting constants so that figures
remain visually identical across refactors.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.figure
from matplotlib import pyplot as plt

P_THRESHOLD = 0.05  # default significance threshold used in plots

DURATION_COLOR_1 = "#010fcc"  # true blue
DURATION_COLOR_2 = "#8ab8fe"  # carolina blue
LATENCY_COLOR_1 = "#e50000"  # red
LATENCY_COLOR_2 = "#ffb19a"  # pale salmon

JOINT_TIMES = (-2.0, -0.8, 0.0)  # times for joint plot

WIDTH = 8.27  # A4
TITLE_FONT_SIZE = 14
FONT_SIZE = 11
MARKER_SIZE = 15
SMALLER_MARKER_SIZE = 5
FACE_COLOR = "darkgray"

# ======================================================================================================================
# Figure export + journal sizing (JNeurosci-oriented)
# ======================================================================================================================

CM_PER_INCH: float = 2.54

JNEURO_1COL_WIDTH_CM: float = 8.5
JNEURO_2COL_WIDTH_CM: float = 17.4  # common full-width; adjust if you standardize a different value

DEFAULT_EXPORT_DPI: int = 300
DEFAULT_TIFF_EXTENSION: str = ".tif"

DEFAULT_TIFF_KWARGS: dict[str, object] = {
    "dpi": DEFAULT_EXPORT_DPI,
    "bbox_inches": "tight",
    "pad_inches": 0.02,
}

DEFAULT_EPS_KWARGS: dict[str, object] = {
    "bbox_inches": "tight",
    "pad_inches": 0.02,
}

# NOTE: Some EPS workflows need explicit font embedding. This is generally safe and helps journals/typesetters.
# If you already set these elsewhere, keep only one copy.
plt.rcParams["ps.fonttype"] = 42  # TrueType
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["savefig.dpi"] = DEFAULT_EXPORT_DPI


def cm_to_inch(length_cm: float) -> float:
    """
    Convert centimeters to inches.

    Parameters
    ----------
    length_cm : float
        Length in centimeters.

    Returns
    -------
    float
        Length in inches.

    Usage example
    -------------
        width_in = cm_to_inch(8.5)
    """
    return length_cm / CM_PER_INCH


@dataclass(frozen=True)
class FigureProfile:
    """
    Standardized figure sizing + export defaults.

    Attributes
    ----------
    width_in : float
        Target figure width in inches.
    dpi : int
        Export DPI (used for raster formats like TIFF).
    formats : tuple[str, ...]
        Formats to export, e.g. ("tiff", "eps").
    tight_layout : bool
        Whether to apply tight_layout before saving.

    Usage example
    -------------
        profile = FIGURE_PROFILES["jneuro_1col"]
    """

    width_in: float
    dpi: int = DEFAULT_EXPORT_DPI
    formats: tuple[str, ...] = ("tiff",)
    tight_layout: bool = True


FIGURE_PROFILES: Mapping[str, FigureProfile] = {
    "jneuro_1col": FigureProfile(width_in=cm_to_inch(JNEURO_1COL_WIDTH_CM), formats=("tiff", "eps")),
    "jneuro_2col": FigureProfile(width_in=cm_to_inch(JNEURO_2COL_WIDTH_CM), formats=("tiff", "eps")),
    "screen": FigureProfile(width_in=7.0, formats=("png",), dpi=150),  # handy for quick debugging
}


def apply_figure_profile(
    fig: matplotlib.figure.Figure,
    profile_name: str,
) -> None:
    """
    Apply a FigureProfile's width while preserving aspect ratio.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to modify in-place.
    profile_name : str
        Key into FIGURE_PROFILES.

    Usage example
    -------------
        apply_figure_profile(fig, "jneuro_1col")
    """
    if profile_name not in FIGURE_PROFILES:
        valid = ", ".join(sorted(FIGURE_PROFILES.keys()))
        raise ValueError(f"Unknown figure profile: {profile_name!r}. Valid: {valid}")

    profile = FIGURE_PROFILES[profile_name]

    current_width_in, current_height_in = fig.get_size_inches()
    if current_width_in <= 0:
        raise ValueError("Figure width is non-positive; cannot apply profile.")

    scale = profile.width_in / current_width_in
    new_height_in = current_height_in * scale
    fig.set_size_inches(profile.width_in, new_height_in, forward=True)


def apply_style(profile_name: str = "jneuro_2col") -> FigureProfile:
    """
    Apply global plotting style and return the selected export profile.
    """
    if profile_name not in FIGURE_PROFILES:
        valid = ", ".join(sorted(FIGURE_PROFILES.keys()))
        raise ValueError(f"Unknown figure profile: {profile_name!r}. Valid: {valid}")
    plt.rcParams["font.size"] = FONT_SIZE
    plt.rcParams["axes.titlesize"] = TITLE_FONT_SIZE
    plt.rcParams["axes.labelsize"] = FONT_SIZE
    return FIGURE_PROFILES[profile_name]


def save_figure(
    fig: matplotlib.figure.Figure,
    save_basepath: str | Path,
    profile_name: str = "jneuro_1col",
    formats: Sequence[str] | None = None,
) -> list[Path]:
    """
    Save a figure using a global export policy (TIFF/EPS + DPI + final-size).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    save_basepath : str | Path
        Path without extension (recommended). If extension is included, it will be replaced per format.
    profile_name : str
        FigureProfile key (e.g., "jneuro_1col").
    formats : Sequence[str] | None
        Override formats; if None uses the profile defaults.

    Returns
    -------
    list[pathlib.Path]
        List of written file paths.

    Notes
    -----
    - "tiff" and "tif" both produce ".tif"
    - EPS may not preserve transparency perfectly; prefer TIFF for topomaps/alpha-heavy plots.

    Usage example
    -------------
        paths = save_figure(fig, "out/figures/erp_timecourse", profile_name="jneuro_1col")
    """
    if profile_name not in FIGURE_PROFILES:
        valid = ", ".join(sorted(FIGURE_PROFILES.keys()))
        raise ValueError(f"Unknown figure profile: {profile_name!r}. Valid: {valid}")

    profile = FIGURE_PROFILES[profile_name]
    export_formats = tuple(formats) if formats is not None else profile.formats

    # Apply sizing at final print width before saving
    apply_figure_profile(fig, profile_name)

    if profile.tight_layout:
        # tight_layout can conflict with manually placed colorbar axes; OK to call but may be skipped by users
        try:
            fig.tight_layout()
        except Exception:
            pass

    basepath = Path(save_basepath)
    basepath.parent.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for fmt in export_formats:
        fmt_norm = fmt.lower().strip().lstrip(".")
        if fmt_norm in ("tif", "tiff"):
            out_path = basepath.with_suffix(DEFAULT_TIFF_EXTENSION)
            fig.savefig(out_path, format="tiff", **DEFAULT_TIFF_KWARGS)
            written.append(out_path)
        elif fmt_norm == "eps":
            out_path = basepath.with_suffix(".eps")
            fig.savefig(out_path, format="eps", **DEFAULT_EPS_KWARGS)
            written.append(out_path)
        elif fmt_norm == "png":
            out_path = basepath.with_suffix(".png")
            fig.savefig(out_path, format="png", dpi=profile.dpi, bbox_inches="tight", pad_inches=0.02)
            written.append(out_path)
        elif fmt_norm == "pdf":
            out_path = basepath.with_suffix(".pdf")
            fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
            written.append(out_path)
        else:
            raise ValueError(f"Unsupported export format: {fmt!r}")

    return written
