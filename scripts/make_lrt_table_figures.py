#!/usr/bin/env python3
"""
Render LRT comparison tables (duration + latency) to .tif/.eps/.png.

This script is intentionally standalone: it works with a src/ layout without
requiring an editable install of the package.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from turntaking.viz._style import (  # noqa: E402
    DEFAULT_EPS_KWARGS,
    DEFAULT_TIFF_EXTENSION,
    DEFAULT_TIFF_KWARGS,
    FIGURE_PROFILES,
    apply_figure_profile,
    apply_style,
)


REQUIRED_COLUMNS: tuple[str, ...] = (
    "outcome",
    "window",
    "family",
    "roi",
    "lrt_chisq",
    "lrt_p",
)


def _fdr_bh(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR correction (returns q-values).

    Notes
    -----
    - Preserves NaNs.
    - Clips to [0, 1].
    """
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan, dtype=float)

    valid_mask = np.isfinite(p_values)
    p_valid = p_values[valid_mask]
    if p_valid.size == 0:
        return q_values

    order = np.argsort(p_valid)
    p_sorted = p_valid[order]
    n_tests = p_sorted.size

    ranks = np.arange(1, n_tests + 1, dtype=float)
    q_sorted = p_sorted * n_tests / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q_valid = np.empty_like(p_valid, dtype=float)
    q_valid[order] = q_sorted
    q_values[valid_mask] = q_valid
    return q_values


def _stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value <= 0.001:
        return "***"
    if p_value <= 0.01:
        return "**"
    if p_value <= 0.05:
        return "*"
    return ""


_SUPERSCRIPT_MAP: dict[str, str] = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "-": "⁻",
    "+": "⁺",
}


def _superscript_int(value: int) -> str:
    """
    Convert an integer to a Unicode superscript string (e.g., -14 -> ⁻¹⁴).
    """
    text = str(int(value))
    return "".join(_SUPERSCRIPT_MAP.get(ch, ch) for ch in text)


def _format_p(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        sci = f"{p_value:.2e}"  # e.g. 4.85e-05
        mantissa_str, exp_str = sci.split("e", 1)
        exponent = int(exp_str)
        return f"{mantissa_str} x 10{_superscript_int(exponent)}"
    return f"{p_value:.3f}"


def _format_chisq(chisq: float) -> str:
    if not np.isfinite(chisq):
        return ""
    return f"{chisq:.2f}"


def _format_beta_se(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:.3f}"


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")


def _family_label(family: str) -> str:
    family_str = str(family).strip()
    key = family_str.lower()
    if key == "erp":
        return "ERP"
    if key == "alpha":
        return "Alpha"
    if key == "beta":
        return "Beta"
    return family_str[:1].upper() + family_str[1:]


def _window_label(window: str) -> str:
    return str(window).upper()


def _roi_label(roi: str) -> str:
    roi_str = str(roi)
    return roi_str[:1].upper() + roi_str[1:]


def _ensure_beta_se_columns(df: pd.DataFrame, *, lrt_csv_path: Path) -> pd.DataFrame:
    """
    Ensure df has `beta` and `se`.

    Priority:
    1) Use existing `beta`/`se` if present.
    2) Backfill from sibling fixed_effects.csv (full model, matching predictor term).
    3) Fall back to NaN (rendered as blank in the figure table).
    """
    out = df.copy()
    has_beta = "beta" in out.columns
    has_se = "se" in out.columns
    if has_beta and has_se:
        return out

    out["beta"] = out["beta"] if has_beta else np.nan
    out["se"] = out["se"] if has_se else np.nan

    fixed_csv = lrt_csv_path.with_name("fixed_effects.csv")
    if not fixed_csv.exists():
        print(f"[WARN] fixed_effects.csv not found at {fixed_csv}; beta/se will be blank where missing.")
        return out

    fixed_df = pd.read_csv(fixed_csv)
    required_fixed = {"model_id", "kind", "term", "estimate", "std_error"}
    if not required_fixed.issubset(set(fixed_df.columns)):
        print(
            "[WARN] fixed_effects.csv missing required columns for beta/se backfill: "
            f"{sorted(required_fixed - set(fixed_df.columns))}. beta/se will be blank where missing."
        )
        return out

    fixed_key = fixed_df.loc[fixed_df["kind"] == "full", ["model_id", "term", "estimate", "std_error"]].copy()
    fixed_key["merge_key"] = fixed_key["model_id"].astype(str) + "||" + fixed_key["term"].astype(str)

    out["merge_key"] = out["model_id"].astype(str) + "||" + out["predictor"].astype(str)
    out = out.merge(
        fixed_key[["merge_key", "estimate", "std_error"]],
        on="merge_key",
        how="left",
        suffixes=("", "_from_fixed"),
    )

    out["beta"] = out["beta"].where(out["beta"].notna(), out["estimate"])
    out["se"] = out["se"].where(out["se"].notna(), out["std_error"])
    out = out.drop(columns=["merge_key", "estimate", "std_error"])
    return out


@dataclass(frozen=True)
class TableFigure:
    fig: plt.Figure
    table: object
    group_bounds: list[tuple[int, int]]


def _build_table_figure(
    df_all: pd.DataFrame,
    *,
    title: str,
    font_size: int,
    group_border_lw: float,
    cell_lw: float,
) -> TableFigure:
    df_plot = df_all.copy()
    family_order = {"erp": 0, "alpha": 1, "beta": 2}
    window_order = {"tw1": 0, "tw2": 1}
    roi_order = {"anterior": 0, "posterior": 1}
    outcome_order = {"self_duration": 0, "duration": 0, "latency": 1}

    df_plot["family_rank"] = df_plot["family"].map(family_order).fillna(999).astype(int)
    df_plot["window_rank"] = df_plot["window"].map(window_order).fillna(999).astype(int)
    df_plot["roi_rank"] = df_plot["roi"].map(roi_order).fillna(999).astype(int)
    df_plot["outcome_rank"] = df_plot["outcome"].map(outcome_order).fillna(999).astype(int)
    df_plot["predictor_group"] = [_family_label(family) for family in df_plot["family"]]

    def _outcome_label(value: str) -> str:
        if value in ("self_duration", "duration"):
            return "Duration"
        if value == "latency":
            return "Latency"
        return str(value)

    col_labels = ["Predictor", "TW", "ROI", "β", "SE", "χ²", "FDR p", "Sig."]
    cell_text: list[list[str]] = []
    row_types: list[str] = []  # "section" | "data"
    row_group_keys: list[str | None] = []

    outcomes = (
        df_plot[["outcome", "outcome_rank"]]
        .drop_duplicates()
        .sort_values(by=["outcome_rank", "outcome"], kind="mergesort")
    )
    for outcome in outcomes["outcome"].tolist():
        block = df_plot[df_plot["outcome"] == outcome].copy()
        if block.empty:
            continue

        block = block.sort_values(by=["family_rank", "window_rank", "roi_rank"], kind="mergesort").reset_index(
            drop=True
        )
        block["fdr_p"] = _fdr_bh(block["lrt_p"].to_numpy(dtype=float))
        block["sig"] = [_stars(v) for v in block["fdr_p"].to_numpy(dtype=float)]

        cell_text.append([_outcome_label(str(outcome)), "", "", "", "", "", "", ""])
        row_types.append("section")
        row_group_keys.append(None)

        predictor_display: list[str] = []
        last_group: str | None = None
        for group in block["predictor_group"]:
            if last_group == group:
                predictor_display.append("")
            else:
                predictor_display.append(str(group))
            last_group = str(group)

        for i in range(len(block)):
            group_value = str(block.loc[i, "predictor_group"])
            cell_text.append(
                [
                    predictor_display[i],
                    _window_label(block.loc[i, "window"]),
                    _roi_label(block.loc[i, "roi"]),
                    _format_beta_se(float(block.loc[i, "beta"])),
                    _format_beta_se(float(block.loc[i, "se"])),
                    _format_chisq(float(block.loc[i, "lrt_chisq"])),
                    _format_p(float(block.loc[i, "fdr_p"])),
                    str(block.loc[i, "sig"]),
                ]
            )
            row_types.append("data")
            row_group_keys.append(f"{outcome}::{group_value}")

    n_rows = len(cell_text)
    fig_height_in = max(2.0, 0.17 * (n_rows + 1) + 0.25)
    fig, ax = plt.subplots(figsize=(7.0, fig_height_in))
    ax.axis("off")

    # Remove default subplot padding so the table can fill the canvas.
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    pad_x = 0.0
    pad_y = 0.0
    title_height = 0.05  # axes fraction reserved for the title line
    ax.text(
        0.5,
        1.0 - pad_y,
        title,
        ha="center",
        va="top",
        fontsize=font_size + 1,
        transform=ax.transAxes,
    )

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        bbox=[
            pad_x,
            pad_y,
            1.0 - 2.0 * pad_x,
            1.0 - title_height - 2.0 * pad_y,
        ],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    # Column widths (sum to 1.0 of axes)
    col_widths = [0.18, 0.08, 0.12, 0.12, 0.12, 0.12, 0.18, 0.08]
    for (row, col), cell in table.get_celld().items():
        cell.PAD = 0.015
        if col < len(col_widths):
            cell.set_width(col_widths[col])

        cell.set_edgecolor("#cfcfcf")
        cell.set_linewidth(cell_lw)

        if row == 0:
            cell.set_facecolor("#f2f2f2")
            cell.get_text().set_weight("bold")
            cell.PAD = 0.02

        if col == 0:  # predictor column left-aligned
            cell.get_text().set_ha("left")

        if row > 0:
            row_type = row_types[row - 1]
            if row_type == "section":
                cell.set_facecolor("#e6ecf5")
                if col == 0:
                    cell.get_text().set_weight("bold")
                else:
                    cell.get_text().set_text("")
                cell.set_linewidth(max(cell_lw, 1.2))
                cell.set_edgecolor("#5f5f5f")

    # Compute predictor-group row bounds from data rows (+1 in matplotlib table for header row)
    group_bounds: list[tuple[int, int]] = []
    group_start: int | None = None
    for idx, key in enumerate(row_group_keys):
        if key is None:
            if group_start is not None:
                group_bounds.append((group_start, idx - 1))
                group_start = None
            continue
        if group_start is None:
            group_start = idx
            continue
        prev_key = row_group_keys[idx - 1]
        if prev_key != key:
            group_bounds.append((group_start, idx - 1))
            group_start = idx
    if group_start is not None:
        group_bounds.append((group_start, len(row_group_keys) - 1))

    # Draw bold rectangles per predictor group after layout is computed.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    n_cols = len(col_labels)
    for start, end in group_bounds:
        table_rows = range(start + 1, end + 2)  # +1 for header; inclusive end
        bboxes = [
            table[row, col].get_window_extent(renderer)
            for row in table_rows
            for col in range(n_cols)
        ]
        if not bboxes:
            continue
        bbox = Bbox.union(bboxes)
        (x0, y0) = ax.transAxes.inverted().transform((bbox.x0, bbox.y0))
        (x1, y1) = ax.transAxes.inverted().transform((bbox.x1, bbox.y1))
        rect = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            lw=group_border_lw,
            edgecolor="black",
            transform=ax.transAxes,
            zorder=10,
        )
        ax.add_patch(rect)

    return TableFigure(fig=fig, table=table, group_bounds=group_bounds)


def _save_figure_tight(
    fig: plt.Figure,
    save_basepath: Path,
    *,
    profile_name: str,
    formats: tuple[str, ...],
    pad_inches: float,
) -> list[Path]:
    apply_figure_profile(fig, profile_name)
    profile = FIGURE_PROFILES[profile_name]

    basepath = Path(save_basepath)
    basepath.parent.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for fmt in formats:
        fmt_norm = fmt.lower().strip().lstrip(".")
        if fmt_norm in ("tif", "tiff"):
            out_path = basepath.with_suffix(DEFAULT_TIFF_EXTENSION)
            kwargs = dict(DEFAULT_TIFF_KWARGS)
            kwargs["pad_inches"] = pad_inches
            fig.savefig(out_path, format="tiff", **kwargs)
            written.append(out_path)
        elif fmt_norm == "eps":
            out_path = basepath.with_suffix(".eps")
            kwargs = dict(DEFAULT_EPS_KWARGS)
            kwargs["pad_inches"] = pad_inches
            fig.savefig(out_path, format="eps", **kwargs)
            written.append(out_path)
        elif fmt_norm == "png":
            out_path = basepath.with_suffix(".png")
            fig.savefig(out_path, format="png", dpi=profile.dpi, bbox_inches="tight", pad_inches=pad_inches)
            written.append(out_path)
        else:
            raise ValueError(f"Unsupported export format: {fmt!r}")

    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lrt-csv",
        type=Path,
        default=Path("/Volumes/work-4T/turn-taking-testing/mixed_effect/lmm/tables/lrt_comparisons.csv"),
        help="Path to lrt_comparisons.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "dev" / "temp" / "lrt_tables",
        help="Output directory for figure assets.",
    )
    parser.add_argument(
        "--out-stem",
        type=str,
        default="lrt_comparisons",
        help="Base filename stem (without extension).",
    )
    parser.add_argument(
        "--profile",
        choices=["jneuro_1col", "jneuro_2col", "screen"],
        default="jneuro_2col",
        help="Export sizing profile.",
    )
    parser.add_argument("--font-size", type=int, default=10)
    parser.add_argument("--group-border-lw", type=float, default=2.2)
    parser.add_argument("--cell-lw", type=float, default=0.6)
    parser.add_argument("--pad-inches", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    lrt_csv: Path = args.lrt_csv
    if not lrt_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {lrt_csv}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(lrt_csv)
    _require_columns(df, REQUIRED_COLUMNS)
    df = _ensure_beta_se_columns(df, lrt_csv_path=lrt_csv)

    apply_style(args.profile)
    if df.empty:
        raise ValueError("No rows available in LRT comparisons CSV.")

    fig_out = _build_table_figure(
        df,
        title="LRT comparisons",
        font_size=int(args.font_size),
        group_border_lw=float(args.group_border_lw),
        cell_lw=float(args.cell_lw),
    )

    written_all = _save_figure_tight(
        fig_out.fig,
        out_dir / args.out_stem,
        profile_name=args.profile,
        formats=("tif", "eps", "png"),
        pad_inches=float(args.pad_inches),
    )
    plt.close(fig_out.fig)

    for path in written_all:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
