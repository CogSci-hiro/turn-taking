from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np

# Optional dependency: only needed for SVG compose.
from lxml import etree

import matplotlib.pyplot as plt
import mne


# =============================================================================
#                     ########################################
#                     #           CONFIG CONSTANTS           #
#                     ########################################
# =============================================================================
_DEFAULT_CMAP: str = "RdBu_r"
_DEFAULT_TOPO_FIGSIZE_IN: tuple[float, float] = (2.2, 2.2)
_DEFAULT_DPI: int = 300
_DEFAULT_PAD_INCHES: float = 0.0


_SVG_NS: str = "http://www.w3.org/2000/svg"
_XLINK_NS: str = "http://www.w3.org/1999/xlink"


def _rewrite_ids_inplace(root: etree._Element, *, prefix: str) -> None:
    """
    Prefix all SVG ids in `root` and update url(#...) / href references.

    Matplotlib/MNE SVGs rely on <defs> entries like clipPaths and gradients.
    When composing multiple snippet SVGs into one template, their ids collide
    (e.g., 'clipPath1'). This breaks clipping and makes topomaps look wrong.
    """
    id_map: dict[str, str] = {}

    # Collect ids
    for el in root.iter():
        el_id = el.get("id")
        if el_id:
            id_map[el_id] = f"{prefix}__{el_id}"

    if not id_map:
        return

    # Rewrite element ids
    for el in root.iter():
        el_id = el.get("id")
        if el_id and el_id in id_map:
            el.set("id", id_map[el_id])

    # Rewrite references inside attribute strings
    url_pat = re.compile(r"url\(#([^)]+)\)")

    def _rewrite_value(v: str) -> str:
        # url(#id)
        v2 = url_pat.sub(lambda m: f"url(#{id_map.get(m.group(1), m.group(1))})", v)
        # href="#id"
        if v2.startswith("#"):
            key = v2[1:]
            if key in id_map:
                return f"#{id_map[key]}"
        return v2

    for el in root.iter():
        # Generic attributes
        for k, v in list(el.attrib.items()):
            if isinstance(v, str):
                new_v = _rewrite_value(v)
                if new_v != v:
                    el.set(k, new_v)

        # xlink:href explicitly
        xhref_key = f"{{{_XLINK_NS}}}href"
        if xhref_key in el.attrib:
            el.set(xhref_key, _rewrite_value(el.attrib[xhref_key]))


def _snippet_children_with_defs(snippet_root: etree._Element, *, slot_id: str) -> list[etree._Element]:
    """
    Return snippet children INCLUDING <defs>, after rewriting ids uniquely.

    We keep <defs> because clipPath definitions are required for correct
    rendering of Matplotlib/MNE SVG output.
    """
    _rewrite_ids_inplace(snippet_root, prefix=slot_id)

    children: list[etree._Element] = []
    for child in snippet_root:
        tag = etree.QName(child).localname
        if tag in {"metadata", "title", "desc"}:
            continue
        children.append(child)
    return children

# =============================================================================
#                     ########################################
#                     #            DATA STRUCTURES           #
#                     ########################################
# =============================================================================
@dataclass(frozen=True)
class ClusterOverlay:
    """
    Visual overlay for a cluster on a topomap.

    Parameters
    ----------
    name
        Cluster label used in legends or IDs (e.g., "C1", "C2").
    ch_names
        Channel names belonging to this cluster.
    marker
        Matplotlib marker (e.g., "o", "s", "^", "P").
    markersize
        Marker size in points.
    markeredgecolor
        Marker edge color.
    markerfacecolor
        Marker fill color.
    markeredgewidth
        Edge width in points.

    Usage example
    -------------
        overlay = ClusterOverlay(
            name="C1",
            ch_names=["Fz", "FCz", "AFz"],
            marker="o",
            markersize=10.0,
            markeredgecolor="black",
            markerfacecolor="none",
            markeredgewidth=1.5,
        )
    """

    name: str
    ch_names: tuple[str, ...]
    marker: str = "o"
    markersize: float = 10.0
    markeredgecolor: str = "black"
    markerfacecolor: str = "none"
    markeredgewidth: float = 1.5


# =============================================================================
#                     ########################################
#                     #          TOPO COORD HELPERS          #
#                     ########################################
# =============================================================================
def _topomap_xy(info: mne.Info, picks: np.ndarray) -> np.ndarray:
    """
    Get 2D topomap coordinates for a set of channel picks.

    Notes
    -----
    MNE's public API does not currently expose a fully stable topomap XY helper,
    so this uses an internal function. We isolate it here so we can swap later
    if MNE provides a public alternative.

    Parameters
    ----------
    info
        MNE Info with montage set.
    picks
        Channel indices.

    Returns
    -------
    xy
        Array of shape (n_picks, 2) in topomap projection coordinates.

    Usage example
    -------------
        picks = np.array([info.ch_names.index("Fz")])
        xy = _topomap_xy(info, picks)
    """
    try:
        # MNE internal helper (commonly available)
        from mne.viz.topomap import _find_topomap_coords  # type: ignore

        return _find_topomap_coords(info, picks=picks)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not compute topomap coordinates. Ensure montage is set on info."
        ) from exc


def _ch_indices(info: mne.Info, ch_names: Sequence[str]) -> np.ndarray:
    """
    Map channel names to indices, raising a helpful error if any are missing.

    Usage example
    -------------
        picks = _ch_indices(info, ["Fz", "Pz"])
    """
    missing: list[str] = [ch for ch in ch_names if ch not in info.ch_names]
    if missing:
        raise ValueError(f"Cluster channels not found in info: {missing}")
    return np.array([info.ch_names.index(ch) for ch in ch_names], dtype=int)


# =============================================================================
#                     ########################################
#                     #            EXPORT FUNCTIONS          #
#                     ########################################
# =============================================================================
def export_topomap_svg(
    *,
    data: np.ndarray,
    info: mne.Info,
    out_svg: Path,
    vlim: tuple[float, float],
    cmap: str = _DEFAULT_CMAP,
    mask: np.ndarray | None = None,
    title: str | None = None,
    show_sensors: bool = False,
    contours: int = 6,
    fig_size_in: tuple[float, float] = (3.2, 3.2),  # bigger default
    dpi: int = _DEFAULT_DPI,
    title_fontsize: int = 16,
    marker_size: float = 9.0,
) -> None:
    """
    Export a single topomap as a stable-size SVG for template placement.

    Key choices:
    - No bbox_inches="tight" => stable canvas size (prevents drift/offset in template).
    - Axes occupy a fixed rectangle so the head is centered.
    - Title is drawn at a fixed position in figure coordinates.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError(f"`data` must be 1D, got shape={data.shape}")
    if len(data) != len(info.ch_names):
        raise ValueError(
            f"`data` length must match info.ch_names. Got {len(data)} vs {len(info.ch_names)}."
        )

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(info.ch_names),):
            raise ValueError(f"`mask` must have shape {(len(info.ch_names),)}, got {mask.shape}.")

    out_svg.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=fig_size_in, dpi=dpi)
    # Fixed axes box (left, bottom, width, height) in figure fraction
    # Tweaking these is the main “centering” control.
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.84])  # big and centered

    mask_params = None
    if mask is not None and np.any(mask):
        mask_params = dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=marker_size,
            linewidth=0,  # important: no “connecting” outline lines
        )

    mne.viz.plot_topomap(
        data,
        info,
        axes=ax,
        show=False,
        vlim=vlim,
        cmap=cmap,
        sensors=show_sensors,
        contours=contours,
        outlines="head",
        sphere=None,
        mask=mask,
        mask_params=mask_params,
    )

    ax.set_axis_off()

    if title is not None:
        # Title in figure coords (stable; not affected by bbox/tight)
        fig.text(0.5, 0.98, title, ha="center", va="top", fontsize=title_fontsize)

    # IMPORTANT: stable canvas, no tight bbox
    fig.savefig(out_svg, format="svg", transparent=True)
    plt.close(fig)


def export_colorbar_svg(
    *,
    out_svg: Path,
    vlim: tuple[float, float],
    cmap: str = _DEFAULT_CMAP,
    label: str = "t value",
    fig_size_in: tuple[float, float] = (0.8, 3.2),  # match topo height
    dpi: int = _DEFAULT_DPI,
    tick_fontsize: int = 12,
    label_fontsize: int = 13,
) -> None:
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=fig_size_in, dpi=dpi)
    ax = fig.add_axes([0.35, 0.08, 0.25, 0.84])  # a tall bar centered

    norm = plt.Normalize(vmin=vlim[0], vmax=vlim[1])
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    cbar = fig.colorbar(mappable, cax=ax)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.set_label(label, fontsize=label_fontsize)

    fig.savefig(out_svg, format="svg", transparent=True)
    plt.close(fig)


def export_cluster_legend_svg(
    *,
    out_svg: Path,
    overlays: Sequence[ClusterOverlay],
    title: str = "Clusters",
    fig_size_in: tuple[float, float] = (2.2, 0.6),
    dpi: int = _DEFAULT_DPI,
) -> None:
    """
    Export a small legend SVG describing cluster marker encodings.

    Usage example
    -------------
        export_cluster_legend_svg(
            out_svg=Path("parts/legend_clusters.svg"),
            overlays=overlays,
        )
    """
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=fig_size_in)
    ax.set_axis_off()

    handles = []
    labels = []
    for overlay in overlays:
        h = ax.scatter(
            [],
            [],
            marker=overlay.marker,
            s=overlay.markersize**2,
            facecolors=overlay.markerfacecolor,
            edgecolors=overlay.markeredgecolor,
            linewidths=overlay.markeredgewidth,
        )
        handles.append(h)
        labels.append(overlay.name)

    ax.legend(
        handles,
        labels,
        title=title,
        loc="center",
        frameon=False,
        ncol=min(len(labels), 4),
    )

    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.0, dpi=dpi)
    plt.close(fig)


def export_text_svg(
    *,
    out_svg: Path,
    lines: Sequence[str],
    fig_size_in: tuple[float, float] = (4.0, 0.5),
    fontsize: int = 11,
    align: str = "left",
) -> None:
    """
    Export a tiny SVG containing text lines (e.g., p-threshold notes).

    Usage example
    -------------
        export_text_svg(
            out_svg=Path("parts/ptext_duration.svg"),
            lines=[
                "Topo colors: unthresholded t-values (common scale)",
                "Markers: cluster-corrected p < 0.05",
            ],
        )
    """
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=fig_size_in)
    ax.set_axis_off()

    x = 0.0 if align == "left" else 0.5 if align == "center" else 1.0
    ha = align

    y = 1.0
    dy = 0.35
    for line in lines:
        ax.text(x, y, line, transform=ax.transAxes, fontsize=fontsize, ha=ha, va="top")
        y -= dy

    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


# =============================================================================
#                     ########################################
#                     #           SVG COMPOSITION            #
#                     ########################################
# =============================================================================
_SVG_NS: str = "http://www.w3.org/2000/svg"


def _load_svg_root(svg_path: Path) -> etree._Element:
    parser = etree.XMLParser(remove_comments=False, recover=True)
    return etree.parse(str(svg_path), parser=parser).getroot()


def _find_by_id(root: etree._Element, element_id: str) -> etree._Element:
    el = root.xpath(f'//*[@id="{element_id}"]')
    if not el:
        raise KeyError(f"Could not find element with id={element_id!r} in template SVG.")
    return el[0]


def _strip_outer_svg(snippet_root: etree._Element) -> list[etree._Element]:
    children: list[etree._Element] = []
    for child in snippet_root:
        tag = etree.QName(child).localname
        if tag in {"metadata", "title", "desc"}:
            continue
        children.append(child)
    return children


def compose_svg_from_template(
    *,
    template_svg: Path,
    slot_to_snippet: Mapping[str, Path],
    out_svg: Path,
) -> None:
    """
    Compose a final SVG by injecting snippet SVGs into <g id="slot_*"> anchors.

    Parameters
    ----------
    template_svg
        Full layout SVG containing placeholder groups with stable IDs.
    slot_to_snippet
        Mapping from slot id -> snippet svg path.
    out_svg
        Output composed SVG.

    Usage example
    -------------
        compose_svg_from_template(
            template_svg=Path("template.svg"),
            slot_to_snippet={
                "slot_dur_tw1": Path("parts/duration_tw1.svg"),
                "slot_colorbar": Path("parts/colorbar.svg"),
            },
            out_svg=Path("ERP-results.svg"),
        )
    """
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    template_root = _load_svg_root(template_svg)

    for slot_id, snippet_path in slot_to_snippet.items():
        slot_g = _find_by_id(template_root, slot_id)

        # Clear current placeholder children
        for child in list(slot_g):
            slot_g.remove(child)

        snippet_root = _load_svg_root(snippet_path)

        # IMPORTANT: keep <defs> and prefix ids uniquely per slot
        snippet_children = _snippet_children_with_defs(snippet_root, slot_id=slot_id)

        wrapper = etree.Element(f"{{{_SVG_NS}}}g")
        for child in snippet_children:
            wrapper.append(child)
        slot_g.append(wrapper)

    out_svg.write_text(
        etree.tostring(
            template_root,
            pretty_print=True,
            xml_declaration=True,
            encoding="UTF-8",
        ).decode("utf-8"),
        encoding="utf-8",
    )
