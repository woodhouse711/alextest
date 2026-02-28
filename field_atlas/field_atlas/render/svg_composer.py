"""
svg_composer.py — First-pass SVG terrain map renderer.

Converts projected terrain contours and a GPX route into a print-ready SVG at
poster dimensions.  Phase 1 focuses on correctness over visual refinement;
this module is expected to be heavily extended in Phase 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import svgwrite

if TYPE_CHECKING:
    import pyproj
    from field_atlas.enrichment.models import EnrichmentData


# ---------------------------------------------------------------------------
# Feature label style constants
# ---------------------------------------------------------------------------

_PEAK_COLOR   = "#3A3A3A"
_WATER_COLOR  = "#7BA7BC"
_OTHER_COLOR  = "#888888"
_FONT_MM_STD  = 2.12     # ≈ 6pt
_FONT_MM_SML  = 1.76     # ≈ 5pt
_CHAR_W_FACTOR = 0.58    # estimated rendered char width / font_size
_MARKER_GAP   = 1.0      # mm gap between marker right edge and label text
_WATER_TYPES  = frozenset({"water", "pond", "reservoir"})


def _feature_style(ftype: str) -> dict:
    """Return marker and text style attrs for a given feature type."""
    if ftype == "peak":
        return {"marker": "triangle", "color": _PEAK_COLOR, "font_mm": _FONT_MM_STD}
    if ftype in _WATER_TYPES:
        return {"marker": "circle", "color": _WATER_COLOR, "font_mm": _FONT_MM_STD, "italic": True}
    if ftype == "viewpoint":
        return {"marker": "diamond", "color": _PEAK_COLOR, "font_mm": _FONT_MM_STD}
    return {"marker": "none", "color": _OTHER_COLOR, "font_mm": _FONT_MM_SML}


def _add_feature_marker(
    g: svgwrite.container.Group,
    dwg: svgwrite.Drawing,
    ftype: str,
    sx: float,
    sy: float,
) -> float:
    """Draw the marker glyph at (sx, sy) and return the x offset for the label.

    (sx, sy) is the vertical centre of the marker.  Returns the horizontal
    distance from sx to where the label text should begin.
    """
    if ftype == "peak":
        # Solid upward-pointing triangle, 2 mm tall × 2 mm wide.
        pts = [(sx, sy - 1.0), (sx - 1.0, sy + 1.0), (sx + 1.0, sy + 1.0)]
        g.add(dwg.polygon(pts, fill=_PEAK_COLOR, stroke="none"))
        return 1.0 + _MARKER_GAP
    if ftype in _WATER_TYPES:
        # Filled circle, r = 0.75 mm.
        g.add(dwg.circle(center=(sx, sy), r=0.75, fill=_WATER_COLOR, stroke="none"))
        return 0.75 + _MARKER_GAP
    if ftype == "viewpoint":
        # Filled diamond, 0.9 mm half-diagonal.
        r = 0.9
        pts = [(sx, sy - r), (sx + r, sy), (sx, sy + r), (sx - r, sy)]
        g.add(dwg.polygon(pts, fill=_PEAK_COLOR, stroke="none"))
        return r + _MARKER_GAP
    return 0.0  # "none" — text only, no horizontal offset needed


def _render_feature_labels(
    dwg: svgwrite.Drawing,
    enrichment: EnrichmentData,
    transformer: pyproj.Transformer,
    proj_to_svg: Callable[[float, float], tuple[float, float]],
    map_x0: float,
    map_y0: float,
    map_x1: float,
    map_y1: float,
) -> None:
    """Project, collide-check, and draw feature markers and labels.

    All features from *enrichment* that fall within the visible map area are
    labelled.  Simple axis-aligned bounding-box collision avoidance shifts
    lower-priority labels down by 3 mm when an overlap is detected.
    """
    from field_atlas.enrichment.features import rank_features

    if enrichment.features is None:
        return

    # Rank by type priority (no route points available at render time).
    ranked = rank_features(enrichment.features, route_points=[], max_labels=8)

    # Project each feature and discard those outside the visible map area.
    candidates: list[dict] = []
    for feat in ranked:
        easting, northing = transformer.transform(feat.lat, feat.lng)
        sx, sy = proj_to_svg(easting, northing)
        if map_x0 <= sx <= map_x1 and map_y0 <= sy <= map_y1:
            candidates.append({"feat": feat, "sx": sx, "sy": sy})

    if not candidates:
        return

    styles = [_feature_style(c["feat"].feature_type) for c in candidates]

    # ------------------------------------------------------------------ #
    # Collision avoidance                                                   #
    # ------------------------------------------------------------------ #
    # Marker half-widths used for label-start x offset (mirrors _add_feature_marker).
    _MARKER_HALF_W = {"triangle": 1.0, "circle": 0.75, "diamond": 0.9, "none": 0.0}

    def _label_box(idx: int, y_off: float) -> tuple[float, float, float, float]:
        """Estimate (x0, y0, x1, y1) of the text label for item at *idx*."""
        sx = candidates[idx]["sx"]
        sy = candidates[idx]["sy"] + y_off
        st = styles[idx]
        x_off = _MARKER_HALF_W.get(st["marker"], 0.0) + _MARKER_GAP
        lx = sx + x_off
        ly = sy - st["font_mm"]
        w  = len(candidates[idx]["feat"].name) * st["font_mm"] * _CHAR_W_FACTOR
        return (lx, ly, lx + w, ly + st["font_mm"] * 1.3)

    def _overlaps(a: tuple, b: tuple) -> bool:
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    y_offsets = [0.0] * len(candidates)
    for i in range(1, len(candidates)):
        for j in range(i):
            if _overlaps(_label_box(i, y_offsets[i]), _label_box(j, y_offsets[j])):
                y_offsets[i] += 3.0
                break  # one adjustment per label; imperfect but good enough

    # ------------------------------------------------------------------ #
    # Render into a dedicated group                                         #
    # ------------------------------------------------------------------ #
    g = dwg.g(id="features")
    for idx, item in enumerate(candidates):
        feat = item["feat"]
        sx   = item["sx"]
        sy   = item["sy"] + y_offsets[idx]
        st   = styles[idx]

        x_off = _add_feature_marker(g, dwg, feat.feature_type, sx, sy)

        txt_attrs: dict = {
            "font_family": "Arial, Helvetica, sans-serif",
            "font_size":   st["font_mm"],
            "fill":        st["color"],
        }
        if st.get("italic"):
            txt_attrs["font_style"] = "italic"

        # Text baseline sits 0.4 mm below marker vertical centre — a neutral
        # position that reads clearly alongside all three marker shapes.
        g.add(dwg.text(feat.name, insert=(sx + x_off, sy + 0.4), **txt_attrs))

    dwg.add(g)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_mm(value: str | float) -> float:
    """Strip a trailing 'mm' suffix and return the numeric value."""
    return float(str(value).replace("mm", "").strip())


def _build_transform(
    bounds: dict,
    offset_x: float,
    offset_y: float,
    scale: float,
) -> Callable[[float, float], tuple[float, float]]:
    """Return a closure that maps projected coords (metres) → SVG coords (mm).

    The y-axis is flipped: geographic y increases upward, SVG y increases
    downward.

    Parameters
    ----------
    bounds:
        Projected bounding box with ``min_x``, ``max_x``, ``min_y``, ``max_y``
        in metres.
    offset_x, offset_y:
        Top-left corner of the map area in SVG mm coordinates.
    scale:
        Metres-to-mm scale factor (equal in x and y to preserve aspect ratio).
    """
    min_x = bounds["min_x"]
    max_y = bounds["max_y"]

    def proj_to_svg(x: float, y: float) -> tuple[float, float]:
        return (
            offset_x + (x - min_x) * scale,
            offset_y + (max_y - y) * scale,
        )

    return proj_to_svg


def _infer_interval(contours: list[dict]) -> float:
    """Estimate the contour interval from the spacing between levels.

    Uses the median of consecutive differences so that a single missing level
    does not corrupt the result.
    """
    if len(contours) < 2:
        return 20.0
    elevations = [c["elevation"] for c in contours]
    diffs = sorted(
        elevations[i + 1] - elevations[i] for i in range(len(elevations) - 1)
    )
    return diffs[len(diffs) // 2]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_terrain_svg(
    contours: list[dict],
    route_points: list[dict],
    bounds: dict,
    output_path: str,
    width_mm: float = 457.2,
    height_mm: float = 609.6,
    margin_mm: float = 25.4,
    enrichment: EnrichmentData | None = None,
    transformer: pyproj.Transformer | None = None,
) -> str:
    """Render contour lines and a hiking route as a print-ready SVG.

    The projected bounds are scaled uniformly to fill the printable area
    (canvas minus margins) and centred if the data and canvas aspect ratios
    differ.

    Parameters
    ----------
    contours:
        Output of :func:`~field_atlas.core.terrain_processor.generate_contours`.
        Each dict has ``elevation`` (float) and ``paths`` (list of [[x, y] …]).
    route_points:
        List of projected point dicts with ``'x'`` and ``'y'`` keys in metres
        (e.g. from :func:`~field_atlas.core.projection.project_points`).
    bounds:
        Projected bounding box ``{min_x, max_x, min_y, max_y}`` in metres,
        e.g. from :func:`~field_atlas.core.projection.project_bounds`.
    output_path:
        Destination file path; parent directories are created as needed.
    width_mm:
        Canvas width in millimetres (default 457.2 = 18 in).
    height_mm:
        Canvas height in millimetres (default 609.6 = 24 in).
    margin_mm:
        Uniform margin on all four sides in millimetres (default 25.4 = 1 in).

    Returns
    -------
    str
        Absolute path of the written SVG file.

    Raises
    ------
    ValueError
        If *bounds* has zero or negative width or height.
    """
    # ------------------------------------------------------------------
    # 1. Compute scale and centring offsets
    # ------------------------------------------------------------------
    printable_w = width_mm - 2.0 * margin_mm
    printable_h = height_mm - 2.0 * margin_mm

    proj_w = bounds["max_x"] - bounds["min_x"]
    proj_h = bounds["max_y"] - bounds["min_y"]
    if proj_w <= 0 or proj_h <= 0:
        raise ValueError("bounds must have positive width and height in both axes")

    # Uniform scale: fit the larger relative dimension, preserve aspect ratio.
    scale = min(printable_w / proj_w, printable_h / proj_h)

    map_w_mm = proj_w * scale
    map_h_mm = proj_h * scale

    # Centre within the printable area.
    offset_x = margin_mm + (printable_w - map_w_mm) / 2.0
    offset_y = margin_mm + (printable_h - map_h_mm) / 2.0

    proj_to_svg = _build_transform(bounds, offset_x, offset_y, scale)

    # ------------------------------------------------------------------
    # 2. Create SVG canvas
    # ------------------------------------------------------------------
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # viewBox in mm user-units (1 user unit = 1 mm) keeps all coordinates
    # simple and makes stroke widths directly interpretable in mm.
    dwg = svgwrite.Drawing(
        str(out),
        size=(f"{width_mm}mm", f"{height_mm}mm"),
        viewBox=f"0 0 {width_mm} {height_mm}",
    )

    # ------------------------------------------------------------------
    # 3. White background
    # ------------------------------------------------------------------
    dwg.add(dwg.rect(insert=(0, 0), size=(width_mm, height_mm), fill="white"))

    # ------------------------------------------------------------------
    # 4. Contour lines
    # ------------------------------------------------------------------
    interval_m = _infer_interval(contours)
    index_interval = interval_m * 5.0  # e.g. 100 m when interval is 20 m

    for contour in contours:
        elev = contour["elevation"]

        # An index contour is one whose elevation is a multiple of 5× the
        # base interval (within floating-point rounding tolerance).
        is_index = (
            index_interval > 0
            and abs(round(elev / index_interval) * index_interval - elev) < 0.1
        )

        stroke_color = "#999999" if is_index else "#cccccc"
        stroke_width = 0.5 if is_index else 0.3

        for path in contour["paths"]:
            if len(path) < 2:
                continue
            pts = [proj_to_svg(xy[0], xy[1]) for xy in path]
            dwg.add(dwg.polyline(
                pts,
                stroke=stroke_color,
                stroke_width=stroke_width,
                fill="none",
                stroke_linejoin="round",
            ))

    # ------------------------------------------------------------------
    # 4. Route polyline
    # ------------------------------------------------------------------
    if len(route_points) >= 2:
        route_pts = [proj_to_svg(pt["x"], pt["y"]) for pt in route_points]
        dwg.add(dwg.polyline(
            route_pts,
            stroke="#2E75B6",
            stroke_width=1.2,
            fill="none",
            stroke_linejoin="round",
            stroke_linecap="round",
        ))

        # Start marker (green) and end marker (red).
        r_mm = 2.0
        sx, sy = proj_to_svg(route_points[0]["x"], route_points[0]["y"])
        ex, ey = proj_to_svg(route_points[-1]["x"], route_points[-1]["y"])

        dwg.add(dwg.circle(
            center=(sx, sy),
            r=r_mm,
            fill="#27AE60",
            stroke="white",
            stroke_width=0.4,
        ))
        dwg.add(dwg.circle(
            center=(ex, ey),
            r=r_mm,
            fill="#E74C3C",
            stroke="white",
            stroke_width=0.4,
        ))

    # ------------------------------------------------------------------
    # 5. Feature labels
    # ------------------------------------------------------------------
    if enrichment is not None and transformer is not None:
        _render_feature_labels(
            dwg, enrichment, transformer, proj_to_svg,
            map_x0=offset_x,
            map_y0=offset_y,
            map_x1=offset_x + map_w_mm,
            map_y1=offset_y + map_h_mm,
        )

    dwg.save()
    return str(out.resolve())


def add_title_block(
    svg_drawing: svgwrite.Drawing,
    track_name: str,
    date: str,
    distance_km: float,
    elevation_gain_m: float,
    position: str = "bottom-left",
    enrichment: EnrichmentData | None = None,
) -> None:
    """Add a metadata text block to a svgwrite Drawing in the margin area.

    The block contains the track name (bold), date, distance, and elevation
    gain.  It is placed within the margin region so it does not overlap the
    map content.

    Parameters
    ----------
    svg_drawing:
        An :class:`svgwrite.Drawing` whose ``size`` attribute is set in
        millimetres (e.g. ``("457.2mm", "609.6mm")``).  The drawing is
        mutated in-place; the caller is responsible for calling
        ``svg_drawing.save()`` afterwards.
    track_name:
        Display name of the route.
    date:
        Formatted date string, e.g. ``"2024-07-14"``.
    distance_km:
        Total route distance in kilometres.
    elevation_gain_m:
        Cumulative elevation gain in metres.
    position:
        Anchor corner: ``"bottom-left"`` (default), ``"bottom-right"``,
        ``"top-left"``, or ``"top-right"``.
    """
    width_mm = _parse_mm(svg_drawing.attribs["width"])
    height_mm = _parse_mm(svg_drawing.attribs["height"])

    # Layout constants (mm, matching the default 25.4 mm margin).
    MARGIN = 25.4
    PAD_X = 3.0    # horizontal inset from the margin edge
    PAD_Y = 3.5    # vertical inset from the printable-area boundary
    LINE_H = 4.2   # baseline-to-baseline spacing for body lines (mm)
    TITLE_SIZE = 3.5   # mm ≈ 10 pt
    BODY_SIZE = 3.0    # mm ≈ 8.5 pt

    body_lines = [
        date,
        f"Distance: {distance_km:.1f} km",
        f"Elevation gain: {elevation_gain_m:.0f} m",
    ]

    # Vertical anchor: title baseline sits just inside the margin area.
    if "bottom" in position:
        # Place title so body lines sit within the bottom margin.
        title_y = height_mm - MARGIN + PAD_Y + TITLE_SIZE
    else:
        title_y = MARGIN - PAD_Y - LINE_H * len(body_lines)

    # Horizontal anchor and text alignment.
    if "right" in position:
        text_x = width_mm - MARGIN - PAD_X
        anchor = "end"
    else:
        text_x = MARGIN + PAD_X
        anchor = "start"

    common_attrs: dict = {
        "font_family": "Arial, Helvetica, sans-serif",
        "text_anchor": anchor,
        "fill": "#333333",
    }

    svg_drawing.add(svg_drawing.text(
        track_name,
        insert=(text_x, title_y),
        font_size=TITLE_SIZE,
        font_weight="bold",
        **common_attrs,
    ))

    for i, line in enumerate(body_lines):
        svg_drawing.add(svg_drawing.text(
            line,
            insert=(text_x, title_y + LINE_H * (i + 1)),
            font_size=BODY_SIZE,
            **common_attrs,
        ))
