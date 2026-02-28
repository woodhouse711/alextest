"""
svg_composer.py — First-pass SVG terrain map renderer.

Converts projected terrain contours and a GPX route into a print-ready SVG at
poster dimensions.  Phase 1 focuses on correctness over visual refinement;
this module is expected to be heavily extended in Phase 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import svgwrite


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

    dwg.save()
    return str(out.resolve())


def add_title_block(
    svg_drawing: svgwrite.Drawing,
    track_name: str,
    date: str,
    distance_km: float,
    elevation_gain_m: float,
    position: str = "bottom-left",
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
