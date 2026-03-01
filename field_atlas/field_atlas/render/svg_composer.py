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

# Contour tier style table: (stroke-color, stroke-width-mm)
# major = every 10th interval (e.g. 100 m at 10 m interval)
# index = every  5th interval (e.g.  50 m at 10 m interval)
# minor = every      interval (e.g.  10 m at 10 m interval)
_CONTOUR_TIERS: dict[str, tuple[str, float]] = {
    "major": ("#777777", 0.55),
    "index": ("#999999", 0.40),
    "minor": ("#C8C8C8", 0.25),
}


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


def _chaikin_smooth(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Apply one iteration of Chaikin's corner-cutting to an open polyline.

    Each segment (P0, P1) is replaced by two points at the 1/4 and 3/4 marks:
        Q = 3/4·P0 + 1/4·P1
        R = 1/4·P0 + 3/4·P1

    The first and last points are preserved so polyline endpoints stay fixed.
    Polylines with fewer than 3 points are returned unchanged.
    """
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        out.append((0.75 * ax + 0.25 * bx, 0.75 * ay + 0.25 * by))
        out.append((0.25 * ax + 0.75 * bx, 0.25 * ay + 0.75 * by))
    out.append(pts[-1])
    return out


def _catmull_rom_smooth(
    pts: list[tuple[float, float]], n_interp: int = 4
) -> list[tuple[float, float]]:
    """Catmull-Rom spline interpolation that adds ~4× intermediate points.

    Intended for sparse GPX routes (< 100 waypoints) to remove angular
    stepping artefacts.  Duplicate-endpoint padding keeps the path anchored
    at the original start and end positions.
    """
    if len(pts) < 4:
        return pts
    padded = [pts[0]] + list(pts) + [pts[-1]]
    result: list[tuple[float, float]] = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for j in range(n_interp):
            t = j / n_interp
            t2, t3 = t * t, t * t * t
            x = 0.5 * (
                2 * p1[0]
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                2 * p1[1]
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            result.append((x, y))
    result.append(pts[-1])
    return result




def _draw_wind_indicator(
    dwg: "svgwrite.Drawing",
    weather: "WeatherData",
    cx: float,
    cy: float,
) -> None:
    """Draw a wind direction compass inset in a <g id='wind-indicator'> group.

    The compass rose has a circle, four cardinal tick marks, and an "N" label.
    A single arrow from the centre points toward the direction the wind is
    *coming from* (meteorological convention).  A speed label sits below.

    Parameters
    ----------
    dwg     : SVGWrite Drawing (mm user-units)
    weather : WeatherData with wind_direction_dominant_deg / cardinal / speed
    cx, cy  : centre of the compass circle in mm (SVG coordinates)
    """
    import math as _math

    COMP_R   = 6.0          # compass circle radius → 12 mm diameter
    TICK_L   = 1.5          # cardinal tick length (extends outside circle)
    ARROW_L  = 5.0          # wind arrow length from centre
    HEAD_LEN = 1.4          # arrowhead depth
    HEAD_WID = 0.6          # arrowhead half-width at base

    PT4   = 4   * 0.3528    # 4 pt  → mm
    PT4_5 = 4.5 * 0.3528    # 4.5 pt → mm

    MUTED = "#BBBBBB"
    FAINT = "#AAAAAA"
    RULE  = "#DDDDDD"
    ARROW = "#999999"
    FONT  = "Arial, Helvetica, sans-serif"

    g = dwg.g(id="wind-indicator")

    # ------------------------------------------------------------------
    # 1. Compass rose: circle + four cardinal ticks + N label
    # ------------------------------------------------------------------
    g.add(dwg.circle(
        center=(cx, cy), r=COMP_R,
        fill="none", stroke=RULE, stroke_width=0.15,
    ))

    # Cardinal direction unit vectors (compass bearing → SVG dx/dy)
    # bearing 0=N, 90=E, 180=S, 270=W; SVG y increases downward
    cardinals = [(0, 0.0, -1.0), (90, 1.0, 0.0), (180, 0.0, 1.0), (270, -1.0, 0.0)]
    for _bearing, dx, dy in cardinals:
        x0, y0 = cx + dx * COMP_R,            cy + dy * COMP_R
        x1, y1 = cx + dx * (COMP_R + TICK_L), cy + dy * (COMP_R + TICK_L)
        g.add(dwg.line(start=(x0, y0), end=(x1, y1), stroke=RULE, stroke_width=0.15))

    # "N" label sits above the north tick; baseline is 0.5 mm above the tick end
    g.add(dwg.text(
        "N",
        insert=(cx, cy - COMP_R - TICK_L - 0.5),
        font_size=PT4, font_family=FONT, fill=MUTED, text_anchor="middle",
    ))

    # ------------------------------------------------------------------
    # 2. Wind arrow — FROM direction (meteorological convention)
    # ------------------------------------------------------------------
    bearing_rad = _math.radians(weather.wind_direction_dominant_deg)
    # compass → SVG screen: dx = sin(bearing), dy = -cos(bearing)
    adx = _math.sin(bearing_rad)
    ady = -_math.cos(bearing_rad)

    tip_x, tip_y = cx + adx * ARROW_L, cy + ady * ARROW_L

    # Shaft
    g.add(dwg.line(
        start=(cx, cy), end=(tip_x, tip_y),
        stroke=ARROW, stroke_width=0.4, stroke_linecap="round",
    ))

    # Filled triangle arrowhead at tip
    bx, by = tip_x - adx * HEAD_LEN, tip_y - ady * HEAD_LEN   # base centre
    px, py = -ady, adx                                          # perpendicular
    pt1 = (bx + px * HEAD_WID, by + py * HEAD_WID)
    pt2 = (bx - px * HEAD_WID, by - py * HEAD_WID)
    g.add(dwg.polygon([(tip_x, tip_y), pt1, pt2], fill=ARROW, stroke="none"))

    # ------------------------------------------------------------------
    # 3. Speed label centred below the south tick
    # ------------------------------------------------------------------
    speed = round(weather.wind_speed_max_kmh)
    label = f"{weather.wind_direction_cardinal} {speed} KM/H"
    label_y = cy + COMP_R + TICK_L + PT4_5 + 1.0
    g.add(dwg.text(
        label,
        insert=(cx, label_y),
        font_size=PT4_5, font_family=FONT, fill=FAINT, text_anchor="middle",
    ))

    dwg.add(g)


def _draw_sun_arc(
    dwg: "svgwrite.Drawing",
    solar: "SolarData",
    origin_x: float,
    origin_y: float,
    width_mm: float = 35.0,
    height_mm: float = 18.0,
) -> None:
    """Draw a sun-path arc inset in a <g id='sun-arc'> group.

    The diagram shows the sun's trajectory across the sky on the day of the
    hike.  East (sunrise) is on the left; West (sunset) on the right.
    Azimuth is mapped linearly across *width_mm*; altitude linearly across
    *height_mm*.  The bottom edge of the diagram is the horizon line.

    Only called when *solar* data is present.  The diagram is deliberately
    low-contrast — a data whisper rather than a shout.

    Parameters
    ----------
    dwg        : SVGWrite Drawing (mm user-units)
    solar      : SolarData instance with a populated sun_path
    origin_x   : left edge of the diagram in mm (SVG coordinates)
    origin_y   : top edge of the diagram in mm (SVG coordinates)
    width_mm   : total width of the diagram
    height_mm  : total height from baseline to peak arc
    """
    from field_atlas.enrichment.solar import sun_arc_points

    arc_pts = sun_arc_points(solar, num_points=48)
    if len(arc_pts) < 2:
        return

    baseline_y = origin_y + height_mm       # horizon = bottom of diagram

    az_vals  = [p["azimuth_deg"]  for p in arc_pts]
    alt_vals = [p["altitude_deg"] for p in arc_pts]
    az_min, az_max = min(az_vals), max(az_vals)
    alt_max = max(alt_vals)

    if az_max <= az_min or alt_max <= 0:
        return

    def _to_svg(az: float, alt: float) -> tuple[float, float]:
        x = origin_x + (az - az_min) / (az_max - az_min) * width_mm
        y = baseline_y - (alt / alt_max) * height_mm
        return x, y

    GOLD   = "#E8C86A"
    MUTED  = "#BBBBBB"
    FAINT  = "#AAAAAA"
    RULE   = "#DDDDDD"
    FONT   = "Arial, Helvetica, sans-serif"
    PT4    = 4   * 0.3528   # 4 pt → mm
    PT4_5  = 4.5 * 0.3528   # 4.5 pt → mm

    g = dwg.g(id="sun-arc")

    # ------------------------------------------------------------------
    # 1. Horizon baseline
    # ------------------------------------------------------------------
    g.add(dwg.line(
        start=(origin_x,            baseline_y),
        end  =(origin_x + width_mm, baseline_y),
        stroke=RULE,
        stroke_width=0.2,
    ))

    # ------------------------------------------------------------------
    # 2. Sun arc polyline
    # ------------------------------------------------------------------
    arc_coords = [_to_svg(p["azimuth_deg"], p["altitude_deg"]) for p in arc_pts]
    g.add(dwg.polyline(
        arc_coords,
        stroke=GOLD,
        stroke_width=0.3,
        fill="none",
        stroke_linejoin="round",
        stroke_linecap="round",
    ))

    # ------------------------------------------------------------------
    # 3. Solar noon — filled circle at the peak altitude point
    # ------------------------------------------------------------------
    peak = max(arc_pts, key=lambda p: p["altitude_deg"])
    px, py = _to_svg(peak["azimuth_deg"], peak["altitude_deg"])
    g.add(dwg.circle(center=(px, py), r=0.5, fill=GOLD, stroke="none"))

    # ------------------------------------------------------------------
    # 4. E / W orientation labels (just below each end of the baseline)
    # ------------------------------------------------------------------
    label_y = baseline_y + PT4 + 0.8   # small gap below the rule
    g.add(dwg.text(
        "E",
        insert=(origin_x, label_y),
        font_size=PT4, font_family=FONT, fill=MUTED, text_anchor="middle",
    ))
    g.add(dwg.text(
        "W",
        insert=(origin_x + width_mm, label_y),
        font_size=PT4, font_family=FONT, fill=MUTED, text_anchor="middle",
    ))

    # ------------------------------------------------------------------
    # 5. Sunrise — sunset time string, centred below the diagram
    # ------------------------------------------------------------------
    time_str = f"{solar.sunrise} — {solar.sunset}"
    time_y   = label_y + PT4_5 + 1.2
    g.add(dwg.text(
        time_str,
        insert=(origin_x + width_mm / 2.0, time_y),
        font_size=PT4_5, font_family=FONT, fill=FAINT, text_anchor="middle",
    ))

    dwg.add(g)


def _sample_hillshade(
    hillshade,       # np.ndarray shape (rows, cols), values in [0, 1]
    hs_transform,    # rasterio Affine: (col, row) → (lng, lat) in WGS84
    transformer,     # pyproj.Transformer: WGS84 (lat, lng) → UTM (E, N)
    utm_x: float,
    utm_y: float,
) -> float:
    """Return the hillshade value [0, 1] at the given UTM coordinate.

    Reverses the UTM projection and the affine raster transform to find the
    array index.  Assumes a north-up raster (no rotation: b = d = 0).
    Returns 1.0 on any error so rendering degrades gracefully.
    """
    import math
    try:
        # UTM (easting, northing) → WGS84 (lat, lng) via inverse projection.
        lat, lng = transformer.transform(utm_x, utm_y, direction="INVERSE")
        # Affine inverse (north-up: b = d = 0):
        #   x = a·col + c  →  col = (x − c) / a   where x = lng
        #   y = e·row + f  →  row = (y − f) / e   where y = lat, e < 0
        col = (lng - hs_transform.c) / hs_transform.a
        row = (lat - hs_transform.f) / hs_transform.e
        r = max(0, min(int(round(row)), hillshade.shape[0] - 1))
        c = max(0, min(int(round(col)), hillshade.shape[1] - 1))
        val = float(hillshade[r, c])
        return val if not math.isnan(val) else 1.0
    except Exception:
        return 1.0


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
    hillshade=None,
    hillshade_transform=None,
    track_name: str = "",
    date: str = "",
    distance_km: float = 0.0,
    elevation_gain_m: float = 0.0,
    centroid_lat: float = 0.0,
    centroid_lng: float = 0.0,
    duration_hours: float | None = None,
    bottom_margin_mm: float = 88.9,
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
    hillshade:
        Optional float32 array (same grid as the DEM, values in [0, 1]) from
        :func:`~field_atlas.core.terrain_processor.generate_hillshade`.  When
        supplied, contour opacity is modulated by local illumination: sunlit
        segments (hillshade > 0.6) are rendered at 0.7 opacity to produce a
        subtle 3-D depth cue.
    hillshade_transform:
        Rasterio ``Affine`` object paired with *hillshade*; maps ``(col, row)``
        pixel indices to WGS84 ``(lng, lat)``.  Required when *hillshade* is
        not ``None``.

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
    printable_h = height_mm - margin_mm - bottom_margin_mm

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
    dwg.add(dwg.rect(
        insert=(0, 0), size=(width_mm, height_mm), fill="white", stroke="none",
    ))

    # ------------------------------------------------------------------
    # 4. Contour lines — three-tier visual hierarchy
    # ------------------------------------------------------------------
    interval_m     = _infer_interval(contours)
    major_interval = interval_m * 10.0   # e.g. 100 m at 10 m interval
    index_interval = interval_m * 5.0    # e.g.  50 m at 10 m interval

    use_hillshade = (
        hillshade is not None
        and hillshade_transform is not None
        and transformer is not None
    )

    def _snap(elev: float, step: float) -> bool:
        """True if *elev* is an integer multiple of *step* (float-safe)."""
        return step > 0 and abs(round(elev / step) * step - elev) < 0.1

    g_contours = dwg.g(id="contours")

    for contour in contours:
        elev = contour["elevation"]

        # Classify: check major before index so a 100 m line isn't also
        # classified as index (both are multiples of 5× the base interval).
        if _snap(elev, major_interval):
            tier = "major"
        elif _snap(elev, index_interval):
            tier = "index"
        else:
            tier = "minor"

        stroke_color, stroke_width = _CONTOUR_TIERS[tier]

        for path in contour["paths"]:
            if len(path) < 2:
                continue

            # Convert to SVG coords then apply one pass of Chaikin smoothing
            # to soften the grid-derived jaggedness.
            pts = [proj_to_svg(xy[0], xy[1]) for xy in path]
            pts = _chaikin_smooth(pts)

            pl = dwg.polyline(
                pts,
                stroke=stroke_color,
                stroke_width=stroke_width,
                fill="none",
                stroke_linejoin="round",
            )

            # Hillshade colour: sample at the path midpoint (UTM coords).
            # Sunlit slopes (hs > 0.6) are overridden to ~220-gray (#DCDCDC)
            # so the contour network visibly recedes on bright faces.
            if use_hillshade:
                mid = path[len(path) // 2]
                hs = _sample_hillshade(
                    hillshade, hillshade_transform, transformer,
                    mid[0], mid[1],
                )
                if hs > 0.6:
                    pl["stroke"] = "#DCDCDC"

            g_contours.add(pl)

    dwg.add(g_contours)

    # ------------------------------------------------------------------
    # 5. Route polyline
    # ------------------------------------------------------------------
    import math as _math

    if len(route_points) >= 2:
        g_route = dwg.g(id="route")

        route_pts_proj = [(pt["x"], pt["y"]) for pt in route_points]
        route_pts = [proj_to_svg(x, y) for x, y in route_pts_proj]

        # Smooth sparse routes (< 100 waypoints) with Catmull-Rom spline
        # to remove the angular GPS-stepping artefacts.
        if len(route_pts) < 100:
            route_pts = _catmull_rom_smooth(route_pts)

        # Pass 1 — white casing: makes the route pop off dense contours.
        g_route.add(dwg.polyline(
            route_pts,
            stroke="#FFFFFF",
            stroke_width=1.5,
            fill="none",
            stroke_linejoin="round",
            stroke_linecap="round",
        ))
        # Pass 2 — blue route on top.
        g_route.add(dwg.polyline(
            route_pts,
            stroke="#2E75B6",
            stroke_width=0.9,
            fill="none",
            stroke_linejoin="round",
            stroke_linecap="round",
        ))

        # Loop detection: start and end within 50 m → single shared marker.
        p0, p1 = route_pts_proj[0], route_pts_proj[-1]
        is_loop = _math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 50.0

        sx, sy = proj_to_svg(route_pts_proj[0][0], route_pts_proj[0][1])

        # Start marker — filled forest-green circle, 2 mm diameter (r=1.0).
        g_route.add(dwg.circle(
            center=(sx, sy),
            r=1.0,
            fill="#3D8B37",
            stroke="white",
            stroke_width=0.3,
        ))

        if not is_loop:
            ex, ey = proj_to_svg(route_pts_proj[-1][0], route_pts_proj[-1][1])
            # End marker — surveyor's benchmark: open ring + centre dot.
            g_route.add(dwg.circle(    # outer ring, 2.5 mm diameter (r=1.25)
                center=(ex, ey),
                r=1.25,
                fill="none",
                stroke="#2E75B6",
                stroke_width=0.5,
            ))
            g_route.add(dwg.circle(    # centre dot, 0.8 mm diameter (r=0.4)
                center=(ex, ey),
                r=0.4,
                fill="#2E75B6",
                stroke="none",
            ))

        dwg.add(g_route)

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

    # ------------------------------------------------------------------
    # 6. Environmental insets — upper corners, only with enrichment data
    # ------------------------------------------------------------------

    # Sun arc — upper-right margin
    if enrichment is not None and enrichment.solar is not None:
        ARC_W, ARC_H = 35.0, 18.0
        arc_x = width_mm - margin_mm - 2.0 - ARC_W  # 2 mm pad from printable edge
        arc_y = 2.5                                   # 2.5 mm from top of page
        _draw_sun_arc(dwg, enrichment.solar, arc_x, arc_y, ARC_W, ARC_H)

    # Wind indicator — upper-left margin, mirroring sun arc
    if enrichment is not None and enrichment.weather is not None:
        COMP_R, TICK_L = 6.0, 1.5
        wind_cx = margin_mm + 2.0 + COMP_R + TICK_L  # = margin_mm + 9.5
        wind_cy = 12.5                                # same vertical band as sun arc
        _draw_wind_indicator(dwg, enrichment.weather, wind_cx, wind_cy)

    # ------------------------------------------------------------------
    # 7. Title block (location, date, stats, wordmark) in bottom margin
    # ------------------------------------------------------------------
    add_title_block(
        dwg,
        track_name=track_name,
        date=date,
        distance_km=distance_km,
        elevation_gain_m=elevation_gain_m,
        centroid_lat=centroid_lat,
        centroid_lng=centroid_lng,
        duration_hours=duration_hours,
        margin_mm=margin_mm,
        bottom_margin_mm=bottom_margin_mm,
        enrichment=enrichment,
    )

    dwg.save()
    return str(out.resolve())


def add_title_block(
    svg_drawing: svgwrite.Drawing,
    track_name: str,
    date: str,
    distance_km: float,
    elevation_gain_m: float,
    centroid_lat: float = 0.0,
    centroid_lng: float = 0.0,
    duration_hours: float | None = None,
    margin_mm: float = 25.4,
    bottom_margin_mm: float = 88.9,
    enrichment: EnrichmentData | None = None,
) -> None:
    """Render a centred typographic information block in the bottom margin.

    Rows (top → bottom within the margin)::

      1  Location name — uppercase, letterspaced (+80)           10 pt
      2  Date, long-form human format                             8 pt
      3  Hairline rule, 40 mm wide, centred, #CCCCCC, 0.3 pt
      4  Stats: distance · gain · duration · coordinates          7 pt
      5  Weather + solar summary  (omitted when no enrichment)   6.5 pt
      6  "FIELD ATLAS" wordmark, right-aligned                    5 pt

    Parameters
    ----------
    svg_drawing:
        A :class:`svgwrite.Drawing` in mm user-units.  Mutated in-place;
        the caller is responsible for ``svg_drawing.save()``.
    track_name:
        Route name used as Row-1 fallback when *enrichment* has no
        ``location_name``.
    date:
        ``"YYYY-MM-DD"`` string that is reformatted for display.
    distance_km, elevation_gain_m:
        Route statistics for the stats line.
    centroid_lat, centroid_lng:
        WGS-84 centroid of the route, displayed in the stats line.
    duration_hours:
        Optional total duration; rendered as ``"4H 32M"`` when provided.
    margin_mm:
        Uniform margin width (default 25.4 = 1 in).  Controls the block's
        vertical position and the wordmark's right-edge x position.
    enrichment:
        When provided, Row 5 (weather + solar) is included and
        ``enrichment.location_name`` is preferred for Row 1.
    """
    from datetime import datetime as _dt
    from field_atlas.enrichment.weather import format_weather_line

    width_mm  = _parse_mm(svg_drawing.attribs["width"])
    height_mm = _parse_mm(svg_drawing.attribs["height"])

    # --- Typography: 1 pt = 0.3528 mm (3× scale for legibility) ----------
    R1_MM = 10.59   # 30 pt — location name
    R2_MM =  8.46   # 24 pt — date
    R4_MM =  7.41   # 21 pt — stats line
    R5_MM =  6.87   # 19.5 pt — weather / solar
    R6_MM =  5.28   # 15 pt — wordmark

    MAIN_COLOR  = "#3A3A3A"
    SOFT_COLOR  = "#666666"
    RULE_COLOR  = "#CCCCCC"
    RULE_HALF_W = 60.0          # rule extends ±60 mm from centre (120 mm total)
    RULE_SW     = 0.318         # ≈ 0.9 pt stroke-width in mm
    FONT        = "Arial, Helvetica, sans-serif"

    cx           = width_mm / 2.0                  # horizontal centre of canvas
    margin_top_y = height_mm - bottom_margin_mm    # top edge of title block strip

    # Vertical positions — Row 1 baseline anchored from top of the strip.
    PAD_TOP = 7.5
    r1_y   = margin_top_y + PAD_TOP + R1_MM
    r2_y   = r1_y +  9.0     # Row 1 → Row 2:  9 mm
    rule_y = r2_y + 12.0     # Row 2 → rule:  12 mm
    r4_y   = rule_y + 12.0   # rule  → Row 4: 12 mm

    has_row5 = enrichment is not None and (
        enrichment.weather is not None or enrichment.solar is not None
    )
    if has_row5:
        r5_y = r4_y +  7.5   # Row 4 → Row 5:  7.5 mm
        r6_y = r5_y + 15.0   # Row 5 → Row 6: 15 mm
    else:
        r6_y = r4_y + 15.0   # no Row 5 — balanced gap to wordmark

    base: dict = {
        "font_family": FONT,
        "fill":        MAIN_COLOR,
        "text_anchor": "middle",
    }

    # ---- Row 1: Location name, uppercase, letterspaced ------------------
    if enrichment is not None and enrichment.location_name:
        location = enrichment.location_name.upper()
    else:
        location = track_name.upper()
    r1 = svg_drawing.text(location, insert=(cx, r1_y), font_size=R1_MM, **base)
    r1["letter-spacing"] = f"{0.08 * R1_MM:.3f}"
    svg_drawing.add(r1)

    # ---- Row 2: Date, long-form human format ----------------------------
    try:
        dt      = _dt.strptime(date, "%Y-%m-%d")
        fmt_date = f"{dt.strftime('%B')} {dt.day}, {dt.year}"
    except ValueError:
        fmt_date = date
    svg_drawing.add(svg_drawing.text(
        fmt_date, insert=(cx, r2_y), font_size=R2_MM, **base,
    ))

    # ---- Row 3: Hairline rule -------------------------------------------
    svg_drawing.add(svg_drawing.line(
        start=(cx - RULE_HALF_W, rule_y),
        end=(cx + RULE_HALF_W, rule_y),
        stroke=RULE_COLOR,
        stroke_width=RULE_SW,
    ))

    # ---- Row 4: Stats line ----------------------------------------------
    parts4: list[str] = [f"{distance_km:.1f} KM"]
    parts4.append(f"{elevation_gain_m:.0f}M GAIN")
    if duration_hours is not None:
        total_min = round(duration_hours * 60)
        h, m = divmod(total_min, 60)
        parts4.append(f"{h}H {m:02d}M")
    lat_str = f"{abs(centroid_lat):.2f}°{'N' if centroid_lat >= 0 else 'S'}"
    lng_str = f"{abs(centroid_lng):.2f}°{'E' if centroid_lng >= 0 else 'W'}"
    parts4.append(f"{lat_str}  {lng_str}")
    svg_drawing.add(svg_drawing.text(
        "  ·  ".join(parts4), insert=(cx, r4_y), font_size=R4_MM, **base,
    ))

    # ---- Row 5: Weather + solar (only when enrichment available) --------
    if has_row5:
        parts5: list[str] = []
        if enrichment.weather is not None:
            parts5.append(format_weather_line(enrichment.weather).upper())
        if enrichment.solar is not None:
            parts5.append(f"SUNRISE {enrichment.solar.sunrise}")
        svg_drawing.add(svg_drawing.text(
            "  ·  ".join(parts5),
            insert=(cx, r5_y),
            font_size=R5_MM,
            font_family=FONT,
            fill=SOFT_COLOR,
            text_anchor="middle",
        ))

    # ---- Row 6: "FIELD ATLAS" wordmark, right-aligned ------------------
    rx = width_mm - margin_mm   # right edge of the printable area
    r6 = svg_drawing.text(
        "FIELD ATLAS",
        insert=(rx, r6_y),
        font_size=R6_MM,
        font_family=FONT,
        fill=MAIN_COLOR,
        text_anchor="end",
    )
    r6["letter-spacing"] = f"{0.12 * R6_MM:.3f}"
    svg_drawing.add(r6)
