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
    from field_atlas.enrichment.osm_vectors import OSMVectors


# ---------------------------------------------------------------------------
# Feature label style constants
# ---------------------------------------------------------------------------

_PEAK_COLOR      = "#2A2218"
_WATER_COLOR     = "#6A97B0"
_WATER_FILL      = "#A8C8CC"   # muted teal fill for lake polygons
_WATER_STROKE    = "#6E9EA6"   # slightly darker teal outline
_OTHER_COLOR     = "#7A6E64"
_FONT_MM_STD  = 2.12     # ≈ 6pt
_FONT_MM_SML  = 1.76     # ≈ 5pt
_CHAR_W_FACTOR = 0.58    # estimated rendered char width / font_size
_MARKER_GAP   = 1.0      # mm gap between marker right edge and label text
_WATER_TYPES  = frozenset({"water", "pond", "reservoir"})

# Contour tier style table: (stroke-color, stroke-width-mm)
# major = every 10th interval (e.g. 100 m at 10 m interval)
# index = every  5th interval (e.g.  50 m at 10 m interval)
# minor = every      interval (e.g.  10 m at 10 m interval)
# Colors are pure grayscale percentages; no RGB variation.
_CONTOUR_TIERS: dict[str, tuple[str, float]] = {
    "major": ("#1A0F06", 0.60),   # deep warm brown — heavy/primary (wider too)
    "index": ("#5C3A1E", 0.40),   # medium warm sienna — clearly distinct from minor
    "minor": ("#8C7058", 0.20),   # warm tan — light
}

# Index contour label style
_CONTOUR_LABEL_FONT_MM = 1.41   # 4pt
_CONTOUR_LABEL_CHAR_W  = _CONTOUR_LABEL_FONT_MM * 0.58

# ---------------------------------------------------------------------------
# Graticule and neatline constants
# ---------------------------------------------------------------------------

_GRAT_STROKE        = "#6A5E54"   # warm brown-gray — solid hairline
_GRAT_STROKE_W      = 0.10        # mm
_GRAT_LABEL_FONT    = 2.12        # 6pt (+2pt from original 4pt)
_GRAT_LABEL_COLOR   = "#6A5E54"   # warm — matches grid lines
_GRAT_LABEL_FONT_F  = "Liberation Sans, Arial, Helvetica, sans-serif"
_GRAT_LABEL_GAP     = 1.0         # mm between label and outer border edge
_GRAT_N_SAMPLE      = 32          # intermediate points when projecting a grid line

_NL_TOTAL_W   = 5.0        # mm — total neatline border width
_NL_BAND_W    = 2.5        # mm — each of the two bands
_NL_BLACK     = "#1C1813"
_NL_WHITE     = "#FFFFFF"
_NL_STROKE    = 0.15       # mm — hairline outlines on inner/outer edges
_NL_BAND_OPY  = 0.62       # opacity applied to checker bands and corners


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
        # + crosshair: two orthogonal 1.6 mm strokes centred on (sx, sy).
        arm = 0.8
        lw  = 0.35
        g.add(dwg.line(start=(sx - arm, sy), end=(sx + arm, sy),
                       stroke=_PEAK_COLOR, stroke_width=lw, stroke_linecap="round"))
        g.add(dwg.line(start=(sx, sy - arm), end=(sx, sy + arm),
                       stroke=_PEAK_COLOR, stroke_width=lw, stroke_linecap="round"))
        return arm + _MARKER_GAP
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


def _render_peak_crosshairs(
    dwg: "svgwrite.Drawing",
    peak_markers: list,
    transformer: "pyproj.Transformer",
    proj_to_svg: "Callable[[float, float], tuple[float, float]]",
    map_x0: float,
    map_y0: float,
    map_x1: float,
    map_y1: float,
) -> None:
    """Render hairline + crosshairs at every detected peak location.

    Drawn as a separate unlabelled layer — a subtle cartographic texture that
    marks high points without competing with the feature-label layer.  Each
    mark is two hairline strokes (1.6 mm span, 0.14 mm weight) centred on the
    projected peak position.
    """
    if not peak_markers:
        return

    g = dwg.g(id="peak-crosshairs")
    arm = 0.80   # half-arm length (mm) — total span 1.6 mm
    lw  = 0.14   # hairline stroke weight

    for feat in peak_markers:
        easting, northing = transformer.transform(feat.lat, feat.lng)
        sx, sy = proj_to_svg(easting, northing)
        if not (map_x0 <= sx <= map_x1 and map_y0 <= sy <= map_y1):
            continue
        g.add(dwg.line(
            start=(sx - arm, sy), end=(sx + arm, sy),
            stroke=_PEAK_COLOR, stroke_width=lw, stroke_linecap="round",
        ))
        g.add(dwg.line(
            start=(sx, sy - arm), end=(sx, sy + arm),
            stroke=_PEAK_COLOR, stroke_width=lw, stroke_linecap="round",
        ))

    dwg.add(g)


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
            "font_family": "Liberation Sans, Arial, Helvetica, sans-serif",
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
# OSM vector style constants
# ---------------------------------------------------------------------------

_ROAD_MAJOR  = frozenset({"motorway", "trunk", "primary", "secondary"})
_ROAD_MINOR  = frozenset({"tertiary", "residential", "unclassified"})
_WATER_FILL  = "#BDD5E3"
_WATERWAY_COLOR = "#6A97B0"
_TRAIL_COLOR    = "#9B8A72"   # warm tan — reads on hillshade without competing with route

# ---------------------------------------------------------------------------
# Terrain hatch pattern styles
# Each entry: (pattern_id, spacing_mm, line_angle_deg, color, stroke_width_mm,
#              second_angle_deg_or_None)  ← second angle gives cross-hatch
# ---------------------------------------------------------------------------
_TERRAIN_PATTERNS: dict[str, tuple] = {
    # terrain_type: (pat_id, spacing, angle, color, sw, second_angle)
    "forest":    ("tp-forest",    2.5,  45, "#3E6633", 0.14, None),
    "wood":      ("tp-forest",    2.5,  45, "#3E6633", 0.14, None),
    "scrub":     ("tp-scrub",     4.0, 135, "#7A7A3A", 0.11, None),
    "heath":     ("tp-heath",     3.0,  45, "#7A5A3A", 0.10,  135),
    "grassland": ("tp-grassland", 3.5,   0, "#5A8A3A", 0.09, None),
    "glacier":   ("tp-glacier",   2.0,   0, "#7AB0E0", 0.14, None),
    "sand":      ("tp-sand",      5.0,  45, "#C8A86A", 0.09,  135),
    "scree":     ("tp-scree",     4.5,  45, "#9A8A7A", 0.10,  135),
    "wetland":   ("tp-wetland",   3.0,   0, "#6A9A9A", 0.11, None),
}
_TERRAIN_FILL_OPACITY = 0.30

# Protected area boundary styles: (dash_pattern, color, label_color)
_BOUNDARY_STYLES: dict[str, tuple] = {
    "national_park":  ("4,2",    "#2E6E2E", "#2E6E2E"),
    "wilderness":     ("6,2,1,2","#6E4E2E", "#6E4E2E"),
    "protected_area": ("3,3",    "#555555", "#555555"),
    "nature_reserve": ("4,2",    "#4A7A4A", "#4A7A4A"),
}
_BOUNDARY_SW = 0.40        # stroke-width mm
_BOUNDARY_LABEL_MM = 2.12  # 6pt


def _road_style(value: str) -> tuple[float, str] | None:
    """Return (stroke_width_mm, stroke_color) for a road, or None to skip."""
    if value in _ROAD_MAJOR:
        return (0.55, "#888888")   # medium gray, clearly legible
    if value in _ROAD_MINOR:
        return (0.35, "#AAAAAA")
    return None


def _project_osm_pt(
    lat: float,
    lng: float,
    transformer,
    proj_to_svg,
) -> tuple[float, float]:
    """Project a single WGS-84 point to SVG mm coordinates."""
    easting, northing = transformer.transform(lat, lng)
    return proj_to_svg(easting, northing)


def _project_osm_geom(
    geometry: list[list[float]],
    transformer,
    proj_to_svg,
) -> list[tuple[float, float]]:
    """Project a list of [[lat, lng], …] to SVG mm coordinates."""
    return [_project_osm_pt(pt[0], pt[1], transformer, proj_to_svg) for pt in geometry]


def _define_terrain_hatch_patterns(dwg: "svgwrite.Drawing") -> set[str]:
    """Register SVG hatch-line ``<pattern>`` defs for every terrain type.

    Returns the set of pattern IDs that were added so callers can use
    ``fill="url(#<id>)"`` on their polygon elements.
    """
    registered: set[str] = set()
    seen_ids: set[str] = set()

    for _ttype, (pat_id, spacing, angle, color, sw, second_angle) in _TERRAIN_PATTERNS.items():
        if pat_id in seen_ids:
            registered.add(pat_id)
            continue
        seen_ids.add(pat_id)

        def _add_hatch(pid: str, ang: int) -> None:
            pat = dwg.defs.add(dwg.pattern(
                id=pid,
                x=0, y=0,
                width=spacing, height=spacing,
                patternUnits="userSpaceOnUse",
                patternTransform=f"rotate({ang},0,0)",
            ))
            pat.add(dwg.line(
                start=(0, 0), end=(spacing, 0),
                stroke=color, stroke_width=sw,
            ))

        _add_hatch(pat_id, angle)
        registered.add(pat_id)

        if second_angle is not None:
            cross_id = pat_id + "-x"
            if cross_id not in seen_ids:
                seen_ids.add(cross_id)
                _add_hatch(cross_id, second_angle)
                registered.add(cross_id)

    return registered


def _render_terrain_fills(
    dwg: "svgwrite.Drawing",
    vectors: "OSMVectors",
    transformer,
    proj_to_svg,
    clip_id: str = "map-area",
) -> None:
    """Draw terrain-type hatch fills for landuse/natural area polygons.

    Each OSMTerrainArea is filled with a semi-transparent SVG pattern
    registered by :func:`_define_terrain_hatch_patterns`.  Layer sits
    above the hillshade raster and below water areas / roads.
    """
    if not vectors.terrain_areas:
        return

    g = dwg.g(id="terrain-fills", clip_path=f"url(#{clip_id})", opacity=_TERRAIN_FILL_OPACITY)

    for area in vectors.terrain_areas:
        spec = _TERRAIN_PATTERNS.get(area.terrain_type)
        if spec is None:
            continue
        pat_id, _spacing, _angle, _color, _sw, second_angle = spec

        pts = _project_osm_geom(area.geometry, transformer, proj_to_svg)
        if len(pts) < 3:
            continue

        # Primary hatch
        g.add(dwg.polygon(pts, fill=f"url(#{pat_id})", stroke="none"))

        # Cross-hatch (second angle)
        if second_angle is not None:
            cross_id = pat_id + "-x"
            g.add(dwg.polygon(pts, fill=f"url(#{cross_id})", stroke="none"))

    dwg.add(g)


def _render_osm_line_labels(
    dwg: "svgwrite.Drawing",
    vectors: "OSMVectors",
    transformer,
    proj_to_svg,
    map_x0: float,
    map_y0: float,
    map_x1: float,
    map_y1: float,
) -> None:
    """Place inline text labels for named waterways and named trails.

    Labels are positioned at the midpoint of each way (by node index),
    rotated to follow the local segment direction, and given a white
    knockout outline for readability against any background.
    """
    import math as _m

    FONT = "Liberation Sans, Arial, Helvetica, sans-serif"

    def _label_way(
        way: "OSMWay",
        font_size: float,
        color: str,
        italic: bool = False,
    ) -> None:
        if not way.name:
            return
        pts = _project_osm_geom(way.geometry, transformer, proj_to_svg)
        if len(pts) < 2:
            return

        # Midpoint by index
        mid = len(pts) // 2
        mx, my = pts[mid]

        # Skip labels whose midpoint falls outside the map canvas
        if not (map_x0 <= mx <= map_x1 and map_y0 <= my <= map_y1):
            return

        # Angle from surrounding segment
        i0 = max(0, mid - 1)
        i1 = min(len(pts) - 1, mid + 1)
        dx = pts[i1][0] - pts[i0][0]
        dy = pts[i1][1] - pts[i0][1]
        angle = _m.degrees(_m.atan2(dy, dx))
        # Keep text upright (never upside-down)
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        transform = f"rotate({angle:.1f},{mx:.3f},{my:.3f})"
        style_italic = "font-style:italic;" if italic else ""

        # White knockout outline — drawn first (behind)
        outline = dwg.text(
            way.name,
            insert=(mx, my - 0.6),
            font_size=font_size,
            font_family=FONT,
            fill="none",
            stroke="white",
            stroke_width=0.6,
            stroke_linejoin="round",
            text_anchor="middle",
            transform=transform,
        )
        if style_italic:
            outline["style"] = style_italic
        dwg.add(outline)

        # Colored fill on top
        label = dwg.text(
            way.name,
            insert=(mx, my - 0.6),
            font_size=font_size,
            font_family=FONT,
            fill=color,
            text_anchor="middle",
            transform=transform,
        )
        if style_italic:
            label["style"] = style_italic
        dwg.add(label)

    # Waterway labels — blue-gray, italic
    for way in vectors.waterways:
        _label_way(way, font_size=1.76, color="#5A8EA8", italic=True)

    # Named trail labels — warm tan, normal weight
    for way in vectors.trails:
        if way.name:
            _label_way(way, font_size=1.59, color="#7A6A56", italic=False)


def _render_protected_areas(
    dwg: "svgwrite.Drawing",
    vectors: "OSMVectors",
    transformer,
    proj_to_svg,
    clip_id: str = "map-area",
    map_x0: float = 0.0,
    map_y0: float = 0.0,
    map_x1: float = 9999.0,
    map_y1: float = 9999.0,
) -> None:
    """Draw dashed boundary lines and inline name labels for protected areas.

    Each boundary type (national park, wilderness, nature reserve) gets a
    distinct dash pattern and color.  The area name is placed at the centroid
    of whichever projected ring has the most points inside the map canvas.
    """
    import math as _m

    if not vectors.protected_areas:
        return

    FONT = "Liberation Sans, Arial, Helvetica, sans-serif"
    g = dwg.g(id="protected-areas", clip_path=f"url(#{clip_id})")

    for area in vectors.protected_areas:
        style = _BOUNDARY_STYLES.get(area.boundary_type, _BOUNDARY_STYLES["protected_area"])
        dash, stroke_color, label_color = style

        best_ring_pts: list[tuple[float, float]] = []

        for ring in area.rings:
            pts = _project_osm_geom(ring, transformer, proj_to_svg)
            if len(pts) < 2:
                continue

            poly = dwg.polyline(
                pts,
                fill="none",
                stroke=stroke_color,
                stroke_width=_BOUNDARY_SW,
                stroke_linejoin="round",
                stroke_linecap="round",
            )
            poly["stroke-dasharray"] = dash
            g.add(poly)

            # Track the ring with the most in-canvas points for label placement
            in_canvas = sum(
                1 for (px, py) in pts if map_x0 <= px <= map_x1 and map_y0 <= py <= map_y1
            )
            if in_canvas > len(best_ring_pts):
                best_ring_pts = pts

        # Place the area name at the centroid of the most-visible ring
        if area.name and best_ring_pts:
            vis_pts = [
                (px, py) for px, py in best_ring_pts
                if map_x0 <= px <= map_x1 and map_y0 <= py <= map_y1
            ]
            if vis_pts:
                cx = sum(p[0] for p in vis_pts) / len(vis_pts)
                cy = sum(p[1] for p in vis_pts) / len(vis_pts)

                # Uppercase, spaced label — white knockout then color
                for pass_fill, pass_stroke, pass_sw in [
                    ("none",        "white",       0.7),
                    (label_color,   "none",        0.0),
                ]:
                    t = dwg.text(
                        area.name.upper(),
                        insert=(cx, cy),
                        font_size=_BOUNDARY_LABEL_MM,
                        font_family=FONT,
                        fill=pass_fill,
                        text_anchor="middle",
                    )
                    if pass_sw > 0:
                        t["stroke"] = pass_stroke
                        t["stroke-width"] = str(pass_sw)
                        t["stroke-linejoin"] = "round"
                    t["letter-spacing"] = f"{0.10 * _BOUNDARY_LABEL_MM:.3f}"
                    dwg.add(t)

    dwg.add(g)


def _render_osm_lower_layers(
    dwg: "svgwrite.Drawing",
    vectors: "OSMVectors",
    transformer,
    proj_to_svg,
    clip_id: str = "map-area",
) -> None:
    """Draw water areas, waterways, and roads into the drawing.

    Layer order within this function (bottom to top):
      1. Water areas — filled pale-blue polygons
      2. Waterways  — blue-gray stroked lines
      3. Roads      — gray stroked lines

    All three groups are clipped to *clip_id* so they cannot bleed into the
    title margin, then added directly to *dwg* below the contour group.
    """
    clip = f"url(#{clip_id})"

    # 1. Water areas — filled lake/reservoir polygons ----------------------
    if vectors.water_areas:
        g_wa = dwg.g(id="water-areas", clip_path=clip)
        for ring in vectors.water_areas:
            pts = _project_osm_geom(ring, transformer, proj_to_svg)
            if len(pts) < 3:
                continue
            g_wa.add(dwg.polygon(
                pts,
                fill=_WATER_FILL,
                stroke=_WATER_STROKE,
                stroke_width=0.18,
                stroke_linejoin="round",
            ))
        dwg.add(g_wa)

    # 2. Waterways (blue-gray lines) ---------------------------------------
    g_ww = dwg.g(id="waterways", clip_path=clip)
    for way in vectors.waterways:
        pts = _project_osm_geom(way.geometry, transformer, proj_to_svg)
        if len(pts) < 2:
            continue
        sw = 1.1 if way.value == "river" else 0.45
        g_ww.add(dwg.polyline(
            pts,
            stroke=_WATERWAY_COLOR,
            stroke_width=sw,
            fill="none",
            stroke_linecap="round",
            stroke_linejoin="round",
        ))
    dwg.add(g_ww)

    # 3. Roads (gray lines) ------------------------------------------------
    g_roads = dwg.g(id="roads", clip_path=clip)
    for way in vectors.roads:
        style = _road_style(way.value)
        if style is None:
            continue
        sw, color = style
        pts = _project_osm_geom(way.geometry, transformer, proj_to_svg)
        if len(pts) < 2:
            continue
        g_roads.add(dwg.polyline(
            pts,
            stroke=color,
            stroke_width=sw,
            fill="none",
            stroke_linecap="round",
            stroke_linejoin="round",
        ))
    dwg.add(g_roads)


def _render_osm_trails(
    dwg: "svgwrite.Drawing",
    vectors: "OSMVectors",
    transformer,
    proj_to_svg,
    clip_id: str = "map-area",
) -> None:
    """Draw the broader trail network as dashed gray lines.

    Renders above contours but below the user's specific route, so the GPX
    route remains visually dominant.
    """
    g_trails = dwg.g(id="trail-network", clip_path=f"url(#{clip_id})")
    for way in vectors.trails:
        pts = _project_osm_geom(way.geometry, transformer, proj_to_svg)
        if len(pts) < 2:
            continue
        line = dwg.polyline(
            pts,
            stroke=_TRAIL_COLOR,
            stroke_width=0.28,
            fill="none",
            stroke_linecap="round",
            stroke_linejoin="round",
        )
        line["stroke-dasharray"] = "1.5,1"
        g_trails.add(line)
    dwg.add(g_trails)


def _render_wind_streamlines(
    dwg: "svgwrite.Drawing",
    streamlines: "list[list[tuple[float, float, float]]]",
    proj_to_svg: "Callable[[float, float], tuple[float, float]]",
    clip_id: str = "map-area",
) -> None:
    """Render pre-computed wind streamlines with a luminous, hint.fm-inspired aesthetic.

    Each streamline is drawn in two passes to simulate a glow effect without
    relying on SVG filter support (which cairosvg does not honour):

    1. **Aura pass** — a wide, soft-edged stroke at low opacity that mimics
       the diffuse glow around each wind particle in the hint.fm map.
    2. **Core pass** — a narrow tapered stroke at higher opacity that gives
       each line a bright, precise centre.

    Together the two passes read like long-exposure light streaks.  Direction
    is implied by the taper (thin tail → thick head); no arrowheads are drawn,
    keeping the layer atmospheric rather than navigational.

    Rendering parameters
    --------------------
    * Color: ``#4A9CC7`` — vibrant sky blue, luminous against cream/hillshade.
    * Core opacity:  0.20 (calm) → 0.72 (gusty).
    * Aura opacity:  0.05 (calm) → 0.18 (gusty)  [~25 % of core].
    * Core widths:   tail 0.035 mm → head 0.20–0.52 mm.
    * Aura widths:   tail 0.12  mm → head 0.65–1.70 mm  (3.25× core head).
    * Taper curve:   square-root ramp for a comet-like thick body.
    * Segments:      40 (core) / 20 (aura) per streamline.

    Layer placement: above graticule, below contours.
    """
    if not streamlines:
        return

    _N_CORE = 40
    _N_AURA = 20
    _COLOR  = "#4A9CC7"

    g = dwg.g(id="wind-streamlines", clip_path=f"url(#{clip_id})")

    # --- Global speed normalisation -----------------------------------------
    mean_speeds: list[float] = []
    for stream in streamlines:
        if len(stream) >= 3:
            spds = [spd for _, _, spd in stream]
            mean_speeds.append(sum(spds) / len(spds))
        else:
            mean_speeds.append(0.0)

    if not mean_speeds:
        return

    sorted_ms = sorted(mean_speeds)
    p90_speed = max(sorted_ms[int(len(sorted_ms) * 0.90)], 1e-6)

    # --- Draw each streamline -----------------------------------------------
    for stream, mean_spd in zip(streamlines, mean_speeds):
        if len(stream) < 4:
            continue

        speed_norm = min(1.0, mean_spd / p90_speed)

        # Core: visible even in calm zones; blazing in fast zones.
        opacity_core = round(0.20 + 0.52 * speed_norm, 3)
        # Aura: ~25 % of core opacity — creates the diffuse halo.
        opacity_aura = round(0.05 + 0.13 * speed_norm, 3)

        # Core widths — thin tail, rounded head.
        wc_tail = 0.035
        wc_head = 0.20 + 0.32 * speed_norm   # 0.20 (calm) → 0.52 mm (gusty)

        # Aura widths — ~3.25× the core head for a wide soft halo.
        wa_tail = 0.12
        wa_head = wc_head * 3.25

        n_pts   = len(stream)
        svg_pts = [proj_to_svg(float(x), float(y)) for x, y, _ in stream]

        sg = dwg.g(style="isolation:isolate")

        def _add_pass(n_segs: int, w_tail: float, w_head: float, opacity: float) -> None:
            pass_g = dwg.g(opacity=opacity)
            for seg_i in range(n_segs):
                i_start = int(round(seg_i       / n_segs * (n_pts - 1)))
                i_end   = int(round((seg_i + 1) / n_segs * (n_pts - 1)))
                if i_end <= i_start:
                    i_end = i_start + 1
                i_end = min(i_end, n_pts - 1)
                seg_svg = svg_pts[i_start : i_end + 1]
                if len(seg_svg) < 2:
                    continue
                t_mid = (seg_i + 0.5) / n_segs
                ramp  = t_mid ** 0.5          # square-root: fast build from tail
                w = w_tail + (w_head - w_tail) * ramp
                pass_g.add(dwg.polyline(
                    seg_svg,
                    stroke=_COLOR,
                    stroke_width=round(w, 4),
                    stroke_linecap="round",
                    stroke_linejoin="round",
                    fill="none",
                ))
            sg.add(pass_g)

        _add_pass(_N_AURA, wa_tail, wa_head, opacity_aura)   # soft halo first
        _add_pass(_N_CORE, wc_tail, wc_head, opacity_core)   # sharp core on top

        g.add(sg)

    dwg.add(g)


# ---------------------------------------------------------------------------
# Graticule and neatline helpers
# ---------------------------------------------------------------------------

def _compute_wgs84_bounds(
    transformer: "pyproj.Transformer",
    bounds_projected: dict,
) -> tuple[float, float, float, float]:
    """Recover WGS84 (lat/lng) bounding box from UTM projected bounds.

    Projects all four corners and returns the axis-aligned envelope.
    """
    corners = [
        (bounds_projected["min_x"], bounds_projected["min_y"]),
        (bounds_projected["max_x"], bounds_projected["min_y"]),
        (bounds_projected["max_x"], bounds_projected["max_y"]),
        (bounds_projected["min_x"], bounds_projected["max_y"]),
    ]
    wgs = [transformer.transform(x, y, direction="INVERSE") for x, y in corners]
    lats = [c[0] for c in wgs]
    lngs = [c[1] for c in wgs]
    return min(lats), max(lats), min(lngs), max(lngs)


def _graticule_interval(
    min_lat: float, max_lat: float,
    min_lng: float, max_lng: float,
) -> tuple[float, float, float]:
    """Auto-select grid interval targeting 3–6 lines on each axis.

    Returns ``(interval_deg, subdiv_deg, interval_min)`` where
    *interval_min* is the interval in minutes (float) for label formatting.
    *subdiv_deg* is 1/5 of the grid interval, used for neatline segments.
    """
    max_extent = max(max_lat - min_lat, max_lng - min_lng)
    if   max_extent > 0.5:  interval_min = 10.0
    elif max_extent > 0.1:  interval_min = 5.0
    elif max_extent > 0.05: interval_min = 2.0
    elif max_extent > 0.01: interval_min = 1.0
    else:                   interval_min = 0.5   # 30 seconds
    interval_deg = interval_min / 60.0
    subdiv_deg   = interval_deg / 5.0
    return interval_deg, subdiv_deg, interval_min


def _gen_values(lo: float, hi: float, step: float) -> list[float]:
    """Return regularly spaced values from the first multiple of *step* ≥ *lo*
    up through *hi*."""
    import math as _m
    start = _m.ceil(lo / step - 1e-9) * step
    vals: list[float] = []
    v = start
    while v <= hi + 1e-9:
        vals.append(v)
        v += step
    return vals


def _project_lat_line(
    transformer: "pyproj.Transformer",
    lat_val: float,
    min_lng: float,
    max_lng: float,
    proj_to_svg: "Callable[[float, float], tuple[float, float]]",
    n: int = _GRAT_N_SAMPLE,
) -> list[tuple[float, float]]:
    """Return SVG points tracing the parallel at *lat_val* across the map."""
    lngs = [min_lng + i * (max_lng - min_lng) / (n - 1) for i in range(n)]
    return [proj_to_svg(*transformer.transform(lat_val, lng)) for lng in lngs]


def _project_lng_line(
    transformer: "pyproj.Transformer",
    lng_val: float,
    min_lat: float,
    max_lat: float,
    proj_to_svg: "Callable[[float, float], tuple[float, float]]",
    n: int = _GRAT_N_SAMPLE,
) -> list[tuple[float, float]]:
    """Return SVG points tracing the meridian at *lng_val* across the map."""
    lats = [min_lat + i * (max_lat - min_lat) / (n - 1) for i in range(n)]
    return [proj_to_svg(*transformer.transform(lat, lng_val)) for lat in lats]


def _cross_horizontal(pts: list[tuple[float, float]], ty: float) -> float | None:
    """Return the x coordinate where polyline *pts* first crosses y = *ty*."""
    for k in range(len(pts) - 1):
        y0, y1 = pts[k][1], pts[k + 1][1]
        if (y0 - ty) * (y1 - ty) <= 0 and abs(y1 - y0) > 1e-9:
            t = (ty - y0) / (y1 - y0)
            return pts[k][0] + t * (pts[k + 1][0] - pts[k][0])
    return None


def _cross_vertical(pts: list[tuple[float, float]], tx: float) -> float | None:
    """Return the y coordinate where polyline *pts* first crosses x = *tx*."""
    for k in range(len(pts) - 1):
        x0, x1 = pts[k][0], pts[k + 1][0]
        if (x0 - tx) * (x1 - tx) <= 0 and abs(x1 - x0) > 1e-9:
            t = (tx - x0) / (x1 - x0)
            return pts[k][1] + t * (pts[k + 1][1] - pts[k][1])
    return None


def _fmt_coord(deg_decimal: float, axis: str, interval_min: float) -> str:
    """Format a decimal degree value as 42°27′N or 71°07′W.

    When *interval_min* < 1.0 (sub-minute precision) the format includes
    whole seconds: 42°27′30″N.
    """
    is_neg  = deg_decimal < 0
    abs_d   = abs(deg_decimal)
    degrees = int(abs_d)
    tot_min = (abs_d - degrees) * 60.0
    suffix  = ("N" if not is_neg else "S") if axis == "lat" else ("W" if is_neg else "E")

    if interval_min < 1.0:      # sub-minute: show whole seconds
        min_int = int(tot_min)
        secs    = int(round((tot_min - min_int) * 60.0))
        if secs == 60:
            min_int += 1; secs = 0
        return f'{degrees}\u00b0{min_int:02d}\u2032{secs:02d}\u2033{suffix}'
    else:
        min_r = int(round(tot_min))
        if min_r == 60:
            degrees += 1; min_r = 0
        return f'{degrees}\u00b0{min_r:02d}\u2032{suffix}'


def _render_graticule(
    dwg: "svgwrite.Drawing",
    transformer: "pyproj.Transformer",
    bounds_projected: dict,
    proj_to_svg: "Callable[[float, float], tuple[float, float]]",
    offset_x: float,
    offset_y: float,
    map_w_mm: float,
    map_h_mm: float,
    clip_id: str = "map-area",
) -> tuple[float, float, float, float, float, float, float]:
    """Render coordinate grid lines inside the map and rotated labels in the margins.

    Grid lines are drawn at every *subdivision* interval (1/5 of the primary
    grid spacing), which matches the neatline border tick positions.  Lines
    are continuous hairlines at 33% black.  Labels on the left and right
    margins are rotated 90° to read vertically along each side; top/bottom
    labels are horizontal.  Label precision is at the seconds level.

    Returns ``(min_lat, max_lat, min_lng, max_lng, interval_deg, subdiv_deg,
    interval_min)`` for use by :func:`_render_checkered_border`.
    """
    min_lat, max_lat, min_lng, max_lng = _compute_wgs84_bounds(
        transformer, bounds_projected
    )
    interval_deg, subdiv_deg, interval_min = _graticule_interval(
        min_lat, max_lat, min_lng, max_lng
    )

    # Grid lines AND labels use the subdivision interval so every neatline
    # tick position corresponds to a visible grid line.
    subdiv_min = interval_min / 5.0   # for seconds-level label formatting
    lat_vals = _gen_values(min_lat, max_lat, subdiv_deg)
    lng_vals = _gen_values(min_lng, max_lng, subdiv_deg)

    # ------------------------------------------------------------------
    # Grid lines — continuous hairlines at 50% opacity, clipped to map area
    # ------------------------------------------------------------------
    g = dwg.g(id="graticule", clip_path=f"url(#{clip_id})")
    g["opacity"] = "0.5"

    for lat_val in lat_vals:
        pts = _project_lat_line(transformer, lat_val, min_lng, max_lng, proj_to_svg)
        g.add(dwg.polyline(
            pts,
            stroke=_GRAT_STROKE,
            stroke_width=_GRAT_STROKE_W,
            fill="none",
            stroke_linecap="round",
        ))

    for lng_val in lng_vals:
        pts = _project_lng_line(transformer, lng_val, min_lat, max_lat, proj_to_svg)
        g.add(dwg.polyline(
            pts,
            stroke=_GRAT_STROKE,
            stroke_width=_GRAT_STROKE_W,
            fill="none",
            stroke_linecap="round",
        ))

    dwg.add(g)

    # ------------------------------------------------------------------
    # Margin labels — just outside the neatline border
    # Left / right labels are rotated 90° to read vertically along the side.
    # ------------------------------------------------------------------
    BW      = _NL_TOTAL_W
    GAP     = _GRAT_LABEL_GAP
    F       = _GRAT_LABEL_FONT       # font-size in mm
    right_x = offset_x + map_w_mm
    bot_y   = offset_y + map_h_mm

    g_lbl = dwg.g(id="graticule-labels")
    _txt = dict(
        font_family=_GRAT_LABEL_FONT_F,
        font_size=F,
        fill=_GRAT_LABEL_COLOR,
    )

    # Left / right edge: lat labels rotated −90° (reads upward, south→north)
    # The label is centred at cx (horizontally, away from the border) and at
    # the grid-line crossing (vertically).  cx is offset outward by half the
    # font height so the nearest edge of the text clears the border.
    for lat_val in lat_vals:
        pts = _project_lat_line(transformer, lat_val, min_lng, max_lng, proj_to_svg)
        lbl = _fmt_coord(lat_val, "lat", subdiv_min)

        y_left = _cross_vertical(pts, offset_x)
        if y_left is not None and offset_y <= y_left <= bot_y:
            cx = offset_x - BW - GAP - F * 0.5
            t = dwg.text(lbl, insert=(cx, y_left), text_anchor="middle", **_txt)
            t["dominant-baseline"] = "central"
            t["transform"] = f"rotate(-90,{cx:.3f},{y_left:.3f})"
            g_lbl.add(t)

        y_right = _cross_vertical(pts, right_x)
        if y_right is not None and offset_y <= y_right <= bot_y:
            cx = right_x + BW + GAP + F * 0.5
            t = dwg.text(lbl, insert=(cx, y_right), text_anchor="middle", **_txt)
            t["dominant-baseline"] = "central"
            t["transform"] = f"rotate(-90,{cx:.3f},{y_right:.3f})"
            g_lbl.add(t)

    # Top / bottom edge: lng labels, horizontal, centred on the grid line
    for lng_val in lng_vals:
        pts = _project_lng_line(transformer, lng_val, min_lat, max_lat, proj_to_svg)
        lbl = _fmt_coord(lng_val, "lng", subdiv_min)

        x_top = _cross_horizontal(pts, offset_y)
        if x_top is not None and offset_x <= x_top <= right_x:
            g_lbl.add(dwg.text(
                lbl,
                insert=(x_top, offset_y - BW - GAP),
                text_anchor="middle",
                **_txt,
            ))

        x_bot = _cross_horizontal(pts, bot_y)
        if x_bot is not None and offset_x <= x_bot <= right_x:
            g_lbl.add(dwg.text(
                lbl,
                insert=(x_bot, bot_y + BW + GAP + F),
                text_anchor="middle",
                **_txt,
            ))

    dwg.add(g_lbl)

    return min_lat, max_lat, min_lng, max_lng, interval_deg, subdiv_deg, interval_min


def _render_checkered_border(
    dwg: "svgwrite.Drawing",
    transformer: "pyproj.Transformer",
    proj_to_svg: "Callable[[float, float], tuple[float, float]]",
    min_lat: float,
    max_lat: float,
    min_lng: float,
    max_lng: float,
    subdiv_deg: float,
    offset_x: float,
    offset_y: float,
    map_w_mm: float,
    map_h_mm: float,
) -> None:
    """Render the USGS-style checkered neatline frame around the map area.

    The border is 3 mm wide, divided into two 1.5 mm alternating bands whose
    segment boundaries align with geographic coordinate subdivisions.  The
    outer band is inverted relative to the inner band (black↔white) to
    produce the classic interlocked checker appearance.  Corners are solid
    black.  Thin hairline rules outline the inner and outer edges for a
    crisp typographic finish.
    """
    BW  = _NL_TOTAL_W   # 3 mm
    HW  = _NL_BAND_W    # 1.5 mm per band
    B   = _NL_BLACK
    W   = _NL_WHITE
    NS  = _GRAT_N_SAMPLE

    right_x = offset_x + map_w_mm
    bot_y   = offset_y + map_h_mm

    # Generate subdivision coordinate values covering the map extent (with
    # one step of padding so partial segments at the edges are included).
    subdiv_lngs = _gen_values(min_lng - subdiv_deg, max_lng + subdiv_deg, subdiv_deg)
    subdiv_lats = _gen_values(min_lat - subdiv_deg, max_lat + subdiv_deg, subdiv_deg)

    # Find where each longitude subdivision meridian crosses the top/bottom edges.
    top_xs: list[float] = []
    bot_xs: list[float] = []
    for lng_val in subdiv_lngs:
        pts = _project_lng_line(transformer, lng_val, min_lat, max_lat, proj_to_svg, n=NS)
        x = _cross_horizontal(pts, offset_y)
        if x is not None and offset_x < x < right_x:
            top_xs.append(x)
        x = _cross_horizontal(pts, bot_y)
        if x is not None and offset_x < x < right_x:
            bot_xs.append(x)
    top_xs.sort()
    bot_xs.sort()

    # Find where each latitude subdivision parallel crosses the left/right edges.
    left_ys: list[float] = []
    right_ys: list[float] = []
    for lat_val in subdiv_lats:
        pts = _project_lat_line(transformer, lat_val, min_lng, max_lng, proj_to_svg, n=NS)
        y = _cross_vertical(pts, offset_x)
        if y is not None and offset_y < y < bot_y:
            left_ys.append(y)
        y = _cross_vertical(pts, right_x)
        if y is not None and offset_y < y < bot_y:
            right_ys.append(y)
    left_ys.sort()
    right_ys.sort()

    g = dwg.g(id="neatline")

    # ------------------------------------------------------------------ #
    # Draw one edge of the checkered border.                               #
    # ------------------------------------------------------------------ #
    # Segment coloring: outer band alternates B/W, inner band is always
    # the opposite of the outer (classic USGS interlocked checker).
    # Segments are indexed from 0; corners (index 0 and last) are always B.

    OPY = _NL_BAND_OPY   # 1/3 opacity for halftone effect on both bands

    def _hsegs(breaks, x0_full, x1_full, inner_y0, outer_y0):
        """Horizontal edge segments.  inner_y0 is the y-start of the inner band
        (adjacent to map content); outer_y0 the y-start of the outer band."""
        segs = [x0_full] + [x for x in breaks if x0_full < x < x1_full] + [x1_full]
        for i in range(len(segs) - 1):
            a, b = segs[i], segs[i + 1]
            c_o = B if i % 2 == 0 else W   # outer colour
            c_i = W if i % 2 == 0 else B   # inner colour
            r_o = dwg.rect(insert=(a, outer_y0), size=(b - a, HW), fill=c_o, stroke="none")
            r_i = dwg.rect(insert=(a, inner_y0), size=(b - a, HW), fill=c_i, stroke="none")
            r_o["fill-opacity"] = f"{OPY:.4f}"
            r_i["fill-opacity"] = f"{OPY:.4f}"
            g.add(r_o); g.add(r_i)

    def _vsegs(breaks, y0_full, y1_full, inner_x0, outer_x0):
        """Vertical edge segments.  inner_x0 is the x-start of the inner band."""
        segs = [y0_full] + [y for y in breaks if y0_full < y < y1_full] + [y1_full]
        for i in range(len(segs) - 1):
            a, b = segs[i], segs[i + 1]
            c_o = B if i % 2 == 0 else W
            c_i = W if i % 2 == 0 else B
            r_o = dwg.rect(insert=(outer_x0, a), size=(HW, b - a), fill=c_o, stroke="none")
            r_i = dwg.rect(insert=(inner_x0, a), size=(HW, b - a), fill=c_i, stroke="none")
            r_o["fill-opacity"] = f"{OPY:.4f}"
            r_i["fill-opacity"] = f"{OPY:.4f}"
            g.add(r_o); g.add(r_i)

    # Top: inner band  = (offset_y - HW) → offset_y
    #       outer band = (offset_y - BW) → (offset_y - HW)
    _hsegs(top_xs, offset_x, right_x, offset_y - HW, offset_y - BW)

    # Bottom: inner band = bot_y → (bot_y + HW)
    #          outer band = (bot_y + HW) → (bot_y + BW)
    _hsegs(bot_xs, offset_x, right_x, bot_y, bot_y + HW)

    # Left: inner band  = (offset_x - HW) → offset_x
    #        outer band = (offset_x - BW) → (offset_x - HW)
    _vsegs(left_ys, offset_y, bot_y, offset_x - HW, offset_x - BW)

    # Right: inner band = right_x → (right_x + HW)
    #         outer band = (right_x + HW) → (right_x + BW)
    _vsegs(right_ys, offset_y, bot_y, right_x, right_x + HW)

    # ------------------------------------------------------------------ #
    # Corner squares — same 25 % opacity as the checker bands             #
    # ------------------------------------------------------------------ #
    for _cx, _cy in [
        (offset_x - BW, offset_y - BW),   # top-left
        (right_x,       offset_y - BW),   # top-right
        (offset_x - BW, bot_y),           # bottom-left
        (right_x,       bot_y),           # bottom-right
    ]:
        _cr = dwg.rect(insert=(_cx, _cy), size=(BW, BW), fill=B, stroke="none")
        _cr["fill-opacity"] = f"{OPY:.4f}"
        g.add(_cr)

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


def _douglas_peucker(
    pts: list[tuple[float, float]],
    tolerance: float,
) -> list[tuple[float, float]]:
    """Iterative Ramer-Douglas-Peucker polyline simplification.

    Removes points that deviate less than *tolerance* from the chord between
    their retained neighbours.  Angular inflections (ridgeline V-shapes,
    valley bottoms) are preserved because they exceed the threshold; only
    pixel-staircase rasterisation noise is eliminated.

    Uses an explicit stack to avoid Python recursion limits on long paths.
    """
    if len(pts) < 3:
        return pts

    def _perp_dist(px: float, py: float,
                   ax: float, ay: float,
                   bx: float, by: float) -> float:
        dx, dy = bx - ax, by - ay
        d2 = dx * dx + dy * dy
        if d2 == 0.0:
            return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / d2))
        return ((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2) ** 0.5

    n = len(pts)
    keep = [False] * n
    keep[0] = keep[-1] = True

    stack = [(0, n - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        ax, ay = pts[first]
        bx, by = pts[last]
        max_d, max_i = 0.0, first
        for i in range(first + 1, last):
            d = _perp_dist(pts[i][0], pts[i][1], ax, ay, bx, by)
            if d > max_d:
                max_d, max_i = d, i
        if max_d > tolerance:
            keep[max_i] = True
            stack.append((first, max_i))
            stack.append((max_i, last))

    return [p for p, k in zip(pts, keep) if k]


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


# Catmull-Rom tangent scale: 0.3 gives gentle smoothing that follows GPS
# points closely and avoids overshoot on sharp bends.
_SPLINE_TENSION = 0.3


def _cr_bezier_cp(
    pts: list[tuple[float, float]], i: int, tension: float = _SPLINE_TENSION
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the two cubic Bézier control points for segment pts[i]→pts[i+1].

    Uses the Catmull-Rom ↔ cubic-Bézier conversion so adjacent segments share
    tangent directions and produce a G1-continuous (smooth-joining) spline.
    Boundary segments duplicate the nearest endpoint as a phantom neighbour.
    The *tension* parameter scales the tangent vectors (1.0 = standard
    Catmull-Rom; 0.3 = gentle curves that hug the GPS track closely).
    """
    n = len(pts)
    p0 = pts[max(0, i - 1)]
    p1 = pts[i]
    p2 = pts[i + 1]
    p3 = pts[min(n - 1, i + 2)]
    scale = tension / 6.0
    cp1 = (p1[0] + (p2[0] - p0[0]) * scale, p1[1] + (p2[1] - p0[1]) * scale)
    cp2 = (p2[0] - (p3[0] - p1[0]) * scale, p2[1] - (p3[1] - p1[1]) * scale)
    return cp1, cp2


def _catmull_rom_path(pts: list[tuple[float, float]], tension: float = _SPLINE_TENSION) -> str:
    """Build a single smooth SVG path string from a Catmull-Rom spline.

    Produces one cubic Bézier ``C`` command per segment. The *tension*
    parameter is forwarded to :func:`_cr_bezier_cp`.
    """
    if len(pts) < 2:
        return ""
    parts = [f"M {pts[0][0]:.4f},{pts[0][1]:.4f}"]
    for i in range(len(pts) - 1):
        (cp1x, cp1y), (cp2x, cp2y) = _cr_bezier_cp(pts, i, tension)
        p2 = pts[i + 1]
        parts.append(
            f"C {cp1x:.4f},{cp1y:.4f} {cp2x:.4f},{cp2y:.4f} {p2[0]:.4f},{p2[1]:.4f}"
        )
    return " ".join(parts)


def _resample_route_spline(
    svg_pts: list[tuple[float, float]],
    pt_norms: list[float],
    n_out: int = 600,
) -> tuple[list[tuple[float, float]], list[float]]:
    """Resample a route to *n_out* uniformly-spaced points via CubicSpline.

    Fits independent ``scipy.interpolate.CubicSpline`` curves to x, y, and
    the normalised speed values, each parameterised by cumulative arc length
    normalised to [0, 1].  Resampling at uniform arc-length intervals removes
    the GPS point-density variation (dense near stoplights, sparse on straight
    fire roads) so the final curve has a steady, river-like quality — no
    bunching or angular stepping.

    Returns
    -------
    (resampled_pts, resampled_norms)
        ``resampled_pts`` — list of *n_out* (x, y) tuples in SVG mm coords.
        ``resampled_norms`` — list of *n_out* speed values clamped to [0, 1].
    """
    import numpy as _np
    from scipy.interpolate import CubicSpline as _CubicSpline

    pts = _np.array(svg_pts, dtype=float)
    n = len(pts)

    if n < 4:
        return svg_pts, pt_norms

    # Arc-length parameterisation — each GPS point gets a t ∈ [0, 1].
    diffs = _np.diff(pts, axis=0)
    seg_lens = _np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = _np.concatenate([[0.0], _np.cumsum(seg_lens)])
    total = float(cumlen[-1])

    if total < 1e-6:
        return svg_pts, pt_norms

    t = cumlen / total

    # Deduplicate: CubicSpline requires strictly increasing t values.
    # Duplicate GPS coordinates (parked, GPS drift) must be collapsed.
    _keep = [0]
    for _k in range(1, n):
        if t[_k] > t[_keep[-1]] + 1e-9:
            _keep.append(_k)
    if len(_keep) < 4:
        return svg_pts, pt_norms
    t = t[_keep]
    pts = pts[_keep]
    svals = _np.array([pt_norms[_k] for _k in _keep], dtype=float)

    # Fit splines for x, y, and speed.
    cs_x = _CubicSpline(t, pts[:, 0])
    cs_y = _CubicSpline(t, pts[:, 1])
    cs_s = _CubicSpline(t, svals)

    # Uniform resample.
    t_out = _np.linspace(0.0, 1.0, n_out)
    x_out = cs_x(t_out)
    y_out = cs_y(t_out)
    s_out = _np.clip(cs_s(t_out), 0.0, 1.0)

    new_pts = [(float(x), float(y)) for x, y in zip(x_out, y_out)]
    new_norms = [float(s) for s in s_out]
    return new_pts, new_norms


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
    FONT  = "Liberation Sans, Arial, Helvetica, sans-serif"

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
    FONT   = "Liberation Sans, Arial, Helvetica, sans-serif"
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


def _draw_scale_bar(
    dwg: "svgwrite.Drawing",
    scale_mm_per_m: float,
    right_edge_x: float,
    bar_top_y: float,
) -> None:
    """Draw a dual-unit cartographic checkered scale bar, right-aligned.

    The bar length is chosen automatically so it prints between 35–85 mm wide
    and corresponds to a clean ground distance.  A one-segment extension to
    the left of zero provides fine-subdivision reading.  Metric labels appear
    below each main-segment tick; a secondary imperial row follows; a
    representative-fraction string concludes the assembly.

    Parameters
    ----------
    dwg :
        svgwrite.Drawing in mm user-units.
    scale_mm_per_m :
        Render scale: mm on paper per projected metre (``scale`` in
        ``render_terrain_svg``).
    right_edge_x :
        X coordinate (mm) of the neatline right outer edge; the bar is
        right-aligned here.
    bar_top_y :
        Y coordinate (mm) of the top edge of the bar rectangle.
    """
    # ── 1. Auto-select a clean bar length ─────────────────────────────────
    # Each entry: (total_ground_m, n_main_segments, metres_per_segment).
    # Tried in order; first whose paper width falls in [35, 85] mm wins.
    CANDIDATES: list[tuple[int, int, int]] = [
        (10_000, 4, 2500),
        ( 5_000, 5, 1000),
        ( 4_000, 4, 1000),
        ( 2_000, 4,  500),
        ( 1_000, 4,  250),
        (   500, 5,  100),
        (   400, 4,  100),
        (   300, 3,  100),
        (   250, 5,   50),
        (   200, 4,   50),
        (   100, 5,   20),
        (    50, 5,   10),
        (    20, 4,    5),
        (    10, 5,    2),
    ]
    chosen = next(
        (c for c in CANDIDATES if 35.0 <= c[0] * scale_mm_per_m <= 85.0),
        min(CANDIDATES, key=lambda c: abs(c[0] * scale_mm_per_m - 60.0)),
    )
    total_m, n_segs, seg_m = chosen
    bar_w  = total_m * scale_mm_per_m   # total main-bar width (mm)
    seg_w  = bar_w / n_segs             # one segment width (mm)

    # Extension: one segment wide, subdivided into 4 equal sub-segments.
    N_EXT   = 4
    ext_sub = seg_w / N_EXT

    # X coordinates — right-aligned.
    x_right = right_edge_x
    x_zero  = x_right - bar_w          # zero mark / start of main bar
    x_left  = x_zero  - seg_w          # left edge of extension

    # ── 2. Style constants ─────────────────────────────────────────────────
    C_DARK = "#333333"
    C_LITE = "#FFFFFF"
    C_LBL  = "#333333"
    C_IMP  = "#666666"
    C_RF   = "#666666"
    FONT   = "Liberation Sans, Arial, Helvetica, sans-serif"
    BAR_H  = 2.0    # mm — bar height
    SW     = 0.15   # mm — border stroke-width
    TICK_H = 1.5    # mm — full tick below bar
    F_MET  = 1.59   # mm — 4.5 pt  metric labels
    F_IMP  = 1.41   # mm — 4 pt    imperial labels
    F_RF   = 1.41   # mm — 4 pt    representative fraction

    g = dwg.g(id="scale-bar")

    # ── 3. Extension segments (left of zero) ──────────────────────────────
    # Colour rule (L→R): dark | white | dark | white
    # The segment immediately left of zero is white, contrasting with the
    # first dark main segment so the zero mark reads as a clear boundary.
    for i in range(N_EXT):
        fill = C_DARK if (i % 2 == 0) else C_LITE
        g.add(dwg.rect(
            insert=(x_left + i * ext_sub, bar_top_y),
            size=(ext_sub, BAR_H),
            fill=fill, stroke=C_DARK, stroke_width=SW,
        ))

    # ── 4. Main bar segments (right of zero) ──────────────────────────────
    # Colour rule (L→R): dark | white | dark | white …
    for i in range(n_segs):
        fill = C_DARK if (i % 2 == 0) else C_LITE
        g.add(dwg.rect(
            insert=(x_zero + i * seg_w, bar_top_y),
            size=(seg_w, BAR_H),
            fill=fill, stroke=C_DARK, stroke_width=SW,
        ))

    # ── 5. Tick marks ─────────────────────────────────────────────────────
    bar_bot = bar_top_y + BAR_H

    def _tick(x: float, full: bool = True) -> None:
        h = TICK_H if full else TICK_H * 0.55
        g.add(dwg.line(
            start=(x, bar_bot), end=(x, bar_bot + h),
            stroke=C_LBL, stroke_width=SW,
        ))

    # Extension: full tick at left edge; short ticks at inner sub-divisions.
    _tick(x_left, full=True)
    for i in range(1, N_EXT):
        _tick(x_left + i * ext_sub, full=False)

    # Zero mark and every main-segment boundary get full ticks.
    _tick(x_zero, full=True)
    for i in range(1, n_segs + 1):
        _tick(x_zero + i * seg_w, full=True)

    # ── 6. Metric distance labels ──────────────────────────────────────────
    lbl_met_y = bar_bot + TICK_H + F_MET + 0.3   # text baseline

    def _fmt_m(dist_m: int) -> str:
        if dist_m == 0:
            return "0"
        if dist_m >= 1000 and dist_m % 1000 == 0:
            return f"{dist_m // 1000} km"
        if dist_m >= 1000:
            return f"{dist_m / 1000:.1f} km"
        return f"{dist_m} m"

    def _met_lbl(x: float, dist_m: int) -> None:
        g.add(dwg.text(
            _fmt_m(dist_m), insert=(x, lbl_met_y),
            text_anchor="middle", font_size=F_MET,
            font_family=FONT, fill=C_LBL,
        ))

    _met_lbl(x_zero, 0)
    for i in range(1, n_segs + 1):
        _met_lbl(x_zero + i * seg_w, i * seg_m)

    # ── 7. Imperial labels (second row) ───────────────────────────────────
    MI_PER_M   = 0.000621371
    total_mi   = total_m * MI_PER_M
    lbl_imp_y  = lbl_met_y + F_MET + 1.3

    # Unicode fraction characters render cleanly in common sans-serif fonts.
    MILE_MARKS: list[tuple[float, str]] = [
        (0.125, "\u215b mi"),   # ⅛
        (0.250, "\u00bc mi"),   # ¼
        (0.500, "\u00bd mi"),   # ½
        (0.750, "\u00be mi"),   # ¾
        (1.0,   "1 mi"),
        (1.5,   "1\u00bd mi"),  # 1½
        (2.0,   "2 mi"),
        (5.0,   "5 mi"),
    ]

    def _imp_lbl(x: float, label: str) -> None:
        g.add(dwg.text(
            label, insert=(x, lbl_imp_y),
            text_anchor="middle", font_size=F_IMP,
            font_family=FONT, fill=C_IMP,
        ))

    _imp_lbl(x_zero, "0")
    for miles, frac_lbl in MILE_MARKS:
        if miles > total_mi * 1.01:
            break
        x_mi = x_zero + (miles / MI_PER_M) * scale_mm_per_m
        if x_mi <= x_right + 0.5:
            _imp_lbl(x_mi, frac_lbl)

    # ── 8. Representative fraction ─────────────────────────────────────────
    # 1 mm on paper == rf_val mm on the ground.
    rf_val = 1_000.0 / scale_mm_per_m
    STANDARD_RF = [
        1_000, 2_000, 2_500, 4_000, 5_000,
        10_000, 12_500, 15_000, 20_000, 24_000, 25_000,
        50_000, 100_000, 250_000,
    ]
    rf_std = min(STANDARD_RF, key=lambda s: abs(s - rf_val))
    # Use a standard value if within 20 %; otherwise round to nearest 500.
    if abs(rf_std - rf_val) / rf_val > 0.20:
        rf_std = max(500, round(rf_val / 500) * 500)

    rf_cx = (x_left + x_right) / 2.0
    rf_y  = lbl_imp_y + F_IMP + 1.8
    g.add(dwg.text(
        f"1:{rf_std:,}",
        insert=(rf_cx, rf_y),
        text_anchor="middle",
        font_size=F_RF,
        font_family=FONT,
        fill=C_RF,
    ))

    dwg.add(g)


def _draw_north_arrow(
    dwg: "svgwrite.Drawing",
    cx: float,
    cy: float,
    r_mm: float,
) -> None:
    """Draw a geometric surveying-style north arrow.

    A tall hollow triangle (outline only, no fill) with a centre stem and
    barb-wing extensions at the base — modelled on a classic orienteering /
    cadastral north arrow.  All elements are pure black strokes; the design
    reads clearly at small print sizes.

    Proportions (r_mm = distance from anchor to apex):
      apex            cy - r_mm
      triangle base   cy + r_mm * 0.40  (half-width ± r_mm * 0.40)
      wing tips       half-width ± r_mm * 0.52, y = triangle-base + r_mm * 0.12
      stem top        cy - r_mm * 0.70  (inside triangle)
      stem bottom     cy + r_mm * 1.05  (below wing tips)
    """
    h = r_mm                    # scale unit

    # Proportions measured from the reference image:
    #   triangle base at ~60 % of total height from apex (tall, narrow triangle)
    #   wing tips at ~70 % from apex, slightly wider than triangle corners
    #   stem exits triangle top at ~14 %, exits bottom at ~93 %
    # Total arrow height ≈ 1.9 × r_mm (apex at top, stem tip at bottom)
    apex_y     = cy - h            # north tip
    tri_base_y = cy + h * 0.20    # triangle sides end here (tall triangle)
    wing_y     = tri_base_y + h * 0.22   # wings flare slightly below corners
    stem_top_y = cy - h * 0.72    # stem starts inside triangle (below apex)
    stem_bot_y = cy + h * 0.88    # stem ends below wing tips

    tri_hw  = h * 0.46   # half-width at triangle base (apex angle ≈ 40°)
    wing_hw = h * 0.58   # wing tips just wider than triangle corners

    sw = h * 0.13   # stroke weight ≈ 7 % of total arrow height

    g = dwg.g(id="north-arrow")

    # Left outer path: apex → tri-base-left → wing-tip-left → stem-bottom
    g.add(dwg.path(
        d=(f"M {cx:.3f},{apex_y:.3f} "
           f"L {cx - tri_hw:.3f},{tri_base_y:.3f} "
           f"L {cx - wing_hw:.3f},{wing_y:.3f} "
           f"L {cx:.3f},{stem_bot_y:.3f}"),
        fill="none", stroke="#000000",
        stroke_width=sw, stroke_linejoin="miter", stroke_linecap="square",
    ))
    # Right outer path: apex → tri-base-right → wing-tip-right → stem-bottom
    g.add(dwg.path(
        d=(f"M {cx:.3f},{apex_y:.3f} "
           f"L {cx + tri_hw:.3f},{tri_base_y:.3f} "
           f"L {cx + wing_hw:.3f},{wing_y:.3f} "
           f"L {cx:.3f},{stem_bot_y:.3f}"),
        fill="none", stroke="#000000",
        stroke_width=sw, stroke_linejoin="miter", stroke_linecap="square",
    ))
    # Centre stem — vertical staff bisecting the whole form
    g.add(dwg.line(
        start=(cx, stem_top_y), end=(cx, stem_bot_y),
        stroke="#000000", stroke_width=sw,
        stroke_linecap="square",
    ))

    dwg.add(g)


def render_terrain_svg(
    contours: list[dict],
    route_points: list[dict],
    bounds: dict,
    output_path: str,
    width_mm: float = 457.2,
    height_mm: float = 609.6,
    margin_mm: float = 38.1,
    enrichment: EnrichmentData | None = None,
    transformer: pyproj.Transformer | None = None,
    hillshade=None,
    hillshade_transform=None,
    hypsometric=None,
    track_name: str = "",
    date: str = "",
    distance_km: float = 0.0,
    elevation_gain_m: float = 0.0,
    centroid_lat: float = 0.0,
    centroid_lng: float = 0.0,
    duration_hours: float | None = None,
    bottom_margin_mm: float = 76.2,
    osm_vectors: OSMVectors | None = None,
    segment_speeds: list[float] | None = None,
    route_palette: str = "coastal",
    route_width: float = 1.8,
    wind_streamlines: "list[list[tuple[float, float, float]]] | None" = None,
    peak_markers: "list | None" = None,
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
        Uniform margin on top and both sides in millimetres (default 38.1 = 1.5 in).
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
    hypsometric:
        Optional uint8 RGBA array of shape ``(rows, cols, 4)`` from
        :func:`~field_atlas.core.terrain_processor.generate_hypsometric_rgba`.
        Rendered as a colour-tint layer beneath the hillshade; high terrain
        takes on the palette colour while low terrain remains near-transparent.

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
        insert=(0, 0), size=(width_mm, height_mm), fill="#FBF7F0", stroke="none",
    ))

    # Parchment underlay — map area only, below hillshade and all vector layers.
    dwg.add(dwg.rect(
        insert=(offset_x, offset_y), size=(map_w_mm, map_h_mm),
        fill="#EDE3B8", stroke="none", opacity=0.50,
    ))

    # Define a clip path that constrains all map content to the map rectangle.
    # This prevents vector lines from bleeding into the title margin.
    from svgwrite.masking import ClipPath as _ClipPath
    _clip = dwg.defs.add(_ClipPath(id="map-area"))
    _clip.add(dwg.rect(insert=(offset_x, offset_y), size=(map_w_mm, map_h_mm)))

    # ------------------------------------------------------------------
    # 3b. Hypsometric tint — RGBA colour ramp keyed to elevation.
    #     Rendered below the hillshade so shadows remain sharp.
    #     Alpha channel in the array already encodes per-pixel opacity:
    #     low terrain is near-transparent; high terrain shows palette colour.
    # ------------------------------------------------------------------
    if hypsometric is not None and hillshade_transform is not None and transformer is not None:
        import io as _io2
        import base64 as _base64_2
        try:
            from PIL import Image as _PILImage2
            _hyp_img = _PILImage2.fromarray(hypsometric, mode="RGBA")
            _buf2 = _io2.BytesIO()
            _hyp_img.save(_buf2, format="PNG", optimize=False)
            _href2 = f"data:image/png;base64,{_base64_2.b64encode(_buf2.getvalue()).decode('ascii')}"

            _hyp_rows, _hyp_cols = hypsometric.shape[:2]
            _corners_pix2 = [(0, 0), (_hyp_cols, 0), (_hyp_cols, _hyp_rows), (0, _hyp_rows)]
            _corners_svg2 = []
            for _pc2, _pr2 in _corners_pix2:
                _lng2, _lat2 = hillshade_transform * (_pc2, _pr2)
                _e2, _n2 = transformer.transform(_lat2, _lng2)
                _corners_svg2.append(proj_to_svg(_e2, _n2))
            _hxs2 = [p[0] for p in _corners_svg2]
            _hys2 = [p[1] for p in _corners_svg2]
            _hyp_insert = (min(_hxs2), min(_hys2))
            _hyp_size   = (max(_hxs2) - min(_hxs2), max(_hys2) - min(_hys2))

            _hyp_el = dwg.image(href=_href2, insert=_hyp_insert, size=_hyp_size)
            _hyp_el["preserveAspectRatio"] = "none"
            _hyp_g = dwg.g(clip_path="url(#map-area)", opacity=0.55)
            _hyp_g.add(_hyp_el)
            dwg.add(_hyp_g)
        except Exception:
            pass  # degrade gracefully if PIL unavailable

    # ------------------------------------------------------------------
    # 3c. Hillshade — embedded grayscale raster with multiply blend.
    #     White (lit) = no change; dark (shadow) darkens underlying layers.
    #     Opacity raised to 0.50 for dramatic East-of-Nowhere style depth.
    # ------------------------------------------------------------------
    if hillshade is not None:
        import io as _io
        import base64 as _base64
        import numpy as _np
        try:
            from PIL import Image as _PILImage
            _hs_u8 = (_np.clip(hillshade, 0.0, 1.0) * 255).astype(_np.uint8)
            _hs_img = _PILImage.fromarray(_hs_u8, mode="L")
            _buf = _io.BytesIO()
            _hs_img.save(_buf, format="PNG", optimize=False)
            _b64 = _base64.b64encode(_buf.getvalue()).decode("ascii")
            _href = f"data:image/png;base64,{_b64}"

            # Position the image using the DEM's actual geographic extent so it
            # aligns with the projected map coordinates rather than being naively
            # stretched to fill the map frame.  hillshade_transform is a rasterio
            # Affine (WGS84): transform * (col, row) → (lng, lat).
            if hillshade_transform is not None and transformer is not None:
                _hs_rows, _hs_cols = hillshade.shape
                _corners_pix = [
                    (0,          0         ),
                    (_hs_cols,   0         ),
                    (_hs_cols,   _hs_rows  ),
                    (0,          _hs_rows  ),
                ]
                _corners_svg = []
                for _pc, _pr in _corners_pix:
                    _lng, _lat = hillshade_transform * (_pc, _pr)
                    _e, _n = transformer.transform(_lat, _lng)
                    _corners_svg.append(proj_to_svg(_e, _n))
                _hxs = [p[0] for p in _corners_svg]
                _hys = [p[1] for p in _corners_svg]
                _hs_insert = (min(_hxs), min(_hys))
                _hs_size   = (max(_hxs) - min(_hxs), max(_hys) - min(_hys))
            else:
                # Fallback: stretch to map frame (no geo-transform available).
                _hs_insert = (offset_x, offset_y)
                _hs_size   = (map_w_mm, map_h_mm)

            _img_el = dwg.image(href=_href, insert=_hs_insert, size=_hs_size)
            _img_el["preserveAspectRatio"] = "none"
            _img_el["style"] = "mix-blend-mode:multiply;image-rendering:smooth;"
            _hs_g = dwg.g(clip_path="url(#map-area)", opacity=0.18)
            _hs_g.add(_img_el)
            dwg.add(_hs_g)
        except Exception as _hs_err:
            pass  # degrade gracefully if PIL unavailable

    # ------------------------------------------------------------------
    # 3d. Terrain hatch fills — landuse/natural area polygons.
    #     Vector hatch patterns overlay the hillshade raster; rendered
    #     at low opacity so depth information remains visible beneath.
    # ------------------------------------------------------------------
    if osm_vectors is not None and transformer is not None and osm_vectors.terrain_areas:
        _define_terrain_hatch_patterns(dwg)
        _render_terrain_fills(dwg, osm_vectors, transformer, proj_to_svg, clip_id="map-area")

    # ------------------------------------------------------------------
    # 4. OSM lower layers: water areas, waterways, roads
    #    Rendered above hillshade and below contour lines.
    # ------------------------------------------------------------------
    if osm_vectors is not None and transformer is not None:
        _render_osm_lower_layers(dwg, osm_vectors, transformer, proj_to_svg, clip_id="map-area")

    # ------------------------------------------------------------------
    # 4a. Protected area boundaries — above roads, below graticule
    # ------------------------------------------------------------------
    if osm_vectors is not None and transformer is not None and osm_vectors.protected_areas:
        # map extents not yet computed at this point — use offset/size directly
        _render_protected_areas(
            dwg, osm_vectors, transformer, proj_to_svg,
            clip_id="map-area",
            map_x0=offset_x, map_y0=offset_y,
            map_x1=offset_x + map_w_mm, map_y1=offset_y + map_h_mm,
        )

    # ------------------------------------------------------------------
    # 4b. Graticule — coordinate grid, above OSM, below wind + contours
    # ------------------------------------------------------------------
    _grat_params: tuple | None = None
    if transformer is not None:
        _grat_params = _render_graticule(
            dwg, transformer, bounds,
            proj_to_svg, offset_x, offset_y, map_w_mm, map_h_mm,
        )

    # ------------------------------------------------------------------
    # 4c. Wind streamlines — above graticule, below contours
    # ------------------------------------------------------------------
    if wind_streamlines:
        _render_wind_streamlines(dwg, wind_streamlines, proj_to_svg, clip_id="map-area")

    # ------------------------------------------------------------------
    # 5. Contour lines — three-tier visual hierarchy
    # ------------------------------------------------------------------
    import math as _cmath

    interval_m     = _infer_interval(contours)
    major_interval = interval_m * 10.0   # e.g. 100 m at 10 m interval
    index_interval = interval_m * 5.0    # e.g.  50 m at 10 m interval

    # Map extents in SVG mm — used for label clipping.
    map_x0 = offset_x
    map_y0 = offset_y
    map_x1 = offset_x + map_w_mm
    map_y1 = offset_y + map_h_mm

    def _snap(elev: float, step: float) -> bool:
        """True if *elev* is an integer multiple of *step* (float-safe)."""
        return step > 0 and abs(round(elev / step) * step - elev) < 0.1

    g_contours = dwg.g(id="contours", clip_path="url(#map-area)")

    # Accumulate the best (longest) label candidate for each MAJOR elevation.
    # Labels are placed only on major (heavy/primary) contour lines so the
    # elevation annotation and the visual hierarchy reinforce each other.
    # key: elevation float → value: (svg_cx, svg_cy, angle_deg, path_point_count)
    _major_label_best: dict[float, tuple] = {}

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

            # Douglas-Peucker in UTM metres before projecting.
            # Tolerance = 0.5 m: removes only sub-pixel staircase jitter
            # introduced by marching-squares boundary crossing (which places
            # points at fractional pixel positions).  All real terrain
            # inflections — ridgeline V-apexes, valley bends — have
            # perpendicular deviations >> 0.5 m and are fully preserved.
            # Keeping the full density of retained points is what gives the
            # path its geological texture and nuance.
            utm_pts = [(float(xy[0]), float(xy[1])) for xy in path]
            utm_pts = _douglas_peucker(utm_pts, tolerance=0.5)
            if len(utm_pts) < 2:
                continue

            pts = [proj_to_svg(xy[0], xy[1]) for xy in utm_pts]

            g_contours.add(dwg.polyline(
                pts,
                stroke=stroke_color,
                stroke_width=stroke_width,
                fill="none",
                stroke_linecap="round",
                stroke_linejoin="round",
            ))

            # Collect label placement for MAJOR contours only: track the
            # longest path at each elevation and record the midpoint + tangent.
            if tier == "major" and len(pts) >= 8:
                n_pts = len(pts)
                current_best = _major_label_best.get(elev)
                if current_best is None or n_pts > current_best[3]:
                    mid = n_pts // 2
                    # Stable tangent: 2-point lookahead/lookbehind.
                    k = min(2, mid, n_pts - mid - 1)
                    p0 = pts[max(0, mid - k)]
                    p1 = pts[min(n_pts - 1, mid + k)]
                    angle = _cmath.degrees(_cmath.atan2(p1[1] - p0[1], p1[0] - p0[0]))
                    # Keep text upright: flip if it would render upside-down.
                    if angle > 90.0:
                        angle -= 180.0
                    elif angle < -90.0:
                        angle += 180.0
                    cx, cy = pts[mid]
                    _major_label_best[elev] = (cx, cy, angle, n_pts)

    # --- Major contour elevation labels (renders above the polylines) --------
    # A white knockout rectangle creates the visual break in the contour line
    # so the elevation numeral floats cleanly within the line itself.
    g_labels = dwg.g(id="contour-labels")
    for elev, (cx, cy, angle_deg, _) in _major_label_best.items():
        # Skip labels whose centre falls outside the visible map.
        if not (map_x0 <= cx <= map_x1 and map_y0 <= cy <= map_y1):
            continue
        label = str(int(elev))
        # Extra horizontal padding widens the break so it reads as a true gap.
        tw = len(label) * _CONTOUR_LABEL_CHAR_W + 1.4
        th = _CONTOUR_LABEL_FONT_MM + 0.6
        g_lbl = dwg.g(transform=f"rotate({angle_deg:.1f},{cx:.3f},{cy:.3f})")
        g_lbl.add(dwg.rect(
            insert=(cx - tw / 2, cy - th / 2),
            size=(tw, th),
            fill="white",
            stroke="none",
        ))
        g_lbl.add(dwg.text(
            label,
            insert=(cx, cy + _CONTOUR_LABEL_FONT_MM * 0.35),
            text_anchor="middle",
            font_family="Liberation Sans, Arial, sans-serif",
            font_size=_CONTOUR_LABEL_FONT_MM,
            fill=_CONTOUR_TIERS["major"][0],
        ))
        g_labels.add(g_lbl)

    g_contours.add(g_labels)
    dwg.add(g_contours)

    # ------------------------------------------------------------------
    # 6. OSM trail network — dashed gray, above contours, below route
    # ------------------------------------------------------------------
    if osm_vectors is not None and transformer is not None:
        _render_osm_trails(dwg, osm_vectors, transformer, proj_to_svg, clip_id="map-area")

    # ------------------------------------------------------------------
    # 6a. Waterway + trail name labels — inline rotated text
    # ------------------------------------------------------------------
    if osm_vectors is not None and transformer is not None:
        _render_osm_line_labels(
            dwg, osm_vectors, transformer, proj_to_svg,
            map_x0=offset_x, map_y0=offset_y,
            map_x1=offset_x + map_w_mm, map_y1=offset_y + map_h_mm,
        )

    # ------------------------------------------------------------------
    # 7. Route polyline
    # ------------------------------------------------------------------
    import math as _math
    from field_atlas.render.color_palettes import PALETTES, get_palette, interpolate_color

    # Determine whether speed coloring is active.
    # segment_speeds must have exactly len(route_points)-1 entries to be usable.
    _speeds = segment_speeds or []
    _use_speed_color = bool(_speeds) and len(_speeds) == len(route_points) - 1

    if len(route_points) >= 2:
        g_route = dwg.g(id="route")

        route_pts_proj = [(pt["x"], pt["y"]) for pt in route_points]
        route_pts = [proj_to_svg(x, y) for x, y in route_pts_proj]

        # Route stroke: 1.8 mm core, 2.7 mm white casing (1.5× ratio)
        _casing_w = route_width * (2.7 / 1.8)
        _core_w   = route_width

        if _use_speed_color:
            # Percentile-based normalisation (5th → 0.0, 95th → 1.0).
            _sorted = sorted(_speeds)
            _n = len(_sorted)
            _lo = _sorted[max(0, int(_n * 0.05))]
            _hi = _sorted[min(_n - 1, int(_n * 0.95))]
            if _hi <= _lo:
                _norm = [0.5] * _n
            else:
                _norm = [max(0.0, min(1.0, (s - _lo) / (_hi - _lo))) for s in _speeds]

            _palette = get_palette(route_palette)

            # --- Build per-point normalised speed values --------------------
            # _norm is per-segment (len = n_pts - 1); average adjacent segs
            # to get per-point values, then apply a rolling average (window=5)
            # to damp GPS jitter before passing to the spline.
            _n_pts = len(route_pts)
            _pt_norms_raw: list[float] = [_norm[0]]
            for _pi in range(1, _n_pts - 1):
                _pt_norms_raw.append((_norm[_pi - 1] + _norm[_pi]) / 2.0)
            _pt_norms_raw.append(_norm[-1])

            _W = 5   # rolling-average window
            _pt_norms: list[float] = []
            for _pi in range(_n_pts):
                _lo = max(0, _pi - _W // 2)
                _hi = min(_n_pts, _pi + _W // 2 + 1)
                _pt_norms.append(sum(_pt_norms_raw[_lo:_hi]) / (_hi - _lo))

            # --- Arc-length CubicSpline resampling --------------------------
            # Resample the raw GPS points onto a uniform arc-length grid so
            # the final curve has steady density regardless of GPS sampling
            # rate.  600 points gives ~3 m between consecutive resampled pts
            # on a 1.9 km route — dense enough for a perfectly smooth stroke.
            _RESAMPLE_N = 600
            _smooth_pts, _smooth_norms = _resample_route_spline(
                route_pts, _pt_norms, _RESAMPLE_N
            )
            _smooth_colors = [interpolate_color(v, _palette) for v in _smooth_norms]

            # Pass 1 — black casing: single smooth Bézier path.
            # _catmull_rom_path produces cubic Bézier C commands through the
            # resampled points (already dense, so curves are essentially arcs).
            g_route.add(dwg.path(
                d=_catmull_rom_path(_smooth_pts),
                stroke="#000000",
                stroke_width=_casing_w,
                fill="none",
                stroke_linejoin="round",
                stroke_linecap="round",
            ))

            # Pass 2 — per-segment along-path gradients.
            #
            # Each short segment A→B gets its own linearGradient that runs
            # exactly from A to B (gradientUnits="userSpaceOnUse").  The
            # gradient is therefore perpendicular to the segment at every
            # cross-section — colour flows along the path, not across some
            # fixed screen-space diagonal.  With round linecaps, adjacent
            # segments blend seamlessly at their shared endpoints.
            g_segs = dwg.g(id="route-speed", clip_path="url(#map-area)")
            _n_smooth = len(_smooth_pts)
            for _i in range(_n_smooth - 1):
                _p0 = _smooth_pts[_i]
                _p1 = _smooth_pts[_i + 1]
                _c0 = _smooth_colors[_i]
                _c1 = _smooth_colors[_i + 1]

                # Skip degenerate zero-length segments (shouldn't occur after
                # deduplication in the resampler, but guard defensively).
                _dx = _p1[0] - _p0[0]
                _dy = _p1[1] - _p0[1]
                if _dx * _dx + _dy * _dy < 1e-10:
                    continue

                _grad = dwg.defs.add(dwg.linearGradient(
                    id=f"rg-{_i}",
                    gradientUnits="userSpaceOnUse",
                    x1=f"{_p0[0]:.4f}", y1=f"{_p0[1]:.4f}",
                    x2=f"{_p1[0]:.4f}", y2=f"{_p1[1]:.4f}",
                ))
                _grad.add_stop_color(0,   _c0)
                _grad.add_stop_color(1.0, _c1)

                g_segs.add(dwg.path(
                    d=f"M {_p0[0]:.4f},{_p0[1]:.4f} L {_p1[0]:.4f},{_p1[1]:.4f}",
                    stroke=f"url(#rg-{_i})",
                    stroke_width=_core_w,
                    stroke_linecap="round",
                    stroke_linejoin="round",
                    fill="none",
                ))
            g_route.add(g_segs)

        else:
            # Solid fallback — smooth sparse routes with Catmull-Rom.
            if len(route_pts) < 100:
                route_pts = _catmull_rom_smooth(route_pts)

            # Pass 1 — black casing.
            g_route.add(dwg.polyline(
                route_pts,
                stroke="#000000",
                stroke_width=_casing_w,
                fill="none",
                stroke_linejoin="round",
                stroke_linecap="round",
            ))
            # Pass 2 — solid blue.
            g_route.add(dwg.polyline(
                route_pts,
                stroke="#2E75B6",
                stroke_width=_core_w,
                fill="none",
                stroke_linejoin="round",
                stroke_linecap="round",
            ))

        # Loop detection: start and end within 50 m → single shared marker.
        p0, p1 = route_pts_proj[0], route_pts_proj[-1]
        is_loop = _math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 50.0

        sx, sy = proj_to_svg(route_pts_proj[0][0], route_pts_proj[0][1])

        # Start marker — white circle with black outline, r=3.0 mm.
        g_route.add(dwg.circle(
            center=(sx, sy),
            r=3.0,
            fill="#FFFFFF",
            stroke="#000000",
            stroke_width=0.45,
        ))

        if not is_loop:
            ex, ey = proj_to_svg(route_pts_proj[-1][0], route_pts_proj[-1][1])
            # End marker — same style as start, 30% smaller.
            g_route.add(dwg.circle(
                center=(ex, ey),
                r=2.1,
                fill="#FFFFFF",
                stroke="#000000",
                stroke_width=0.45,
            ))

        dwg.add(g_route)

    # ------------------------------------------------------------------
    # 5a. Peak crosshairs — hairline marks at every detected high point
    # ------------------------------------------------------------------
    if peak_markers and transformer is not None:
        _render_peak_crosshairs(
            dwg, peak_markers, transformer, proj_to_svg,
            map_x0=offset_x,
            map_y0=offset_y,
            map_x1=offset_x + map_w_mm,
            map_y1=offset_y + map_h_mm,
        )

    # ------------------------------------------------------------------
    # 5b. Feature labels
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

    # ------------------------------------------------------------------
    # 8. Speed legend — only when speed data is available
    # Centred on the same horizontal strip as the north arrow (step 10).
    # ------------------------------------------------------------------
    # Pre-compute bottom-strip shared position (also used by north arrow).
    # 4 mm clearance gap below the outer neatline border edge.
    _R_ARROW      = 7.0
    _BELOW_BORDER = _NL_TOTAL_W + 4.0   # mm below map bottom → top of below-border strip
    _cy_arrow     = height_mm - bottom_margin_mm + _BELOW_BORDER + _R_ARROW

    if _use_speed_color:
        _palette = get_palette(route_palette)
        LEGEND_W, LEGEND_H = 35.0, 5.0   # wider + taller than before
        # Right-align to the outer neatline right edge (matches scale bar)
        legend_x = offset_x + map_w_mm + _NL_TOTAL_W - LEGEND_W
        legend_y = _cy_arrow - LEGEND_H / 2   # vertically centred on arrow

        # Define a horizontal linearGradient for the legend bar.
        _LEGEND_STOPS = 21
        _grad = dwg.linearGradient(id="speed-legend-grad", x1="0%", y1="0%", x2="100%", y2="0%")
        for _li in range(_LEGEND_STOPS):
            _lt = _li / (_LEGEND_STOPS - 1)
            _grad.add_stop_color(_lt, interpolate_color(_lt, _palette))
        dwg.defs.add(_grad)

        g_legend = dwg.g(id="speed-legend")
        g_legend.add(dwg.rect(
            insert=(legend_x, legend_y),
            size=(LEGEND_W, LEGEND_H),
            fill="url(#speed-legend-grad)",
            rx=0.8, ry=0.8,
        ))

        _lbl_sz  = 2.82    # 8pt — doubled from 4pt
        _lbl_col = "#888888"
        _lbl_y   = legend_y + LEGEND_H + _lbl_sz * 0.9
        g_legend.add(dwg.text(
            "SLOW",
            insert=(legend_x, _lbl_y),
            text_anchor="start",
            font_family="Liberation Sans, Arial, sans-serif",
            font_size=_lbl_sz,
            fill=_lbl_col,
            **{"letter-spacing": "0.04em"},
        ))
        g_legend.add(dwg.text(
            "FAST",
            insert=(legend_x + LEGEND_W, _lbl_y),
            text_anchor="end",
            font_family="Liberation Sans, Arial, sans-serif",
            font_size=_lbl_sz,
            fill=_lbl_col,
            **{"letter-spacing": "0.04em"},
        ))
        dwg.add(g_legend)

    # ------------------------------------------------------------------
    # 9. Checkered neatline border — above all map content, below margins
    # ------------------------------------------------------------------
    if _grat_params is not None:
        _min_lat, _max_lat, _min_lng, _max_lng, _iv_deg, _subdiv_deg, _iv_min = _grat_params
        _render_checkered_border(
            dwg, transformer, proj_to_svg,
            _min_lat, _max_lat, _min_lng, _max_lng, _subdiv_deg,
            offset_x, offset_y, map_w_mm, map_h_mm,
        )

    # ------------------------------------------------------------------
    # 10. North arrow — clean two-tone navigator's needle, lower-left
    # Left wing aligns to the outer edge of the neatline border.
    # Vertical centre shared with speed legend (computed in step 8).
    # ------------------------------------------------------------------
    # Left wing tip sits at the outer neatline left edge
    _cx_arrow = (offset_x - _NL_TOTAL_W) + _R_ARROW * 0.62
    _draw_north_arrow(dwg, cx=_cx_arrow, cy=_cy_arrow, r_mm=_R_ARROW)

    # ------------------------------------------------------------------
    # 11. Scale bar — lower-right margin, right-aligned with neatline
    # ------------------------------------------------------------------
    # The bar sits in the bottom-margin strip at the same vertical level as
    # the north arrow, mirrored to the right side of the page.
    # Right-align to the outer right edge of the neatline border.
    _draw_scale_bar(
        dwg,
        scale_mm_per_m=scale,
        right_edge_x=offset_x + map_w_mm + _NL_TOTAL_W,
        bar_top_y=height_mm - bottom_margin_mm + _BELOW_BORDER,
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
      6  "FIELDNOTES" wordmark, right-aligned                     5 pt

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

    MAIN_COLOR  = "#1C1813"
    SOFT_COLOR  = "#5A4E42"
    RULE_COLOR  = "#C0A888"
    RULE_HALF_W = 80.0          # rule extends ±80 mm from centre (160 mm total)
    RULE_SW     = 0.50          # ≈ 1.4 pt stroke-width in mm
    FONT        = "Liberation Sans, Arial, Helvetica, sans-serif"
    FONT_TITLE  = "Cinzel, 'Trajan Pro', Palatino, 'Times New Roman', serif"
    FONT_DATE   = "'Cormorant Garamond', 'Cormorant Garamond Light', Garamond, Palatino, serif"
    FONT_MONO   = "Liberation Mono, DejaVu Sans Mono, 'Courier New', Courier, monospace"

    cx           = width_mm / 2.0                  # horizontal centre of canvas
    margin_top_y = height_mm - bottom_margin_mm    # top edge of title block strip

    # Vertical positions — Row 1 baseline anchored from top of the strip.
    # Extra 6.35 mm (¼ in) of breathing room below the neatline border.
    PAD_TOP = 13.85
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
    r1 = svg_drawing.text(location, insert=(cx, r1_y), font_size=R1_MM,
                          font_family=FONT_TITLE, fill=MAIN_COLOR, text_anchor="middle")
    r1["letter-spacing"] = f"{0.18 * R1_MM:.3f}"
    svg_drawing.add(r1)

    # ---- Row 2: Date, long-form human format ----------------------------
    try:
        dt      = _dt.strptime(date, "%Y-%m-%d")
        fmt_date = f"{dt.strftime('%B')} {dt.day}, {dt.year}"
    except ValueError:
        fmt_date = date
    r2 = svg_drawing.text(
        fmt_date, insert=(cx, r2_y), font_size=R2_MM,
        font_family=FONT_DATE, fill=SOFT_COLOR, text_anchor="middle",
    )
    r2["font-style"] = "italic"
    svg_drawing.add(r2)

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
        "  ·  ".join(parts4), insert=(cx, r4_y), font_size=R4_MM,
        font_family=FONT_MONO, fill=SOFT_COLOR, text_anchor="middle",
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

    # ---- Row 6: "FIELDNOTES" wordmark — right-aligned to outer neatline edge,
    # vertically pinned to the stats row so it reads as a colophon in-line
    # with the route data rather than a floating footer.
    rx = width_mm - margin_mm + _NL_TOTAL_W   # outer right edge of neatline border
    r6 = svg_drawing.text(
        "FIELDNOTES",
        insert=(rx, r4_y),
        font_size=R6_MM,
        font_family=FONT_MONO,
        fill=SOFT_COLOR,
        text_anchor="end",
    )
    r6["letter-spacing"] = f"{0.18 * R6_MM:.3f}"
    svg_drawing.add(r6)
