"""field_atlas.render.template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate a blank, Illustrator-ready SVG template that faithfully
represents the field_atlas page layout.

Every group uses the same ``id`` that the live data pipeline injects
into, so this file serves as both:

  * a visual reference / layout guide you can open straight in Illustrator
  * a starting shell for the "SVG template + auto-populate" workflow:
    open, style, save as .ai, then point the pipeline at it

Layer / group IDs
-----------------
  background          Cream page fill (#FBF7F0)
  map-content         Map area — hillshade, contours, route injected here
  neatline-border     Checkered cartographic border (data-driven in live render)
  graticule           Coordinate grid labels (data-driven in live render)
  north-arrow         Survey-style geometric north arrow
  scale-bar           Dual-unit metric/imperial scale bar
  speed-legend        Speed colour ramp (omitted when no speed data)
  title-block
    └─ title-text     Route / location name  (Cinzel, upper-case, spaced)
    └─ date-text      Date  (Cormorant Garamond italic)
    └─ rule           Decorative hairline rule
    └─ stats-text     Distance · gain · duration · coordinates (Liberation Sans)
    └─ wordmark       "FIELDNOTES" brand mark

Usage
-----
    from field_atlas.render.template import generate_template
    generate_template("my_template.svg")

Or via CLI:
    python -m field_atlas template [OUTPUT_PATH]
"""
from __future__ import annotations

from pathlib import Path

import svgwrite

from .svg_composer import (
    _NL_TOTAL_W,
    _NL_BAND_W,
    _NL_BLACK,
    _NL_WHITE,
    _NL_STROKE,
    _NL_BAND_OPY,
    _draw_north_arrow,
    _draw_scale_bar,
)

# ---------------------------------------------------------------------------
# Page geometry — mirrors render_terrain_svg defaults
# ---------------------------------------------------------------------------
PAGE_W_MM    = 457.2   # 18 in
PAGE_H_MM    = 609.6   # 24 in
MARGIN_MM    = 38.1    # 1.5 in  (left / right / top)
BOTTOM_MM    = 76.2    # 3 in   (title block)

_PW          = PAGE_W_MM - 2 * MARGIN_MM   # 381.0 mm printable width
_PH          = PAGE_H_MM - MARGIN_MM - BOTTOM_MM  # 495.3 mm printable height

# Representative map area — centred in printable region.
# Chosen to give a ~2:3 map with a little breathing room all round.
MAP_W_MM     = 358.0
MAP_H_MM     = 462.0
OFFSET_X     = MARGIN_MM + (_PW - MAP_W_MM) / 2   # ≈ 49.6 mm
OFFSET_Y     = MARGIN_MM + (_PH - MAP_H_MM) / 2   # ≈ 54.75 mm

# Scale typical of a 12 km-wide hiking area on this canvas
_SCALE       = MAP_W_MM / 12_000   # 0.02983 mm/m → ~1:33 500

# ---------------------------------------------------------------------------
# Typography constants — identical to svg_composer
# ---------------------------------------------------------------------------
_FONT_TITLE  = "Cinzel, 'Trajan Pro', Palatino, 'Times New Roman', serif"
_FONT_DATE   = ("'Cormorant Garamond', 'Cormorant Garamond Light', "
                "Garamond, Palatino, serif")
_FONT_SANS   = "Liberation Sans, Arial, Helvetica, sans-serif"
_FONT_MONO   = "Liberation Mono, 'Courier New', Courier, monospace"

_MAIN  = "#1C1813"
_SOFT  = "#5A4E42"
_RULE  = "#C0A888"
_DIM   = "#888880"
_GHOST = "#BBBBBB"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_template(output_path: str | Path | None = None) -> str:
    """Write the template SVG and return its absolute path."""
    if output_path is None:
        output_path = Path("fieldnotes_template.svg")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    dwg = svgwrite.Drawing(
        str(out),
        size=(f"{PAGE_W_MM}mm", f"{PAGE_H_MM}mm"),
        viewBox=f"0 0 {PAGE_W_MM} {PAGE_H_MM}",
    )

    ox, oy, mw, mh = OFFSET_X, OFFSET_Y, MAP_W_MM, MAP_H_MM

    _add_background(dwg)
    _add_map_placeholder(dwg, ox, oy, mw, mh)
    _add_neatline(dwg, ox, oy, mw, mh)
    _add_graticule_placeholders(dwg, ox, oy, mw, mh)

    # Shared vertical position for below-border strip elements
    _R_ARROW      = 7.0
    _BELOW_BORDER = _NL_TOTAL_W + 4.0
    _cy_arrow     = PAGE_H_MM - BOTTOM_MM + _BELOW_BORDER + _R_ARROW
    _cx_arrow     = (ox - _NL_TOTAL_W) + _R_ARROW * 0.62

    _draw_north_arrow(dwg, cx=_cx_arrow, cy=_cy_arrow, r_mm=_R_ARROW)
    _draw_scale_bar(
        dwg,
        scale_mm_per_m=_SCALE,
        right_edge_x=ox + mw + _NL_TOTAL_W,
        bar_top_y=PAGE_H_MM - BOTTOM_MM + _BELOW_BORDER,
    )
    _add_legend_placeholder(dwg, ox, mw, _cy_arrow)
    _add_title_block(dwg, ox, PAGE_W_MM, PAGE_H_MM, BOTTOM_MM, MARGIN_MM)

    dwg.save()
    return str(out.resolve())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _add_background(dwg: svgwrite.Drawing) -> None:
    g = dwg.g(id="background")
    g.add(dwg.rect(
        insert=(0, 0), size=(PAGE_W_MM, PAGE_H_MM),
        fill="#FBF7F0", stroke="none",
    ))
    dwg.add(g)


def _add_map_placeholder(
    dwg: svgwrite.Drawing,
    ox: float, oy: float, mw: float, mh: float,
) -> None:
    """Gray-parchment map area with centred annotation text."""
    # Define a diagonal hatch pattern
    pat = dwg.defs.add(dwg.pattern(
        id="map-hatch",
        x=0, y=0, width=18, height=18,
        patternUnits="userSpaceOnUse",
    ))
    pat.add(dwg.line(
        start=(0, 18), end=(18, 0),
        stroke="#C8BEAA", stroke_width=0.25,
    ))

    g = dwg.g(id="map-content")
    # Parchment base (same colour as live render)
    g.add(dwg.rect(
        insert=(ox, oy), size=(mw, mh),
        fill="#EDE3B8", stroke="none", opacity=0.55,
    ))
    # Subtle diagonal hatch to signal "placeholder"
    g.add(dwg.rect(
        insert=(ox, oy), size=(mw, mh),
        fill="url(#map-hatch)", stroke="none", opacity=0.25,
    ))

    # Centre annotation
    cxc = ox + mw / 2
    cyc = oy + mh / 2
    for txt, dy, sz, col in (
        ("MAP CONTENT", -10.0, 9.0, "#8A8072"),
        ("Hillshade · Contours · Route · Vector Layers", 2.0, 5.0, "#AAAAAA"),
        ("Injected from GPX + DEM data by field_atlas", 9.5, 4.2, "#BBBBBB"),
    ):
        t = dwg.text(
            txt,
            insert=(cxc, cyc + dy),
            text_anchor="middle",
            font_family=_FONT_SANS,
            font_size=sz,
            fill=col,
            **{"letter-spacing": "0.06em"},
        )
        t["dominant-baseline"] = "middle"
        g.add(t)

    dwg.add(g)


def _add_neatline(
    dwg: svgwrite.Drawing,
    ox: float, oy: float, mw: float, mh: float,
) -> None:
    """Simplified checkered neatline — evenly spaced segments.

    The live render drives segment spacing from geographic coordinate
    subdivisions; here we use a fixed 15 mm pitch so the template
    looks accurate without needing real spatial data.
    """
    BW  = _NL_TOTAL_W   # 5 mm total border width
    HW  = _NL_BAND_W    # 2.5 mm per band
    B   = _NL_BLACK
    W   = _NL_WHITE
    OPY = _NL_BAND_OPY
    SEG = 15.0           # checker segment pitch (mm)

    rx  = ox + mw
    by  = oy + mh

    g = dwg.g(id="neatline-border")

    def _checker_row(x0, y0, length, dx, dy, band_x, band_y, bw, bh):
        """Draw two alternating-colour bands along a side."""
        pos = 0.0
        i   = 0
        while pos < length:
            seg = min(SEG, length - pos)
            c1 = B if i % 2 == 0 else W
            c2 = W if i % 2 == 0 else B
            # outer band
            g.add(dwg.rect(
                insert=(x0 + dx * pos + band_x[0], y0 + dy * pos + band_y[0]),
                size=(bw[0], bh[0]),
                fill=c1, opacity=OPY,
            ))
            # inner band
            g.add(dwg.rect(
                insert=(x0 + dx * pos + band_x[1], y0 + dy * pos + band_y[1]),
                size=(bw[1], bh[1]),
                fill=c2, opacity=OPY,
            ))
            pos += seg
            i   += 1

    # Top side  (y = oy - BW … oy)
    _checker_row(ox, oy, mw, 1, 0,
                 [0, 0], [-BW, -HW], [SEG, SEG], [HW, HW])
    # Bottom side  (y = by … by + BW)
    _checker_row(ox, by, mw, 1, 0,
                 [0, 0], [0, HW],   [SEG, SEG], [HW, HW])
    # Left side  (x = ox - BW … ox)
    _checker_row(ox, oy, mh, 0, 1,
                 [-BW, -HW], [0, 0], [HW, HW], [SEG, SEG])
    # Right side  (x = rx … rx + BW)
    _checker_row(rx, oy, mh, 0, 1,
                 [0,   HW],  [0, 0], [HW, HW], [SEG, SEG])

    # Corner squares — solid black
    for cx_, cy_ in [(ox - BW, oy - BW), (rx, oy - BW),
                     (ox - BW, by),       (rx, by)]:
        g.add(dwg.rect(
            insert=(cx_, cy_), size=(BW, BW),
            fill=B, opacity=OPY,
        ))

    # Hairline inner and outer frame lines
    for rx2, ry2, rw, rh in [
        (ox - BW, oy - BW, mw + 2 * BW, mh + 2 * BW),  # outer
        (ox,      oy,      mw,           mh),             # inner
    ]:
        g.add(dwg.rect(
            insert=(rx2, ry2), size=(rw, rh),
            fill="none", stroke=B, stroke_width=_NL_STROKE,
        ))

    dwg.add(g)


def _add_graticule_placeholders(
    dwg: svgwrite.Drawing,
    ox: float, oy: float, mw: float, mh: float,
) -> None:
    """Representative coordinate labels and tick marks.

    Real values are computed from the route bounding box; these
    placeholders use typical Pacific-Northwest coordinates so the
    template looks realistic.
    """
    BW  = _NL_TOTAL_W
    GAP = 1.5
    FSZ = 3.5   # label font size

    g = dwg.g(id="graticule")

    def _lbl(txt, x, y, anchor="middle", rotate=0):
        t = dwg.text(
            txt,
            insert=(x, y),
            text_anchor=anchor,
            font_family=_FONT_SANS,
            font_size=FSZ,
            fill=_DIM,
        )
        t["dominant-baseline"] = "middle"
        if rotate:
            t["transform"] = f"rotate({rotate},{x},{y})"
        g.add(t)

    # Top edge — longitude labels (3 evenly spaced)
    for i, lng_label in enumerate(["121°20'W", "121°10'W", "121°00'W"]):
        x = ox + mw * (i + 1) / 4
        g.add(dwg.line(
            start=(x, oy - BW), end=(x, oy - BW - 2.0),
            stroke=_DIM, stroke_width=0.2,
        ))
        _lbl(lng_label, x, oy - BW - GAP - 1.2)

    # Bottom edge — longitude labels
    for i, lng_label in enumerate(["121°20'W", "121°10'W", "121°00'W"]):
        x = ox + mw * (i + 1) / 4
        g.add(dwg.line(
            start=(x, oy + mh + BW), end=(x, oy + mh + BW + 2.0),
            stroke=_DIM, stroke_width=0.2,
        ))
        _lbl(lng_label, x, oy + mh + BW + GAP + FSZ * 0.6)

    # Left edge — latitude labels (rotated, reads upward)
    for i, lat_label in enumerate(["48°20'N", "48°25'N", "48°30'N"]):
        y = oy + mh * (3 - i) / 4
        g.add(dwg.line(
            start=(ox - BW, y), end=(ox - BW - 2.0, y),
            stroke=_DIM, stroke_width=0.2,
        ))
        _lbl(lat_label, ox - BW - GAP - FSZ * 0.5, y, anchor="middle", rotate=-90)

    # Right edge — latitude labels
    for i, lat_label in enumerate(["48°20'N", "48°25'N", "48°30'N"]):
        y = oy + mh * (3 - i) / 4
        g.add(dwg.line(
            start=(ox + mw + BW, y), end=(ox + mw + BW + 2.0, y),
            stroke=_DIM, stroke_width=0.2,
        ))
        _lbl(lat_label, ox + mw + BW + GAP + FSZ * 0.5, y, anchor="middle", rotate=90)

    dwg.add(g)


def _add_legend_placeholder(
    dwg: svgwrite.Drawing,
    ox: float, mw: float, cy_arrow: float,
) -> None:
    """Speed colour-ramp legend — shows the gradient with SLOW/FAST labels."""
    LEGEND_W = 35.0
    LEGEND_H = 5.0
    legend_x = ox + mw + _NL_TOTAL_W - LEGEND_W
    legend_y = cy_arrow - LEGEND_H / 2

    # Define a representative blue→orange gradient
    grad = dwg.linearGradient(
        id="template-speed-grad", x1="0%", y1="0%", x2="100%", y2="0%",
    )
    grad.add_stop_color(0.0,  "#2060C8")
    grad.add_stop_color(0.33, "#20B8A0")
    grad.add_stop_color(0.66, "#E0C020")
    grad.add_stop_color(1.0,  "#E04020")
    dwg.defs.add(grad)

    g = dwg.g(id="speed-legend")
    g.add(dwg.rect(
        insert=(legend_x, legend_y), size=(LEGEND_W, LEGEND_H),
        fill="url(#template-speed-grad)", rx=0.8, ry=0.8,
    ))

    lbl_sz  = 2.82
    lbl_col = "#888888"
    lbl_y   = legend_y + LEGEND_H + lbl_sz * 0.9
    for txt, x, anchor in [
        ("SLOW", legend_x,              "start"),
        ("FAST", legend_x + LEGEND_W,   "end"),
    ]:
        g.add(dwg.text(
            txt,
            insert=(x, lbl_y),
            text_anchor=anchor,
            font_family=_FONT_SANS,
            font_size=lbl_sz,
            fill=lbl_col,
            **{"letter-spacing": "0.04em"},
        ))
    dwg.add(g)


def _add_title_block(
    dwg: svgwrite.Drawing,
    ox: float,
    page_w: float,
    page_h: float,
    bottom_mm: float,
    margin_mm: float,
) -> None:
    """Placeholder title block — same layout and font sizes as the live render."""
    # Row positions — identical vertical arithmetic to add_title_block()
    R1_MM  = 10.59   # 30 pt — location name
    R2_MM  =  8.46   # 24 pt — date
    R4_MM  =  7.41   # 21 pt — stats
    R6_MM  =  5.28   # 15 pt — wordmark

    cx           = page_w / 2.0
    margin_top_y = page_h - bottom_mm
    PAD_TOP      = 13.85
    r1_y         = margin_top_y + PAD_TOP + R1_MM
    r2_y         = r1_y  +  9.0
    rule_y       = r2_y  + 12.0
    r4_y         = rule_y + 12.0
    r6_y         = r4_y  + 15.0

    g = dwg.g(id="title-block")

    # Row 1 — location name
    g1 = dwg.g(id="title-text")
    t1 = dwg.text(
        "LOCATION NAME",
        insert=(cx, r1_y),
        text_anchor="middle",
        font_family=_FONT_TITLE,
        font_size=R1_MM,
        fill=_MAIN,
        **{"letter-spacing": "0.22em"},
    )
    t1["dominant-baseline"] = "auto"
    g1.add(t1)
    g.add(g1)

    # Row 2 — date
    g2 = dwg.g(id="date-text")
    t2 = dwg.text(
        "Month DD, YYYY",
        insert=(cx, r2_y),
        text_anchor="middle",
        font_family=_FONT_DATE,
        font_size=R2_MM,
        font_style="italic",
        fill=_SOFT,
        **{"letter-spacing": "0.04em"},
    )
    t2["dominant-baseline"] = "auto"
    g2.add(t2)
    g.add(g2)

    # Decorative rule
    g_rule = dwg.g(id="rule")
    RULE_HALF_W = 80.0
    g_rule.add(dwg.line(
        start=(cx - RULE_HALF_W, rule_y),
        end=(cx + RULE_HALF_W, rule_y),
        stroke=_RULE, stroke_width=0.50,
    ))
    g.add(g_rule)

    # Row 4 — stats
    g4 = dwg.g(id="stats-text")
    t4 = dwg.text(
        "XX.X\u202FKM  ·  XXXX\u202FM GAIN  ·  XX H XX M  ·  48.51°N\u2002121.22°W",
        insert=(cx, r4_y),
        text_anchor="middle",
        font_family=_FONT_SANS,
        font_size=R4_MM,
        fill=_SOFT,
        **{"letter-spacing": "0.04em"},
    )
    t4["dominant-baseline"] = "auto"
    g4.add(t4)
    g.add(g4)

    # Row 6 — wordmark
    g6 = dwg.g(id="wordmark")
    wm_x = ox + MAP_W_MM + _NL_TOTAL_W   # right-aligned to neatline right edge
    t6 = dwg.text(
        "FIELDNOTES",
        insert=(wm_x, r6_y),
        text_anchor="end",
        font_family=_FONT_SANS,
        font_size=R6_MM,
        fill=_DIM,
        **{"letter-spacing": "0.20em"},
    )
    t6["dominant-baseline"] = "auto"
    g6.add(t6)
    g.add(g6)

    dwg.add(g)
