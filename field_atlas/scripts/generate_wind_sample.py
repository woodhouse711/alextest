"""generate_wind_sample.py — Botany Bay full-map wind layer preview.

Runs the complete GPX → DEM → terrain → wind pipeline for Botany Bay
and writes a print-preview PNG showing contours, hillshade, route, and
wind arrows as they will appear in the final atlas page.

Wind arrows use grid seeding with *variable length*: arrows at each grid
cell are integrated for 3–12 steps depending on local wind speed, so
fast zones show long arrows and calm zones show short stubs.

Outputs
-------
  • sample_data/wind_streamlines_botany_bay.json  — streamline cache
  • output/botany_bay_wind.svg                    — full-map SVG
  • docs/preview/<YYYY-MM-DD>/wind_draft_<HHMMSS>.png — PNG preview

Usage::

    cd field_atlas
    python scripts/generate_wind_sample.py
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from field_atlas.core.dem_fetcher import load_dem
from field_atlas.core.gpx_parser import padded_bounds, parse_gpx
from field_atlas.core.projection import get_projection, project_bounds, project_points
from field_atlas.core.terrain_processor import generate_contours, generate_hillshade
from field_atlas.enrichment.wind import (
    build_wind_field,
    save_streamlines_to_file,
    trace_streamlines,
)
from field_atlas.render.svg_composer import render_terrain_svg

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SAMPLE_DIR   = Path(__file__).parents[1] / "sample_data"
GPX_PATH     = SAMPLE_DIR / "Botany_Bay.gpx"
DEM_PATH     = SAMPLE_DIR / "dem_botany_bay.tif"
OUT_JSON     = SAMPLE_DIR / "wind_streamlines_botany_bay.json"
OUT_SVG      = Path(__file__).parents[1] / "output" / "botany_bay_wind.svg"
PREVIEW_ROOT = Path(__file__).parents[1] / "docs" / "preview"

# SW wind at 19 km/h  (meteorological: comes FROM 225°)
WIND_SPEED_KMH = 19.0
WIND_DIR_DEG   = 225.0

# Grid: 24×24 = 576 seeds; max 12 integration steps per arrow (min 3).
GRID_ROWS = 24
GRID_COLS = 24
STEPS     = 12   # max steps for fastest arrows; calm → 3 steps (auto-scaled)

_MARGIN_MM        = 1.5 * 25.4   # 38.1 mm
_BOTTOM_MARGIN_MM = 3.0 * 25.4   # 76.2 mm

# ---------------------------------------------------------------------------
# 1. Parse GPX
# ---------------------------------------------------------------------------
print("Loading GPX…")
track = parse_gpx(str(GPX_PATH))
n_pts = len(track.points)
print(f"  {track.name}: {n_pts} points, {track.total_distance_km:.1f} km")

bounds_wgs84 = padded_bounds(track, padding_pct=0.2)
centroid_lat = (track.bounds["min_lat"] + track.bounds["max_lat"]) / 2.0
centroid_lng = (track.bounds["min_lng"] + track.bounds["max_lng"]) / 2.0

transformer      = get_projection(centroid_lat, centroid_lng)
bounds_projected = project_bounds(bounds_wgs84, transformer)
route_pts_utm    = project_points(track.points, transformer)

print(f"  Bounds (UTM): x [{bounds_projected['min_x']:.0f}, {bounds_projected['max_x']:.0f}]"
      f"  y [{bounds_projected['min_y']:.0f}, {bounds_projected['max_y']:.0f}]")

# ---------------------------------------------------------------------------
# 2. Load DEM, hillshade, contours
# ---------------------------------------------------------------------------
print("Loading DEM…")
elevation, meta = load_dem(str(DEM_PATH))
hillshade = generate_hillshade(elevation)
print(f"  Elevation shape: {elevation.shape}")

print("Generating contours (20 m interval)…")
contours_raw = generate_contours(elevation, meta["transform"], interval_m=20.0)

# Project contour paths to UTM.
contours = []
for contour in contours_raw:
    projected_paths = []
    for path in contour["paths"]:
        if not path:
            continue
        lats = [xy[1] for xy in path]
        lngs = [xy[0] for xy in path]
        eastings, northings = transformer.transform(lats, lngs)
        projected_paths.append([[float(e), float(n)] for e, n in zip(eastings, northings)])
    contours.append({"elevation": contour["elevation"], "paths": projected_paths})

print(f"  {len(contours)} contour levels")

# ---------------------------------------------------------------------------
# 3. Build wind field and trace streamlines
# ---------------------------------------------------------------------------
print(f"Building wind field  (SW {WIND_SPEED_KMH} km/h)…")
u_field, v_field = build_wind_field(
    elevation=elevation,
    wind_speed_kmh=WIND_SPEED_KMH,
    wind_direction_deg=WIND_DIR_DEG,
    bounds_projected=bounds_projected,
    grid_resolution=64,
    date_seed=20091231,
)

print(f"Tracing streamlines  ({GRID_ROWS}×{GRID_COLS} grid, max {STEPS} steps)…")
streamlines = trace_streamlines(
    u_field, v_field,
    bounds_projected=bounds_projected,
    steps=STEPS,
    wind_direction_deg=WIND_DIR_DEG,
    wind_speed_kmh=WIND_SPEED_KMH,
    seed=20091231,
    grid_rows=GRID_ROWS,
    grid_cols=GRID_COLS,
)
lens = [len(s) for s in streamlines]
print(f"  {len(streamlines)} arrows  |  steps: min={min(lens)}, mean={sum(lens)/len(lens):.1f}, max={max(lens)}")

print(f"Saving JSON → {OUT_JSON}")
save_streamlines_to_file(streamlines, OUT_JSON)

# ---------------------------------------------------------------------------
# 4. Render full SVG (terrain + route + wind)
# ---------------------------------------------------------------------------
proj_w = bounds_projected["max_x"] - bounds_projected["min_x"]
proj_h = bounds_projected["max_y"] - bounds_projected["min_y"]
if proj_w > proj_h:
    canvas_w_mm, canvas_h_mm = 24.0 * 25.4, 18.0 * 25.4
else:
    canvas_w_mm, canvas_h_mm = 18.0 * 25.4, 24.0 * 25.4

track_date = track.start_time.strftime("%Y-%m-%d") if track.start_time else "2009-12-31"

print("Composing SVG…")
OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
svg_path = render_terrain_svg(
    contours=contours,
    route_points=route_pts_utm,
    bounds=bounds_projected,
    output_path=str(OUT_SVG),
    width_mm=canvas_w_mm,
    height_mm=canvas_h_mm,
    margin_mm=_MARGIN_MM,
    bottom_margin_mm=_BOTTOM_MARGIN_MM,
    hillshade=hillshade,
    hillshade_transform=meta["transform"],
    track_name=track.name,
    date=track_date,
    distance_km=track.total_distance_km,
    elevation_gain_m=track.elevation_gain_m,
    centroid_lat=centroid_lat,
    centroid_lng=centroid_lng,
    wind_streamlines=streamlines,
)
print(f"  SVG → {svg_path}")

# ---------------------------------------------------------------------------
# 5. Export PNG preview to dated folder
# ---------------------------------------------------------------------------
today   = date.today().strftime("%Y-%m-%d")
ts      = datetime.now().strftime("%H%M%S")
out_dir = PREVIEW_ROOT / today
out_dir.mkdir(parents=True, exist_ok=True)
png_path = out_dir / f"wind_draft_{ts}.png"

try:
    import cairosvg
    cairosvg.svg2png(url=svg_path, write_to=str(png_path), dpi=150)
    print(f"  PNG → {png_path}")
except Exception as exc:
    print(f"  cairosvg failed ({exc}); falling back to matplotlib render…")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    min_x = bounds_projected["min_x"]
    max_x = bounds_projected["max_x"]
    min_y = bounds_projected["min_y"]
    max_y = bounds_projected["max_y"]

    fig_w = 10.0
    fig_h = fig_w * ((max_y - min_y) / (max_x - min_x))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal")
    ax.set_facecolor("#F5F0E8")
    fig.patch.set_facecolor("#F5F0E8")
    ax.axis("off")

    # Contours
    for c in contours:
        for path in c["paths"]:
            if len(path) < 2:
                continue
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, color="#888877", linewidth=0.3, alpha=0.5)

    # Route
    rxs = [p["x"] for p in route_pts_utm]
    rys = [p["y"] for p in route_pts_utm]
    ax.plot(rxs, rys, color="#1a6b8a", linewidth=1.5, alpha=0.9, solid_capstyle="round")

    # Wind arrows
    mean_speeds = []
    for stream in streamlines:
        spds = [s for _, _, s in stream]
        mean_speeds.append(sum(spds) / len(spds) if spds else 0.0)
    p90 = float(np.percentile(mean_speeds, 90)) if mean_speeds else 1.0
    p90 = max(p90, 1e-6)
    color_wind = "#2E3F50"
    for stream, ms in zip(streamlines, mean_speeds):
        if len(stream) < 2:
            continue
        speed_norm = min(1.0, ms / p90)
        alpha = 0.20 + 0.60 * speed_norm
        lw    = 0.5 + 0.9 * speed_norm
        xs = [pt[0] for pt in stream]
        ys = [pt[1] for pt in stream]
        ax.plot(xs, ys, color=color_wind, alpha=alpha, linewidth=lw, solid_capstyle="round")
        tx, ty = stream[-1][0], stream[-1][1]
        px, py = stream[-2][0], stream[-2][1]
        dx, dy = tx - px, ty - py
        dist = (dx**2 + dy**2) ** 0.5
        if dist > 1e-6:
            ux, uy = dx / dist, dy / dist
            hl = (max_x - min_x) * 0.007 * (0.5 + 0.5 * speed_norm)
            ax.annotate("", xy=(tx, ty),
                        xytext=(tx - ux * hl, ty - uy * hl),
                        arrowprops=dict(arrowstyle="-|>", color=color_wind,
                                        alpha=alpha, lw=lw,
                                        mutation_scale=6 + 4 * speed_norm))

    ax.text(0.98, 0.02, f"SW {int(WIND_SPEED_KMH)} km/h",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#2E3F50", alpha=0.7, fontfamily="monospace")

    plt.tight_layout(pad=0)
    fig.savefig(png_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  PNG → {png_path}")

print("Done.")
