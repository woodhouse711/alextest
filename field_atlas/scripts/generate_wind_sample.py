"""generate_wind_sample.py — Botany Bay full-map wind layer preview.

Runs the complete GPX → DEM → terrain → wind pipeline for Botany Bay
and writes a print-preview PNG.  Data sources are all free and open-access:

  DEM:     USGS 3DEP (US) → AWS Terrain Tiles / SRTM (global fallback)
  Weather: Open-Meteo Archive API  (no key required, 5-day lag)
  OSM:     Not fetched here; pass --vectors-file to the CLI for that layer

The script tries to fetch live data on every run and caches results to
sample_data/.  Pass --offline to skip network requests and use the cache.

Outputs
-------
  • sample_data/dem_botany_bay.tif              — cached DEM (overwritten on fetch)
  • sample_data/weather_botany_bay_<date>.json  — cached weather
  • sample_data/wind_streamlines_botany_bay.json — streamline cache
  • output/botany_bay_wind.svg                  — full-map SVG
  • docs/preview/<YYYY-MM-DD>/wind_draft_<HHMMSS>.png — PNG preview

Usage::

    cd field_atlas
    python scripts/generate_wind_sample.py            # fetch fresh data
    python scripts/generate_wind_sample.py --offline  # use cache only
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from field_atlas.core.dem_fetcher import fetch_dem_global, load_dem
from field_atlas.core.gpx_parser import padded_bounds, parse_gpx
from field_atlas.core.projection import get_projection, project_bounds, project_points
from field_atlas.core.terrain_processor import generate_contours, generate_hillshade
from field_atlas.enrichment.weather import fetch_weather, load_weather_from_file, save_weather_to_file
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

# Fallback wind params — used only if weather fetch fails and no cache exists.
_FALLBACK_WIND_SPEED_KMH = 19.0
_FALLBACK_WIND_DIR_DEG   = 225.0

# Grid: 24×24 = 576 seeds; max 12 integration steps per arrow (min 3).
GRID_ROWS = 24
GRID_COLS = 24
STEPS     = 12

_MARGIN_MM        = 1.5 * 25.4   # 38.1 mm
_BOTTOM_MARGIN_MM = 3.0 * 25.4   # 76.2 mm

# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--offline", action="store_true",
                    help="Skip all network requests; use cached files only.")
args = parser.parse_args()

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

print(f"  Bounds (WGS84): lat [{bounds_wgs84['min_lat']:.4f}, {bounds_wgs84['max_lat']:.4f}]"
      f"  lng [{bounds_wgs84['min_lng']:.4f}, {bounds_wgs84['max_lng']:.4f}]")
print(f"  Bounds (UTM):   x [{bounds_projected['min_x']:.0f}, {bounds_projected['max_x']:.0f}]"
      f"  y [{bounds_projected['min_y']:.0f}, {bounds_projected['max_y']:.0f}]")

# ---------------------------------------------------------------------------
# 1b. Fetch weather — Open-Meteo Archive (free, no auth, ~1940-present)
# ---------------------------------------------------------------------------
track_date = track.start_time.strftime("%Y-%m-%d") if track.start_time else "2009-12-31"
weather_cache = SAMPLE_DIR / f"weather_botany_bay_{track_date}.json"

wind_speed_kmh = _FALLBACK_WIND_SPEED_KMH
wind_dir_deg   = _FALLBACK_WIND_DIR_DEG
weather_source = "fallback defaults"

if not args.offline:
    print(f"Fetching weather  ({track_date}, Open-Meteo Archive)…")
    try:
        weather = fetch_weather(centroid_lat, centroid_lng, track_date)
        save_weather_to_file(weather, weather_cache)
        wind_speed_kmh = weather.wind_speed_max_kmh
        wind_dir_deg   = weather.wind_direction_dominant_deg
        weather_source = f"Open-Meteo Archive  ({weather.wind_direction_cardinal} {wind_speed_kmh:.0f} km/h)"
        print(f"  {weather.conditions_summary}")
    except Exception as exc:
        print(f"  Weather fetch failed: {exc}")
        if weather_cache.exists():
            print(f"  Falling back to cached weather: {weather_cache.name}")
            try:
                weather = load_weather_from_file(weather_cache)
                wind_speed_kmh = weather.wind_speed_max_kmh
                wind_dir_deg   = weather.wind_direction_dominant_deg
                weather_source = f"cached ({weather.wind_direction_cardinal} {wind_speed_kmh:.0f} km/h)"
            except Exception:
                print(f"  Cache read failed too; using fallback defaults.")
        else:
            print(f"  No cache found; using fallback defaults ({_FALLBACK_WIND_SPEED_KMH} km/h SW).")
elif weather_cache.exists():
    print(f"Offline mode — loading cached weather…")
    try:
        weather = load_weather_from_file(weather_cache)
        wind_speed_kmh = weather.wind_speed_max_kmh
        wind_dir_deg   = weather.wind_direction_dominant_deg
        weather_source = f"cached ({weather.wind_direction_cardinal} {wind_speed_kmh:.0f} km/h)"
        print(f"  {weather.conditions_summary}")
    except Exception as exc:
        print(f"  Cache read failed ({exc}); using defaults.")
else:
    print(f"Offline mode — no weather cache; using fallback defaults.")

print(f"  Wind source: {weather_source}")

# ---------------------------------------------------------------------------
# 2. Fetch or load DEM
#    Sources tried in order (all free, open-access, no authentication):
#      1. USGS 3DEP        — https://nationalmap.gov  (US only)
#      2. AWS Terrain Tiles — https://registry.opendata.aws/terrain-tiles/
#         (global SRTM/NASADEM, Terrarium RGB encoding)
# ---------------------------------------------------------------------------
dem_source = "cache"

if not args.offline:
    print(f"Fetching DEM  (USGS 3DEP → AWS Terrain Tiles fallback)…")
    try:
        source = fetch_dem_global(bounds_wgs84, str(DEM_PATH))
        dem_source = source
        print(f"  DEM fetched from: {source.upper()}")
    except Exception as exc:
        print(f"  All DEM sources failed: {exc}")
        if DEM_PATH.exists():
            print(f"  Falling back to cached DEM: {DEM_PATH.name}")
            dem_source = "cache"
        else:
            print("  No cached DEM and all fetches failed — cannot continue.")
            sys.exit(1)
else:
    if not DEM_PATH.exists():
        print(f"Offline mode — no cached DEM at {DEM_PATH}; cannot continue.")
        sys.exit(1)
    print(f"Offline mode — using cached DEM.")

print("Loading DEM…")
elevation, meta = load_dem(str(DEM_PATH))
hillshade = generate_hillshade(elevation)
print(f"  Shape: {elevation.shape}  |  source: {dem_source}"
      f"  |  elevation range: {float(elevation[~__import__('numpy').isnan(elevation)].min()):.0f}–"
      f"{float(elevation[~__import__('numpy').isnan(elevation)].max()):.0f} m")

# ---------------------------------------------------------------------------
# 3. Contours
# ---------------------------------------------------------------------------
print("Generating contours (20 m interval)…")
contours_raw = generate_contours(elevation, meta["transform"], interval_m=20.0)

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
# 4. Wind field and streamlines
# ---------------------------------------------------------------------------
_cardinal = {0:"N",45:"NE",90:"E",135:"SE",180:"S",225:"SW",270:"W",315:"NW"}
_card = min(_cardinal, key=lambda d: abs(d - wind_dir_deg % 360))
print(f"Building wind field  ({_cardinal[_card]} {wind_speed_kmh:.0f} km/h  |  {wind_dir_deg:.0f}°)…")

date_seed = int(track_date.replace("-", ""))
u_field, v_field = build_wind_field(
    elevation=elevation,
    wind_speed_kmh=wind_speed_kmh,
    wind_direction_deg=wind_dir_deg,
    bounds_projected=bounds_projected,
    grid_resolution=64,
    date_seed=date_seed,
)

print(f"Tracing streamlines  ({GRID_ROWS}×{GRID_COLS} grid, max {STEPS} steps)…")
streamlines = trace_streamlines(
    u_field, v_field,
    bounds_projected=bounds_projected,
    steps=STEPS,
    wind_direction_deg=wind_dir_deg,
    wind_speed_kmh=wind_speed_kmh,
    seed=date_seed,
    grid_rows=GRID_ROWS,
    grid_cols=GRID_COLS,
)
lens = [len(s) for s in streamlines]
print(f"  {len(streamlines)} arrows  |  steps: min={min(lens)}, mean={sum(lens)/len(lens):.1f}, max={max(lens)}")

print(f"Saving JSON → {OUT_JSON}")
save_streamlines_to_file(streamlines, OUT_JSON)

# ---------------------------------------------------------------------------
# 5. Render SVG
# ---------------------------------------------------------------------------
proj_w = bounds_projected["max_x"] - bounds_projected["min_x"]
proj_h = bounds_projected["max_y"] - bounds_projected["min_y"]
if proj_w > proj_h:
    canvas_w_mm, canvas_h_mm = 24.0 * 25.4, 18.0 * 25.4
else:
    canvas_w_mm, canvas_h_mm = 18.0 * 25.4, 24.0 * 25.4

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
# 6. Export PNG preview
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

    for c in contours:
        for path in c["paths"]:
            if len(path) < 2:
                continue
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, color="#888877", linewidth=0.3, alpha=0.5)

    rxs = [p["x"] for p in route_pts_utm]
    rys = [p["y"] for p in route_pts_utm]
    ax.plot(rxs, rys, color="#1a6b8a", linewidth=1.5, alpha=0.9, solid_capstyle="round")

    mean_speeds = []
    for stream in streamlines:
        spds = [s for _, _, s in stream]
        mean_speeds.append(sum(spds) / len(spds) if spds else 0.0)
    p90 = float(np.percentile(mean_speeds, 90)) if mean_speeds else 1.0
    p90 = max(p90, 1e-6)
    for stream, ms in zip(streamlines, mean_speeds):
        if len(stream) < 2:
            continue
        speed_norm = min(1.0, ms / p90)
        alpha = 0.20 + 0.60 * speed_norm
        lw    = 0.5 + 0.9 * speed_norm
        xs = [pt[0] for pt in stream]
        ys = [pt[1] for pt in stream]
        ax.plot(xs, ys, color="#C6DDE8", alpha=alpha, linewidth=lw, solid_capstyle="round")

    ax.text(0.98, 0.02,
            f"{_cardinal[_card]} {wind_speed_kmh:.0f} km/h  ·  {weather_source}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#2E3F50", alpha=0.7, fontfamily="monospace")

    plt.tight_layout(pad=0)
    fig.savefig(png_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  PNG → {png_path}")

print("Done.")
