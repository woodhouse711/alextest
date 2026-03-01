"""generate_wind_sample.py — Pre-compute wind streamlines for Botany Bay.

Loads the local DEM and GPX to recover bounds, then runs the wind field
builder and streamline tracer with a SW 19 km/h wind, and writes the result
to sample_data/wind_streamlines_botany_bay.json.

Usage::

    cd field_atlas
    python scripts/generate_wind_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parents[1]))

from field_atlas.core.dem_fetcher import load_dem
from field_atlas.core.gpx_parser import padded_bounds, parse_gpx
from field_atlas.core.projection import get_projection, project_bounds
from field_atlas.enrichment.wind import (
    build_wind_field,
    save_streamlines_to_file,
    trace_streamlines,
)

SAMPLE_DIR = Path(__file__).parents[1] / "sample_data"
GPX_PATH   = SAMPLE_DIR / "Botany_Bay.gpx"
DEM_PATH   = SAMPLE_DIR / "dem_botany_bay.tif"
OUT_PATH   = SAMPLE_DIR / "wind_streamlines_botany_bay.json"

# SW wind at 19 km/h (meteorological: wind comes FROM SW = 225°)
WIND_SPEED_KMH = 19.0
WIND_DIR_DEG   = 225.0   # FROM south-west
N_STREAMLINES  = 60

print("Loading GPX…")
track = parse_gpx(str(GPX_PATH))
bounds_wgs84 = padded_bounds(track, padding_pct=0.2)

centroid_lat = (track.bounds["min_lat"] + track.bounds["max_lat"]) / 2.0
centroid_lng = (track.bounds["min_lng"] + track.bounds["max_lng"]) / 2.0

transformer      = get_projection(centroid_lat, centroid_lng)
bounds_projected = project_bounds(bounds_wgs84, transformer)

print(f"Bounds (UTM):  x [{bounds_projected['min_x']:.1f}, {bounds_projected['max_x']:.1f}]"
      f"  y [{bounds_projected['min_y']:.1f}, {bounds_projected['max_y']:.1f}]")

print("Loading DEM…")
elevation, _meta = load_dem(str(DEM_PATH))
print(f"  Elevation array shape: {elevation.shape}")

print(f"Building wind field  (SW {WIND_SPEED_KMH} km/h)…")
u_field, v_field = build_wind_field(
    elevation=elevation,
    wind_speed_kmh=WIND_SPEED_KMH,
    wind_direction_deg=WIND_DIR_DEG,
    bounds_projected=bounds_projected,
    grid_resolution=64,
    date_seed=20091231,   # Botany Bay track date
)
print(f"  u ∈ [{u_field.min():.2f}, {u_field.max():.2f}] km/h")
print(f"  v ∈ [{v_field.min():.2f}, {v_field.max():.2f}] km/h")

print(f"Tracing {N_STREAMLINES} streamlines…")
streamlines = trace_streamlines(
    u_field, v_field,
    bounds_projected=bounds_projected,
    num_streamlines=N_STREAMLINES,
    wind_direction_deg=WIND_DIR_DEG,
    wind_speed_kmh=WIND_SPEED_KMH,
    seed=20091231,
)
print(f"  Traced {len(streamlines)} streamlines  "
      f"(avg {sum(len(s) for s in streamlines)/max(1,len(streamlines)):.1f} pts each)")

print(f"Saving → {OUT_PATH}")
save_streamlines_to_file(streamlines, OUT_PATH)
print("Done.")
