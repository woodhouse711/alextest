"""generate_wind_sample.py — Pre-compute wind streamlines for Botany Bay.

Loads the local DEM and GPX to recover bounds, then runs the wind field
builder and streamline tracer with a SW 19 km/h wind, and writes:

  • sample_data/wind_streamlines_botany_bay.json  — streamline data
  • docs/preview/<YYYY-MM-DD>/wind_draft_<HHMMSS>.png — visual preview

Grid seeding mode: seeds are placed on a regular NxN grid across the
whole map extent, each producing a short arrow (~1 grid cell long), giving
a uniform direction-field rather than long edge-to-edge flow lines.

Usage::

    cd field_atlas
    python scripts/generate_wind_sample.py
"""

from __future__ import annotations

import sys
from datetime import date, datetime
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

PREVIEW_ROOT = Path(__file__).parents[1] / "docs" / "preview"

# SW wind at 19 km/h (meteorological: wind comes FROM SW = 225°)
WIND_SPEED_KMH = 19.0
WIND_DIR_DEG   = 225.0   # FROM south-west

# Dense grid: 24 cols × 24 rows = 576 short arrows covering the whole map.
GRID_ROWS = 24
GRID_COLS = 24
STEPS     = 10   # integration steps per arrow (keeps each arrow ≈ 1 cell long)

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

print(f"Tracing streamlines  ({GRID_ROWS}×{GRID_COLS} grid, {STEPS} steps each)…")
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
print(f"  Traced {len(streamlines)} streamlines  "
      f"(avg {sum(len(s) for s in streamlines)/max(1,len(streamlines)):.1f} pts each)")

print(f"Saving streamlines → {OUT_PATH}")
save_streamlines_to_file(streamlines, OUT_PATH)

# ---------------------------------------------------------------------------
# PNG preview — matplotlib quiver-style render of the traced arrows
# ---------------------------------------------------------------------------
print("Rendering PNG preview…")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

min_x = bounds_projected["min_x"]
max_x = bounds_projected["max_x"]
min_y = bounds_projected["min_y"]
max_y = bounds_projected["max_y"]
proj_w = max_x - min_x
proj_h = max_y - min_y

fig_w = 10.0
fig_h = fig_w * (proj_h / proj_w)
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)
ax.set_aspect("equal")
ax.set_facecolor("#F5F0E8")
fig.patch.set_facecolor("#F5F0E8")
ax.axis("off")

# Collect mean speed per streamline for opacity scaling.
mean_speeds = []
for stream in streamlines:
    spds = [s for _, _, s in stream]
    mean_speeds.append(sum(spds) / len(spds) if spds else 0.0)

p90 = float(np.percentile(mean_speeds, 90)) if mean_speeds else 1.0
p90 = max(p90, 1e-6)

color = "#2E3F50"

for stream, ms in zip(streamlines, mean_speeds):
    if len(stream) < 2:
        continue

    speed_norm = min(1.0, ms / p90)
    alpha = 0.15 + 0.65 * speed_norm
    lw    = 0.4 + 0.8 * speed_norm

    xs = [pt[0] for pt in stream]
    ys = [pt[1] for pt in stream]
    ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw, solid_capstyle="round")

    # Small arrowhead at the terminus.
    if len(stream) >= 2:
        tx, ty = stream[-1][0], stream[-1][1]
        px, py = stream[-2][0], stream[-2][1]
        dx, dy = tx - px, ty - py
        dist = (dx**2 + dy**2) ** 0.5
        if dist > 1e-6:
            ux, uy = dx / dist, dy / dist
            head_len = proj_w * 0.008 * (0.5 + 0.5 * speed_norm)
            ax.annotate(
                "",
                xy=(tx, ty),
                xytext=(tx - ux * head_len, ty - uy * head_len),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    alpha=alpha,
                    lw=lw,
                    mutation_scale=6 + 4 * speed_norm,
                ),
            )

# Wind label
ax.text(
    0.98, 0.02,
    f"SW {int(WIND_SPEED_KMH)} km/h",
    transform=ax.transAxes,
    ha="right", va="bottom",
    fontsize=9, color="#2E3F50", alpha=0.7,
    fontfamily="monospace",
)

plt.tight_layout(pad=0)

# Save to dated preview folder.
today     = date.today().strftime("%Y-%m-%d")
ts        = datetime.now().strftime("%H%M%S")
out_dir   = PREVIEW_ROOT / today
out_dir.mkdir(parents=True, exist_ok=True)
png_path  = out_dir / f"wind_draft_{ts}.png"

fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  PNG saved → {png_path}")

print("Done.")
