"""
mock_speed_render.py — Visualise the speed-color gradient with synthetic data.

Injects random speeds (4–12 km/h) for every segment of sample.gpx (the
BELLEVUE survey route, which has no real GPS timestamps) so we can verify the
maroon-to-turquoise palette before real-timed recordings are available.

Run from the field_atlas/ project root:
    python tests/mock_speed_render.py
"""

from __future__ import annotations

import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Make sure the package is importable when run from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from field_atlas.core.gpx_parser import parse_gpx, padded_bounds
from field_atlas.core.dem_fetcher import load_dem
from field_atlas.core.projection import get_projection, project_bounds, project_points
from field_atlas.core.terrain_processor import generate_contours, generate_hillshade
from field_atlas.render.svg_composer import render_terrain_svg

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------

GPX_FILE = "sample_data/sample.gpx"
DEM_FILE = "sample_data/dem.tif"
OUT_SVG  = "output/sample_mock_speed.svg"
PREVIEW  = "docs/preview"

_MARGIN_MM        = 1.5 * 25.4   # 38.1 mm
_BOTTOM_MARGIN_MM = 3.0 * 25.4   # 76.2 mm

random.seed(42)


def main() -> None:
    print(f"Parsing {GPX_FILE}…")
    track = parse_gpx(GPX_FILE)
    n_pts = len(track.points)
    n_segs = n_pts - 1
    print(f"  {track.name}: {n_pts} points, {n_segs} segments")

    # Inject random speeds: smooth-ish ramp + jitter so the gradient is visible.
    mock_speeds: list[float] = []
    for i in range(n_segs):
        base = 4.0 + 8.0 * (i / max(1, n_segs - 1))   # 4 → 12 km/h ramp
        jitter = random.uniform(-1.5, 1.5)
        mock_speeds.append(max(0.5, base + jitter))

    print(f"  Mock speeds: {min(mock_speeds):.1f} — {max(mock_speeds):.1f} km/h")

    # Standard pipeline ---------------------------------------------------
    bounds = padded_bounds(track)
    centroid_lat = (track.bounds["min_lat"] + track.bounds["max_lat"]) / 2.0
    centroid_lng = (track.bounds["min_lng"] + track.bounds["max_lng"]) / 2.0

    transformer = get_projection(centroid_lat, centroid_lng)

    print(f"Loading DEM: {DEM_FILE}…")
    elevation, meta = load_dem(DEM_FILE)
    hillshade = generate_hillshade(elevation)

    print("Generating contours…")
    contours = generate_contours(elevation, meta["transform"], interval_m=20.0)

    route_pts_utm = project_points(track.points, transformer)
    projected_bounds = project_bounds(bounds, transformer)

    for contour in contours:
        projected_paths = []
        for path in contour["paths"]:
            if not path:
                continue
            lats = [xy[1] for xy in path]
            lngs = [xy[0] for xy in path]
            eastings, northings = transformer.transform(lats, lngs)
            projected_paths.append(
                [[float(e), float(n)] for e, n in zip(eastings, northings)]
            )
        contour["paths"] = projected_paths

    proj_w = projected_bounds["max_x"] - projected_bounds["min_x"]
    proj_h = projected_bounds["max_y"] - projected_bounds["min_y"]
    if proj_w > proj_h:
        canvas_w_mm, canvas_h_mm = 24.0 * 25.4, 18.0 * 25.4
    else:
        canvas_w_mm, canvas_h_mm = 18.0 * 25.4, 24.0 * 25.4

    print("Composing SVG…")
    svg_path = render_terrain_svg(
        contours=contours,
        route_points=route_pts_utm,
        bounds=projected_bounds,
        output_path=OUT_SVG,
        width_mm=canvas_w_mm,
        height_mm=canvas_h_mm,
        margin_mm=_MARGIN_MM,
        bottom_margin_mm=_BOTTOM_MARGIN_MM,
        hillshade=hillshade,
        hillshade_transform=meta["transform"],
        track_name=track.name,
        date="mock",
        distance_km=track.total_distance_km,
        elevation_gain_m=track.elevation_gain_m,
        centroid_lat=centroid_lat,
        centroid_lng=centroid_lng,
        segment_speeds=mock_speeds,
        route_palette="maroon_turquoise",
    )

    # PDF export
    try:
        import cairosvg
        Path(PREVIEW).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = f"{PREVIEW}/sample_mock_speed_{ts}.pdf"
        cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
        print(f"  PDF:    {pdf_path}")
    except Exception as exc:
        print(f"  PDF export failed: {exc}")

    print(f"\n✓ Mock speed render complete")
    print(f"  SVG:    {svg_path}")


if __name__ == "__main__":
    main()
