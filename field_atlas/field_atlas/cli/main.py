"""main.py — Field Atlas command-line interface.

Entry-point that wires together the full GPX → DEM → enrichment → SVG pipeline.
Accepts a GPX file and produces a print-ready terrain map SVG.

Usage::

    python -m field_atlas render path/to/hike.gpx
    python -m field_atlas render path/to/hike.gpx --date 2024-06-15
    python -m field_atlas render path/to/hike.gpx --date 2024-06-15 --notes "First hike of summer"
"""

from __future__ import annotations

import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import click

from field_atlas.core.dem_fetcher import fetch_dem, load_dem
from field_atlas.core.gpx_parser import TrackData, padded_bounds, parse_gpx
from field_atlas.core.projection import get_projection, project_bounds, project_points
from field_atlas.core.terrain_processor import generate_contours, generate_hillshade
from field_atlas.enrichment.features import load_features_from_file
from field_atlas.enrichment.models import EnrichmentData, derive_location_name, enrich, format_info_block
from field_atlas.enrichment.weather import load_weather_from_file
from field_atlas.render.svg_composer import render_terrain_svg

# Default cache directory (relative to cwd, mirroring output/)
_CACHE_DIR = Path("cache")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """Convert a track name to a safe, lowercase filename stem."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_-]+", "_", name)
    return name.strip("_") or "track"


def _date_str(track: TrackData) -> str:
    """Return a formatted date string from the track's start time."""
    if track.start_time is not None:
        return track.start_time.strftime("%Y-%m-%d")
    return "Unknown"


def _run_enrichment(
    centroid_lat: float,
    centroid_lng: float,
    date: str,
    bounds: dict,
    route_points: list[dict],
    notes: str | None,
    weather_file: str | None,
    features_file: str | None,
    track_name: str = "",
) -> EnrichmentData:
    """Fetch enrichment data and apply any explicit file overrides.

    Calls :func:`~field_atlas.enrichment.models.enrich` for the live pipeline
    (weather, solar, features), then replaces individual fields when the user
    supplied a ``--weather-file`` or ``--features-file`` flag.  Each step is
    non-fatal; failures degrade gracefully to ``None``.
    """
    click.echo("Fetching enrichment data…")
    enrichment = enrich(
        centroid_lat,
        centroid_lng,
        date,
        bounds,
        route_points=route_points,
        timezone_str="America/New_York",
        notes=notes,
        track_name=track_name,
    )

    # --- honour explicit file overrides (bypass live API / cache) -----------
    if weather_file:
        try:
            enrichment.weather = load_weather_from_file(weather_file)
            click.echo(f"  Weather: loaded from {weather_file}")
        except Exception as exc:
            click.echo(f"  Weather: could not read {weather_file} — {exc}", err=True)

    if features_file:
        try:
            enrichment.features = load_features_from_file(features_file)
            click.echo(f"  Features: loaded from {features_file}")
            # Re-derive location name now that features are available.
            enrichment.location_name = derive_location_name(track_name, enrichment.features)
        except Exception as exc:
            click.echo(
                f"  Features: could not read {features_file} — {exc}", err=True
            )

    return enrichment


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """Field Atlas — print-quality terrain map generator."""


@cli.command()
@click.argument("gpx_file", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    "--output", "-o",
    default=None,
    metavar="PATH",
    help="Output SVG path.  Default: output/<track_name>.svg",
)
@click.option(
    "--resolution",
    default=1024,
    show_default=True,
    metavar="PIXELS",
    help=(
        "DEM pixel resolution (width = height).  Higher values (1024, 2048) "
        "produce better contour detail but take longer to fetch and process.  "
        "2048 is the maximum the USGS 3DEP API supports in a single request."
    ),
)
@click.option(
    "--contour-interval",
    default=20.0,
    show_default=True,
    metavar="METRES",
    help="Vertical interval between contour lines.",
)
@click.option(
    "--padding",
    default=0.2,
    show_default=True,
    metavar="FRACTION",
    help="Fractional padding added to each edge of the bounding box.",
)
@click.option(
    "--width",
    default=18.0,
    show_default=True,
    metavar="INCHES",
    help="Print canvas width.",
)
@click.option(
    "--height",
    default=24.0,
    show_default=True,
    metavar="INCHES",
    help="Print canvas height.",
)
@click.option(
    "--dem-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    metavar="PATH",
    help="Use a local GeoTIFF instead of fetching from USGS 3DEP.",
)
@click.option(
    "--weather-file",
    default=None,
    type=click.Path(dir_okay=False, readable=True),
    metavar="PATH",
    help="Use a local WeatherData JSON instead of fetching from Open-Meteo.",
)
@click.option(
    "--features-file",
    default=None,
    type=click.Path(dir_okay=False, readable=True),
    metavar="PATH",
    help="Use a local FeaturesData JSON instead of querying Overpass.",
)
@click.option(
    "--date",
    default=None,
    metavar="YYYY-MM-DD",
    help=(
        "Date for enrichment (weather, solar position, geographic features). "
        "When omitted the map is rendered terrain-only."
    ),
)
@click.option(
    "--notes",
    default=None,
    metavar="TEXT",
    help="User notes about the hike, stored in the enrichment record.",
)
def render(
    gpx_file: str,
    output: str | None,
    resolution: int,
    contour_interval: float,
    padding: float,
    width: float,
    height: float,
    dem_file: str | None,
    weather_file: str | None,
    features_file: str | None,
    date: str | None,
    notes: str | None,
) -> None:
    """Render a terrain map SVG from GPX_FILE.

    Without --date the map is terrain-only (contours + route).  Supply --date
    to activate the full enrichment pipeline: weather, solar position, and
    geographic feature labels are fetched and shown in the terminal summary
    (SVG integration comes in Phase 3).

    \b
      1. Parse GPX
      2. Compute padded bounding box
      3. Fetch enrichment data (if --date supplied)
      4. Auto-detect UTM projection from track centroid
      5. Fetch USGS 3DEP DEM for the bounding box
      6. Load and lightly smooth the elevation raster
      7. Generate contour lines
      8. Project route + bounds to UTM metres
      9. Compose and write the SVG
    """
    # ------------------------------------------------------------------
    # Step 1: Parse GPX
    # ------------------------------------------------------------------
    try:
        track = parse_gpx(gpx_file)
    except FileNotFoundError:
        click.echo(f"Error: GPX file not found: {gpx_file}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error: Could not parse GPX — {exc}", err=True)
        sys.exit(1)

    slug = _slugify(track.name)
    if output is None:
        output = str(Path("output") / f"{slug}.svg")

    # The display date: prefer the explicit --date flag, fall back to GPX
    # embedded time, then "Unknown".
    display_date = date if date else _date_str(track)

    # ------------------------------------------------------------------
    # Step 2: Padded bounding box (WGS84) — shared by DEM and features
    # ------------------------------------------------------------------
    bounds = padded_bounds(track, padding_pct=padding)

    centroid_lat = (track.bounds["min_lat"] + track.bounds["max_lat"]) / 2.0
    centroid_lng = (track.bounds["min_lng"] + track.bounds["max_lng"]) / 2.0

    # ------------------------------------------------------------------
    # Step 3: Enrichment (weather, solar, features) — only when --date given
    # ------------------------------------------------------------------
    enrichment: EnrichmentData | None = None
    if date:
        enrichment = _run_enrichment(
            centroid_lat, centroid_lng,
            date, bounds,
            route_points=track.points,   # list[dict] with "lat"/"lng" keys
            notes=notes,
            weather_file=weather_file,
            features_file=features_file,
            track_name=track.name,
        )

    # ------------------------------------------------------------------
    # Step 4: UTM projection from track centroid
    # ------------------------------------------------------------------
    transformer = get_projection(centroid_lat, centroid_lng)

    # ------------------------------------------------------------------
    # Steps 5–6: Obtain DEM (local file or remote fetch), load, smooth
    # ------------------------------------------------------------------
    if dem_file:
        click.echo(f"Loading local DEM: {dem_file}")
        try:
            elevation, meta = load_dem(dem_file)
        except Exception as exc:
            click.echo(f"Error: Could not load DEM file — {exc}", err=True)
            sys.exit(1)
    else:
        click.echo("Fetching elevation data…")
        try:
            with tempfile.TemporaryDirectory(prefix="field_atlas_") as tmp_dir:
                dem_path = str(Path(tmp_dir) / "dem.tif")
                fetch_dem(bounds, dem_path, resolution=resolution)
                elevation, meta = load_dem(dem_path)
        except Exception as exc:
            click.echo(f"Error: DEM fetch failed — {exc}", err=True)
            sys.exit(1)

    # Hillshade from the raw elevation (preserve_geology=True smoothing inside
    # generate_contours() is σ=0.3, gentle enough for hillshade too).
    hillshade = generate_hillshade(elevation)

    # ------------------------------------------------------------------
    # Step 7: Generate contour lines
    # ------------------------------------------------------------------
    click.echo("Generating contours…")
    contours = generate_contours(
        elevation,
        meta["transform"],
        interval_m=contour_interval,
    )

    # ------------------------------------------------------------------
    # Step 8: Project route points and bounds to UTM
    # ------------------------------------------------------------------
    route_points_utm = project_points(track.points, transformer)
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

    # ------------------------------------------------------------------
    # Step 9: Render SVG
    # ------------------------------------------------------------------
    click.echo("Composing SVG…")
    svg_path = render_terrain_svg(
        contours=contours,
        route_points=route_points_utm,
        bounds=projected_bounds,
        output_path=output,
        width_mm=width * 25.4,
        height_mm=height * 25.4,
        enrichment=enrichment,
        transformer=transformer,
        hillshade=hillshade,
        hillshade_transform=meta["transform"],
        track_name=track.name,
        date=display_date,
        distance_km=track.total_distance_km,
        elevation_gain_m=track.elevation_gain_m,
        centroid_lat=centroid_lat,
        centroid_lng=centroid_lng,
        duration_hours=track.duration_hours,
    )

    # ------------------------------------------------------------------
    # Step 10: Timestamped PNG preview (committed archive in docs/preview/)
    # ------------------------------------------------------------------
    import cairosvg
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    preview_dir = Path("docs") / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    png_path = str(preview_dir / f"{slug}_{ts}.png")
    try:
        # 960 px wide → 1280 px tall for an 18×24 canvas; compact for GitHub.
        cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=960)
    except Exception as exc:
        png_path = f"(failed — {exc})"

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_levels = len(contours)

    click.echo("")
    click.echo("✓ Field Atlas — Terrain Composition")
    click.echo(f"  Track:    {track.name}")
    click.echo(f"  Date:     {display_date}")
    click.echo(
        f"  Distance: {track.total_distance_km:.1f} km"
        f" | Gain: {track.elevation_gain_m:,.0f}m"
    )

    if enrichment is not None:
        info = format_info_block(enrichment, route_points=track.points)

        if info["weather_line"]:
            click.echo(f"  Weather:  {info['weather_line']}")
        else:
            click.echo("  Weather:  (not available)")

        if info["solar_line"]:
            click.echo(f"  Sun:      {info['solar_line']}")
        else:
            click.echo("  Sun:      (not available)")

        labels = info["feature_labels"]
        if labels:
            n = len(labels)
            preview = ", ".join(lb["name"] for lb in labels[:3])
            if n > 3:
                preview += ", …"
            click.echo(f"  Features: {n} labeled ({preview})")
        else:
            click.echo("  Features: (none)")

    click.echo(f"  Contours: {n_levels} lines at {contour_interval:.0f}m interval")
    click.echo(f"  Output:   {svg_path}")
    click.echo(f"  Preview:  {png_path}")


# ---------------------------------------------------------------------------
# Direct execution: python -m field_atlas.cli.main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
