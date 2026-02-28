"""main.py — Field Atlas command-line interface.

Entry-point that wires together the full GPX → DEM → SVG pipeline.
Accepts a GPX file and produces a print-ready terrain map SVG.

Usage::

    python -m field_atlas render path/to/hike.gpx
    python -m field_atlas.cli.main render path/to/hike.gpx
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import click

from field_atlas.core.dem_fetcher import fetch_dem, load_dem
from field_atlas.core.gpx_parser import TrackData, padded_bounds, parse_gpx
from field_atlas.core.projection import get_projection, project_bounds, project_points
from field_atlas.core.terrain_processor import generate_contours, smooth_elevation
from field_atlas.enrichment.features import (
    FeatureSet,
    fetch_features,
    format_features_line,
    load_features_from_file,
    save_features_to_file,
)
from field_atlas.enrichment.weather import (
    WeatherData,
    fetch_weather,
    load_weather_from_file,
    save_weather_to_file,
)
from field_atlas.render.svg_composer import render_terrain_svg

# Default cache directory (relative to cwd, mirroring output/)
_CACHE_DIR = Path("cache")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_weather(
    lat: float,
    lng: float,
    date: str,
    weather_file: str | None,
) -> WeatherData | None:
    """Try to obtain weather data in priority order:
    1. Explicit --weather-file flag
    2. Live Open-Meteo fetch (auto-saved to cache/ on success)
    3. Cached file in cache/
    Returns None if all sources fail (non-fatal; enrichment is skipped).
    """
    cache_path = _CACHE_DIR / f"weather_{date}.json"

    # 1. Explicit file
    if weather_file:
        try:
            w = load_weather_from_file(weather_file)
            click.echo(f"Weather: loaded from {weather_file}")
            return w
        except Exception as exc:
            click.echo(f"Weather: could not read {weather_file} — {exc}", err=True)
            return None

    # 2. Live fetch
    try:
        w = fetch_weather(lat, lng, date)
        save_weather_to_file(w, cache_path)
        click.echo(f"Weather: fetched from Open-Meteo (cached to {cache_path})")
        return w
    except Exception as exc:
        click.echo(f"Weather: API unavailable — {exc}")

    # 3. Cache fallback
    if cache_path.exists():
        try:
            w = load_weather_from_file(cache_path)
            click.echo(f"Weather: loaded from cache {cache_path}")
            return w
        except Exception as exc:
            click.echo(f"Weather: cache unreadable — {exc}", err=True)

    click.echo("Weather: API unavailable, no cache found — skipping")
    return None


def _load_features(
    bounds: dict,
    date: str,
    features_file: str | None,
) -> FeatureSet | None:
    """Try to obtain features data in priority order:
    1. Explicit --features-file flag
    2. Live Overpass fetch (auto-saved to cache/ on success)
    3. Cached file in cache/
    Returns None if all sources fail (non-fatal; enrichment is skipped).
    """
    cache_path = _CACHE_DIR / f"features_{date}.json"

    # 1. Explicit file
    if features_file:
        try:
            f = load_features_from_file(features_file)
            click.echo(f"Features: loaded from {features_file}")
            return f
        except Exception as exc:
            click.echo(f"Features: could not read {features_file} — {exc}", err=True)
            return None

    # 2. Live fetch
    try:
        f = fetch_features(bounds)
        save_features_to_file(f, cache_path)
        click.echo(f"Features: fetched from Overpass (cached to {cache_path})")
        return f
    except Exception as exc:
        click.echo(f"Features: API unavailable — {exc}")

    # 3. Cache fallback
    if cache_path.exists():
        try:
            f = load_features_from_file(cache_path)
            click.echo(f"Features: loaded from cache {cache_path}")
            return f
        except Exception as exc:
            click.echo(f"Features: cache unreadable — {exc}", err=True)

    click.echo("Features: API unavailable, no cache found — skipping")
    return None


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
    default=512,
    show_default=True,
    metavar="PIXELS",
    help="DEM pixel resolution (width = height).",
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
) -> None:
    """Render a terrain map SVG from GPX_FILE.

    The full pipeline:

    \b
      1. Parse GPX
      2. Compute padded bounding box
      3. Auto-detect UTM projection from track centroid
      4. Fetch USGS 3DEP DEM for the bounding box
      5. Load and lightly smooth the elevation raster
      6. Generate contour lines
      7. Project route + bounds to UTM metres
      8. Compose and write the SVG
    """
    # ------------------------------------------------------------------
    # Step 1: Parse GPX
    # ------------------------------------------------------------------
    try:
        track = parse_gpx(gpx_file)
    except FileNotFoundError:
        click.echo(f"Error: GPX file not found: {gpx_file}", err=True)
        sys.exit(1)
    except Exception as exc:  # gpxpy.gpx.GPXException, ValueError, etc.
        click.echo(f"Error: Could not parse GPX — {exc}", err=True)
        sys.exit(1)

    # Derive default output path from the track name.
    slug = _slugify(track.name)
    if output is None:
        output = str(Path("output") / f"{slug}.svg")

    date = _date_str(track)

    # ------------------------------------------------------------------
    # Step 2: Padded bounding box (WGS84) — needed by both DEM and features
    # ------------------------------------------------------------------
    bounds = padded_bounds(track, padding_pct=padding)

    centroid_lat = (track.bounds["min_lat"] + track.bounds["max_lat"]) / 2.0
    centroid_lng = (track.bounds["min_lng"] + track.bounds["max_lng"]) / 2.0

    # ------------------------------------------------------------------
    # Step 3: Enrichment data (weather + features) — non-fatal if absent
    # ------------------------------------------------------------------
    weather = _load_weather(centroid_lat, centroid_lng, date, weather_file)
    features = _load_features(bounds, date, features_file)

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
                # Both objects are now fully in memory; the temp file can go.
        except Exception as exc:
            click.echo(f"Error: DEM fetch failed — {exc}", err=True)
            sys.exit(1)

    elevation = smooth_elevation(elevation, sigma=0.8)

    # ------------------------------------------------------------------
    # Step 6: Generate contour lines
    # ------------------------------------------------------------------
    click.echo("Generating contours…")  # Step 7
    contours = generate_contours(
        elevation,
        meta["transform"],
        interval_m=contour_interval,
    )

    # ------------------------------------------------------------------
    # Step 7: Project route points and bounds to UTM
    # ------------------------------------------------------------------
    route_points = project_points(track.points, transformer)
    projected_bounds = project_bounds(bounds, transformer)

    # Project contour paths from the DEM's CRS (WGS84 degrees) to UTM metres
    # so they share the same coordinate system as the route and bounds.
    # Each path point is [lng, lat] (x=east, y=north in geographic convention).
    for contour in contours:
        projected_paths = []
        for path in contour["paths"]:
            if not path:
                continue
            lats = [xy[1] for xy in path]
            lngs = [xy[0] for xy in path]
            # EPSG:326XX axis order: (Easting, Northing)
            eastings, northings = transformer.transform(lats, lngs)
            projected_paths.append(
                [[float(e), float(n)] for e, n in zip(eastings, northings)]
            )
        contour["paths"] = projected_paths

    # ------------------------------------------------------------------
    # Step 8: Render SVG
    # ------------------------------------------------------------------
    click.echo("Composing SVG…")
    svg_path = render_terrain_svg(
        contours=contours,
        route_points=route_points,
        bounds=projected_bounds,
        output_path=output,
        width_mm=width * 25.4,
        height_mm=height * 25.4,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_levels = len(contours)

    click.echo("")
    click.echo("✓ Field Atlas — Terrain Composition")
    click.echo(f"  Track:    {track.name}")
    click.echo(f"  Date:     {date}")
    click.echo(
        f"  Distance: {track.total_distance_km:.1f} km"
        f" | Gain: {track.elevation_gain_m:,.0f}m"
    )
    click.echo(f"  Contours: {n_levels} lines at {contour_interval:.0f}m interval")

    if weather:
        from field_atlas.enrichment.weather import format_weather_line
        click.echo(f"  Weather:  {format_weather_line(weather)}")
    else:
        click.echo("  Weather:  (not available)")

    if features:
        click.echo(f"  Features: {format_features_line(features)}")
    else:
        click.echo("  Features: (not available)")

    click.echo(f"  Output:   {svg_path}")


# ---------------------------------------------------------------------------
# Direct execution: python -m field_atlas.cli.main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
