"""
gpx_parser.py — Parse GPX files into structured TrackData.

Extracts trackpoints, computes distances via haversine, and summarises
elevation gain/loss and bounding geometry.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import gpxpy
import gpxpy.gpx


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrackData:
    """All derived data extracted from a single GPX track."""

    name: str
    points: list[dict]                  # keys: lat, lng, elevation, time
    bounds: dict                        # min_lat, max_lat, min_lng, max_lng
    total_distance_km: float
    elevation_gain_m: float
    elevation_loss_m: float
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration_hours: Optional[float]

    def __str__(self) -> str:
        dur = f"{self.duration_hours:.2f} h" if self.duration_hours is not None else "n/a"
        return (
            f'TrackData("{self.name}")\n'
            f"  Points        : {len(self.points)}\n"
            f"  Distance      : {self.total_distance_km:.3f} km\n"
            f"  Elev gain     : {self.elevation_gain_m:.1f} m\n"
            f"  Elev loss     : {self.elevation_loss_m:.1f} m\n"
            f"  Bounds        : "
            f"({self.bounds['min_lat']:.5f}, {self.bounds['min_lng']:.5f}) -> "
            f"({self.bounds['max_lat']:.5f}, {self.bounds['max_lng']:.5f})\n"
            f"  Start         : {self.start_time}\n"
            f"  End           : {self.end_time}\n"
            f"  Duration      : {dur}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EARTH_RADIUS_KM = 6371.0


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Return great-circle distance in kilometres between two WGS-84 points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_gpx(filepath: str) -> TrackData:
    """Parse a GPX file and return a populated :class:`TrackData`.

    Parameters
    ----------
    filepath:
        Path to the ``.gpx`` file to parse.

    Returns
    -------
    TrackData
        Fully populated track summary.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    gpxpy.gpx.GPXException
        If the file is not valid GPX.
    ValueError
        If the file contains no trackpoints.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"GPX file not found: {filepath}")

    with path.open("r", encoding="utf-8") as fh:
        gpx = gpxpy.parse(fh)

    # --- collect raw points (tracks first, fall back to routes) --------------
    points: list[dict] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                points.append(
                    {
                        "lat": pt.latitude,
                        "lng": pt.longitude,
                        "elevation": pt.elevation,   # metres, may be None
                        "time": pt.time,             # datetime or None
                    }
                )

    if not points:
        for route in gpx.routes:
            for pt in route.points:
                points.append(
                    {
                        "lat": pt.latitude,
                        "lng": pt.longitude,
                        "elevation": pt.elevation,
                        "time": pt.time,
                    }
                )

    if not points:
        raise ValueError(f"No trackpoints or route points found in {filepath}")

    # --- track/route name ----------------------------------------------------
    track_name = None
    if gpx.tracks and gpx.tracks[0].name:
        track_name = gpx.tracks[0].name
    elif gpx.routes and gpx.routes[0].name:
        track_name = gpx.routes[0].name
    track_name = track_name or path.stem

    # --- bounding box --------------------------------------------------------
    lats = [p["lat"] for p in points]
    lngs = [p["lng"] for p in points]
    bounds = {
        "min_lat": min(lats),
        "max_lat": max(lats),
        "min_lng": min(lngs),
        "max_lng": max(lngs),
    }

    # --- cumulative distance (haversine) -------------------------------------
    total_distance_km = 0.0
    for i in range(1, len(points)):
        total_distance_km += _haversine(
            points[i - 1]["lat"], points[i - 1]["lng"],
            points[i]["lat"],     points[i]["lng"],
        )

    # --- elevation gain / loss (noise-filtered at > 2 m) --------------------
    ELEV_THRESHOLD_M = 2.0
    elevation_gain_m = 0.0
    elevation_loss_m = 0.0

    elevations = [p["elevation"] for p in points if p["elevation"] is not None]
    if len(elevations) >= 2:
        pending: float = 0.0
        for i in range(1, len(elevations)):
            delta = elevations[i] - elevations[i - 1]
            pending += delta
            if abs(pending) > ELEV_THRESHOLD_M:
                if pending > 0:
                    elevation_gain_m += pending
                else:
                    elevation_loss_m += abs(pending)
                pending = 0.0

    # --- timing --------------------------------------------------------------
    times = [p["time"] for p in points if p["time"] is not None]
    start_time: Optional[datetime] = times[0] if times else None
    end_time: Optional[datetime] = times[-1] if times else None
    duration_hours: Optional[float] = None
    if start_time is not None and end_time is not None:
        duration_hours = (end_time - start_time).total_seconds() / 3600.0

    return TrackData(
        name=track_name,
        points=points,
        bounds=bounds,
        total_distance_km=total_distance_km,
        elevation_gain_m=elevation_gain_m,
        elevation_loss_m=elevation_loss_m,
        start_time=start_time,
        end_time=end_time,
        duration_hours=duration_hours,
    )


def padded_bounds(track: TrackData, padding_pct: float = 0.2) -> dict:
    """Return the track bounding box expanded by *padding_pct* on every side.

    A minimum span of 0.01 degrees is enforced on each axis before padding is
    applied, preventing degenerate boxes on very short or single-point routes.

    Parameters
    ----------
    track:
        A :class:`TrackData` instance whose ``bounds`` will be used.
    padding_pct:
        Fraction of the (post-minimum-clamp) lat/lng span added to each edge.
        ``0.2`` means 20 % on every side, growing each dimension by 40 %.

    Returns
    -------
    dict
        ``{min_lat, max_lat, min_lng, max_lng}`` with padding applied.
    """
    MIN_SPAN = 0.01  # degrees

    lat_centre = (track.bounds["min_lat"] + track.bounds["max_lat"]) / 2
    lng_centre = (track.bounds["min_lng"] + track.bounds["max_lng"]) / 2

    raw_lat_span = track.bounds["max_lat"] - track.bounds["min_lat"]
    raw_lng_span = track.bounds["max_lng"] - track.bounds["min_lng"]

    # Expand to minimum span, centred on the track midpoint
    if raw_lat_span < MIN_SPAN:
        base_min_lat = lat_centre - MIN_SPAN / 2
        base_max_lat = lat_centre + MIN_SPAN / 2
        lat_span = MIN_SPAN
    else:
        base_min_lat = track.bounds["min_lat"]
        base_max_lat = track.bounds["max_lat"]
        lat_span = raw_lat_span

    if raw_lng_span < MIN_SPAN:
        base_min_lng = lng_centre - MIN_SPAN / 2
        base_max_lng = lng_centre + MIN_SPAN / 2
        lng_span = MIN_SPAN
    else:
        base_min_lng = track.bounds["min_lng"]
        base_max_lng = track.bounds["max_lng"]
        lng_span = raw_lng_span

    lat_pad = lat_span * padding_pct
    lng_pad = lng_span * padding_pct

    return {
        "min_lat": base_min_lat - lat_pad,
        "max_lat": base_max_lat + lat_pad,
        "min_lng": base_min_lng - lng_pad,
        "max_lng": base_max_lng + lng_pad,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m field_atlas.core.gpx_parser <file.gpx|directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    gpx_files = sorted(target.glob("*.gpx")) if target.is_dir() else [target]

    if not gpx_files:
        print(f"No .gpx files found in {target}")
        sys.exit(1)

    for gpx_file in gpx_files:
        print(f"=== {gpx_file} ===")
        track = parse_gpx(str(gpx_file))
        print(track)
        print()
        pb = padded_bounds(track)
        print(
            f"Padded bounds (20 %):\n"
            f"  lat [{pb['min_lat']:.5f}, {pb['max_lat']:.5f}]\n"
            f"  lng [{pb['min_lng']:.5f}, {pb['max_lng']:.5f}]"
        )
        print()
