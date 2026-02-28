"""
projection.py — WGS84 <-> UTM coordinate conversion helpers.

Converts lat/lng points into projected metre-based UTM coordinates so that
downstream terrain processing can work in consistent metric units.
"""

from __future__ import annotations

import math

import pyproj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def auto_utm_zone(lat: float, lng: float) -> int:
    """Return the UTM zone number (1–60) for a given WGS84 coordinate.

    Parameters
    ----------
    lat:
        Latitude in decimal degrees (–90 to 90).
    lng:
        Longitude in decimal degrees (–180 to 180).

    Returns
    -------
    int
        UTM zone number.
    """
    return math.floor((lng + 180.0) / 6.0) % 60 + 1


def get_projection(lat: float, lng: float) -> pyproj.Transformer:
    """Return a :class:`pyproj.Transformer` from WGS84 to the local UTM zone.

    The UTM zone is derived from the supplied centroid coordinate.  Northern
    and southern hemisphere variants are handled via the standard EPSG codes:

    * Northern: ``EPSG:326{zone}``
    * Southern: ``EPSG:327{zone}``

    Parameters
    ----------
    lat:
        Centroid latitude in decimal degrees.
    lng:
        Centroid longitude in decimal degrees.

    Returns
    -------
    pyproj.Transformer
        Ready-to-use transformer; call ``transformer.transform(lat, lng)``
        to obtain ``(easting, northing)`` in metres.  The output axis order
        follows the EPSG:326XX definition (Easting first, Northing second).
    """
    zone = auto_utm_zone(lat, lng)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=False)


def project_points(
    points: list[dict],
    transformer: pyproj.Transformer,
) -> list[dict]:
    """Add UTM ``'x'`` (easting) and ``'y'`` (northing) keys to each point.

    The input list is mutated in-place **and** returned for convenience.

    Parameters
    ----------
    points:
        List of point dicts as produced by :func:`~field_atlas.core.gpx_parser.parse_gpx`.
        Each dict must have ``'lat'`` and ``'lng'`` keys.
    transformer:
        A transformer returned by :func:`get_projection`.

    Returns
    -------
    list[dict]
        The same list with ``'x'`` and ``'y'`` added to every element.
    """
    for pt in points:
        # With always_xy=False the output axis order follows the target CRS
        # definition.  EPSG:326XX (UTM) defines axes as (Easting, Northing),
        # so transform(lat, lng) → (easting, northing).
        easting, northing = transformer.transform(pt["lat"], pt["lng"])
        pt["x"] = easting
        pt["y"] = northing
    return points


def project_bounds(bounds: dict, transformer: pyproj.Transformer) -> dict:
    """Project the four corners of a lat/lng bounding box into UTM metres.

    All four corners are projected and the true axis-aligned envelope in
    projected space is returned, so the result is safe even for bounding
    boxes that span significant areas where meridian convergence matters.

    Parameters
    ----------
    bounds:
        Dict with keys ``min_lat``, ``max_lat``, ``min_lng``, ``max_lng``
        as returned by :func:`~field_atlas.core.gpx_parser.parse_gpx`.
    transformer:
        A transformer returned by :func:`get_projection`.

    Returns
    -------
    dict
        ``{min_x, max_x, min_y, max_y}`` in metres (UTM easting/northing).
    """
    corners = [
        (bounds["min_lat"], bounds["min_lng"]),
        (bounds["min_lat"], bounds["max_lng"]),
        (bounds["max_lat"], bounds["min_lng"]),
        (bounds["max_lat"], bounds["max_lng"]),
    ]
    projected = [transformer.transform(lat, lng) for lat, lng in corners]
    # EPSG:326XX axis order is (Easting, Northing), so p[0]=easting, p[1]=northing.
    eastings  = [p[0] for p in projected]
    northings = [p[1] for p in projected]
    return {
        "min_x": min(eastings),
        "max_x": max(eastings),
        "min_y": min(northings),
        "max_y": max(northings),
    }
