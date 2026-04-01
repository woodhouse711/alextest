"""
field_atlas/enrichment/osm_vectors.py

Fetch road, waterway, water-area, and trail vector geometry from OpenStreetMap
via the Overpass API.  Returns structured geometry used by the SVG renderer to
draw geographic context layers (below contours) and a trail network layer
(above contours, below the user's route).

API:  https://overpass-api.de/api/interpreter
No key required; public instance — be respectful of rate limits.

Note: uses "out body geom" which returns full geometry for each way, unlike
the feature query in features.py which uses "out center tags" (centre-point
only).  Full geometry is required here because we render actual line/polygon
shapes, not just placement markers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_REQUEST_TIMEOUT_S = 45

# Overpass QL — {{bbox}} is replaced with south,west,north,east.
_QUERY_TEMPLATE = """\
[out:json][timeout:45];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified"]({{bbox}});
  way["waterway"~"river|stream|canal"]({{bbox}});
  way["natural"="water"]({{bbox}});
  relation["natural"="water"]({{bbox}});
  way["highway"~"path|track|footway|bridleway"]({{bbox}});
);
out body geom;"""

_ROAD_VALUES   = frozenset("motorway|trunk|primary|secondary|tertiary|residential|unclassified".split("|"))
_WATERWAY_VALUES = frozenset("river|stream|canal".split("|"))
_TRAIL_VALUES  = frozenset("path|track|footway|bridleway".split("|"))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OSMWay:
    """A single OpenStreetMap way with full node geometry."""

    id: int
    name: str | None          # OSM "name" tag, or None
    tag: str                  # OSM tag key: "highway", "waterway", or "natural"
    value: str                # OSM tag value: e.g. "primary", "river", "water"
    geometry: list[list[float]]  # [[lat, lng], …] in WGS-84


@dataclass
class OSMVectors:
    """All vector layers for a single bounding box query."""

    roads: list[OSMWay]
    waterways: list[OSMWay]
    water_areas: list[list[list[float]]]    # list of closed-ring polygons [[lat,lng],…]
    trails: list[OSMWay]
    buildings_area: list[list[list[float]]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_query(bounds: dict) -> str:
    """Substitute bounding box into the Overpass QL template.

    Overpass uses south,west,north,east order.
    """
    bbox = (
        f"{bounds['min_lat']},{bounds['min_lng']},"
        f"{bounds['max_lat']},{bounds['max_lng']}"
    )
    return _QUERY_TEMPLATE.replace("{{bbox}}", bbox)


def _extract_geometry(el: dict) -> list[list[float]]:
    """Extract [[lat, lng], …] from an element's ``geometry`` array."""
    return [
        [float(n["lat"]), float(n["lon"])]
        for n in el.get("geometry", [])
        if "lat" in n and "lon" in n
    ]


def _is_closed(geom: list[list[float]]) -> bool:
    """True if the geometry forms a closed ring (first point == last point)."""
    if len(geom) < 3:
        return False
    return geom[0][0] == geom[-1][0] and geom[0][1] == geom[-1][1]


# ---------------------------------------------------------------------------
# Public API — fetch
# ---------------------------------------------------------------------------


def fetch_osm_vectors(bounds: dict) -> OSMVectors:
    """Fetch road, waterway, water-area, and trail vectors from Overpass.

    Parameters
    ----------
    bounds:
        Dict with ``min_lat``, ``max_lat``, ``min_lng``, ``max_lng``
        in decimal degrees (WGS-84).  Typically the padded route bounding box.

    Returns
    -------
    OSMVectors
        All vector layers for the queried bounding box.

    Raises
    ------
    requests.RequestException
        On any network-level failure.
    RuntimeError
        If the Overpass API returns an unexpected response.
    """
    query = _build_query(bounds)

    try:
        resp = requests.post(
            _OVERPASS_URL,
            data={"data": query},
            timeout=_REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise requests.RequestException(
            f"Overpass API request failed: {exc}"
        ) from exc

    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Overpass returned non-JSON response: {resp.text[:200]}"
        ) from exc

    if "elements" not in payload:
        raise RuntimeError(
            f"Unexpected Overpass response (no 'elements' key): {str(payload)[:200]}"
        )

    roads: list[OSMWay] = []
    waterways: list[OSMWay] = []
    water_areas: list[list[list[float]]] = []
    trails: list[OSMWay] = []

    for el in payload["elements"]:
        el_type = el.get("type")
        if el_type not in ("way", "relation"):
            continue

        tags     = el.get("tags", {})
        natural  = tags.get("natural")

        # Relations (multipolygons) don't carry a top-level geometry array —
        # geometry lives inside each member way.  Handle water relations first
        # so large lakes stored as multipolygons are captured.
        if el_type == "relation":
            if natural == "water":
                for member in el.get("members", []):
                    if member.get("role") in ("outer", "") and member.get("type") == "way":
                        mgeom = [
                            [float(n["lat"]), float(n["lon"])]
                            for n in member.get("geometry", [])
                            if "lat" in n and "lon" in n
                        ]
                        if mgeom and _is_closed(mgeom):
                            water_areas.append(mgeom)
            continue  # relations only used for water; skip other tags

        geom = _extract_geometry(el)
        if not geom:
            continue

        osm_id  = int(el.get("id", 0))
        name    = tags.get("name") or tags.get("name:en") or None
        highway  = tags.get("highway")
        waterway = tags.get("waterway")

        if natural == "water":
            if _is_closed(geom):
                water_areas.append(geom)
        elif highway in _ROAD_VALUES:
            roads.append(OSMWay(id=osm_id, name=name, tag="highway", value=highway, geometry=geom))
        elif highway in _TRAIL_VALUES:
            trails.append(OSMWay(id=osm_id, name=name, tag="highway", value=highway, geometry=geom))
        elif waterway in _WATERWAY_VALUES:
            waterways.append(OSMWay(id=osm_id, name=name, tag="waterway", value=waterway, geometry=geom))

    return OSMVectors(roads=roads, waterways=waterways, water_areas=water_areas, trails=trails)


# ---------------------------------------------------------------------------
# File I/O — cache / offline fallback
# ---------------------------------------------------------------------------


def save_osm_vectors_to_file(vectors: OSMVectors, filepath: str | Path) -> None:
    """Serialise *vectors* to a JSON file at *filepath*.

    Parent directories are created automatically.  Overwrites any existing
    file at that path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "roads":          [asdict(w) for w in vectors.roads],
        "waterways":      [asdict(w) for w in vectors.waterways],
        "water_areas":    vectors.water_areas,
        "trails":         [asdict(w) for w in vectors.trails],
        "buildings_area": vectors.buildings_area,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_osm_vectors_from_file(filepath: str | Path) -> OSMVectors:
    """Deserialise an :class:`OSMVectors` previously written by
    :func:`save_osm_vectors_to_file`.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the JSON cannot be decoded into an OSMVectors.
    """
    path = Path(filepath)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    try:
        return OSMVectors(
            roads=          [OSMWay(**w) for w in data.get("roads", [])],
            waterways=      [OSMWay(**w) for w in data.get("waterways", [])],
            water_areas=    data.get("water_areas", []),
            trails=         [OSMWay(**w) for w in data.get("trails", [])],
            buildings_area= data.get("buildings_area", []),
        )
    except (TypeError, KeyError) as exc:
        raise ValueError(
            f"Could not parse OSMVectors from {filepath}: {exc}"
        ) from exc
