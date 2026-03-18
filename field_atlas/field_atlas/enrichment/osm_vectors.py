"""
field_atlas/enrichment/osm_vectors.py

Fetch road, waterway, water-area, trail, terrain-area, and protected-area
vector geometry from OpenStreetMap via the Overpass API.  Returns structured
geometry used by the SVG renderer to draw geographic context layers.

API:  https://overpass-api.de/api/interpreter
No key required; public instance — be respectful of rate limits.

Note: uses "out body geom" which returns full geometry for each way/relation,
including member way geometry for relations.
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
_REQUEST_TIMEOUT_S = 60

# Overpass QL — {{bbox}} is replaced with south,west,north,east.
_QUERY_TEMPLATE = """\
[out:json][timeout:60];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified"]({{bbox}});
  way["waterway"~"river|stream|canal"]({{bbox}});
  way["natural"="water"]({{bbox}});
  relation["natural"="water"]({{bbox}});
  way["highway"~"path|track|footway|bridleway"]({{bbox}});
  way["landuse"~"forest|meadow|farmland"]({{bbox}});
  relation["landuse"~"forest|meadow|farmland"]({{bbox}});
  way["natural"~"wood|scrub|heath|grassland|glacier|sand|scree|wetland"]({{bbox}});
  relation["natural"~"wood|scrub|heath|grassland|glacier|sand|scree|wetland"]({{bbox}});
  relation["boundary"~"national_park|protected_area"]({{bbox}});
  relation["leisure"="nature_reserve"]({{bbox}});
);
out body geom;"""

_ROAD_VALUES    = frozenset("motorway|trunk|primary|secondary|tertiary|residential|unclassified".split("|"))
_WATERWAY_VALUES = frozenset("river|stream|canal".split("|"))
_TRAIL_VALUES   = frozenset("path|track|footway|bridleway".split("|"))

# natural= values that map to terrain types
_TERRAIN_NATURAL: dict[str, str] = {
    "wood":      "forest",
    "scrub":     "scrub",
    "heath":     "heath",
    "grassland": "grassland",
    "glacier":   "glacier",
    "sand":      "sand",
    "scree":     "scree",
    "wetland":   "wetland",
}

# landuse= values that map to terrain types
_TERRAIN_LANDUSE: dict[str, str] = {
    "forest":   "forest",
    "meadow":   "grassland",
    "farmland": "grassland",
}


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
class OSMTerrainArea:
    """A terrain-typed area polygon from OSM landuse/natural tags."""

    terrain_type: str              # e.g. "forest", "scrub", "glacier"
    geometry: list[list[float]]    # [[lat, lng], …] closed ring


@dataclass
class OSMProtectedArea:
    """A protected-area boundary from OSM boundary/leisure tags."""

    name: str | None
    boundary_type: str             # "national_park", "wilderness", "protected_area", "nature_reserve"
    rings: list[list[list[float]]] # outer rings — each is [[lat, lng], …]


@dataclass
class OSMVectors:
    """All vector layers for a single bounding box query."""

    roads: list[OSMWay]
    waterways: list[OSMWay]
    water_areas: list[list[list[float]]]    # list of closed-ring polygons [[lat,lng],…]
    trails: list[OSMWay]
    buildings_area: list[list[list[float]]] = field(default_factory=list)
    terrain_areas: list[OSMTerrainArea]     = field(default_factory=list)
    protected_areas: list[OSMProtectedArea] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_query(bounds: dict) -> str:
    """Substitute bounding box into the Overpass QL template."""
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


def _terrain_type_from_tags(tags: dict) -> str | None:
    """Return a canonical terrain type string from OSM tags, or None."""
    natural = tags.get("natural")
    if natural and natural in _TERRAIN_NATURAL:
        return _TERRAIN_NATURAL[natural]
    landuse = tags.get("landuse")
    if landuse and landuse in _TERRAIN_LANDUSE:
        return _TERRAIN_LANDUSE[landuse]
    return None


def _boundary_type_from_tags(tags: dict) -> str | None:
    """Return a canonical boundary type from OSM tags, or None if not a boundary."""
    boundary = tags.get("boundary")
    if boundary == "national_park":
        return "national_park"
    if boundary == "protected_area":
        protect_class = tags.get("protect_class", "")
        if protect_class in ("1", "1a", "1b", "2"):
            return "wilderness"
        return "protected_area"
    if tags.get("leisure") == "nature_reserve":
        return "nature_reserve"
    return None


def _extract_relation_outer_rings(el: dict) -> list[list[list[float]]]:
    """Extract outer-role member way geometries from a relation element."""
    rings: list[list[list[float]]] = []
    for member in el.get("members", []):
        if member.get("type") != "way":
            continue
        if member.get("role") not in ("outer", ""):
            continue
        geom = [
            [float(n["lat"]), float(n["lon"])]
            for n in member.get("geometry", [])
            if "lat" in n and "lon" in n
        ]
        if geom:
            rings.append(geom)
    return rings


# ---------------------------------------------------------------------------
# Public API — fetch
# ---------------------------------------------------------------------------


def fetch_osm_vectors(bounds: dict) -> OSMVectors:
    """Fetch all vector layers from Overpass for the given bounding box.

    Parameters
    ----------
    bounds:
        Dict with ``min_lat``, ``max_lat``, ``min_lng``, ``max_lng``
        in decimal degrees (WGS-84).

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
    terrain_areas: list[OSMTerrainArea] = []
    protected_areas: list[OSMProtectedArea] = []

    for el in payload["elements"]:
        el_type = el.get("type")
        if el_type not in ("way", "relation"):
            continue

        tags    = el.get("tags", {})
        natural = tags.get("natural")
        landuse = tags.get("landuse")

        if el_type == "relation":
            # ---- Water relations (multipolygon lakes) --------------------
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
                continue

            # ---- Protected area boundaries ------------------------------
            btype = _boundary_type_from_tags(tags)
            if btype is not None:
                rings = _extract_relation_outer_rings(el)
                if rings:
                    name = tags.get("name") or tags.get("name:en") or None
                    protected_areas.append(OSMProtectedArea(
                        name=name,
                        boundary_type=btype,
                        rings=rings,
                    ))
                continue

            # ---- Terrain area relations (multipolygon forest/scrub/etc.) -
            ttype = _terrain_type_from_tags(tags)
            if ttype is not None:
                for member in el.get("members", []):
                    if member.get("type") != "way" or member.get("role") not in ("outer", ""):
                        continue
                    mgeom = [
                        [float(n["lat"]), float(n["lon"])]
                        for n in member.get("geometry", [])
                        if "lat" in n and "lon" in n
                    ]
                    if mgeom and _is_closed(mgeom):
                        terrain_areas.append(OSMTerrainArea(terrain_type=ttype, geometry=mgeom))
            continue

        # ---- Ways --------------------------------------------------------
        geom = _extract_geometry(el)
        if not geom:
            continue

        osm_id   = int(el.get("id", 0))
        name     = tags.get("name") or tags.get("name:en") or None
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
        else:
            # Terrain area ways
            ttype = _terrain_type_from_tags(tags)
            if ttype is not None and _is_closed(geom):
                terrain_areas.append(OSMTerrainArea(terrain_type=ttype, geometry=geom))

    return OSMVectors(
        roads=roads,
        waterways=waterways,
        water_areas=water_areas,
        trails=trails,
        terrain_areas=terrain_areas,
        protected_areas=protected_areas,
    )


# ---------------------------------------------------------------------------
# File I/O — cache / offline fallback
# ---------------------------------------------------------------------------


def save_osm_vectors_to_file(vectors: OSMVectors, filepath: str | Path) -> None:
    """Serialise *vectors* to a JSON file at *filepath*."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "roads":           [asdict(w) for w in vectors.roads],
        "waterways":       [asdict(w) for w in vectors.waterways],
        "water_areas":     vectors.water_areas,
        "trails":          [asdict(w) for w in vectors.trails],
        "buildings_area":  vectors.buildings_area,
        "terrain_areas":   [asdict(a) for a in vectors.terrain_areas],
        "protected_areas": [asdict(p) for p in vectors.protected_areas],
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_osm_vectors_from_file(filepath: str | Path) -> OSMVectors:
    """Deserialise an :class:`OSMVectors` previously written by
    :func:`save_osm_vectors_to_file`.
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
            terrain_areas=  [OSMTerrainArea(**a) for a in data.get("terrain_areas", [])],
            protected_areas=[
                OSMProtectedArea(**p) for p in data.get("protected_areas", [])
            ],
        )
    except (TypeError, KeyError) as exc:
        raise ValueError(
            f"Could not parse OSMVectors from {filepath}: {exc}"
        ) from exc
