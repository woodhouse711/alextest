"""
field_atlas/enrichment/features.py

Fetch named geographic features near a location from the OpenStreetMap
Overpass API (peaks, viewpoints, shelters, water sources, trailheads, etc.)
and package them into a FeaturesData dataclass.

API:  https://overpass-api.de/api/interpreter
No key required; public instance with reasonable rate limits.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_REQUEST_TIMEOUT_S = 45

# Map from (key, value) OSM tag pair → internal feature_type label.
# Checked in order; first match wins.
_TAG_TYPE_MAP: list[tuple[tuple[str, str], str]] = [
    (("natural", "peak"),          "peak"),
    (("natural", "saddle"),        "saddle"),
    (("tourism", "viewpoint"),     "viewpoint"),
    (("man_made", "tower"),        "tower"),
    (("historic", "tower"),        "tower"),
    (("amenity", "shelter"),       "shelter"),
    (("tourism", "wilderness_hut"), "shelter"),
    (("tourism", "camp_site"),     "campsite"),
    (("natural", "spring"),        "water"),
    (("amenity", "drinking_water"), "water"),
    (("amenity", "fountain"),      "water"),
    (("amenity", "toilets"),       "toilets"),
    (("amenity", "parking"),       "parking"),
    (("highway", "trailhead"),     "trailhead"),
    (("tourism", "information"),   "information"),
    (("information", "board"),     "information"),
    (("information", "map"),       "information"),
    (("leisure", "picnic_table"),  "picnic"),
    (("tourism", "picnic_site"),   "picnic"),
    (("natural", "cave_entrance"), "cave"),
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Feature:
    """A single geographic point-of-interest from OpenStreetMap."""

    osm_id: int
    osm_type: str          # "node", "way", or "relation"
    feature_type: str      # normalised type label, e.g. "peak", "viewpoint"
    name: str | None       # from OSM "name" tag, or None
    latitude: float        # centroid latitude (degrees)
    longitude: float       # centroid longitude (degrees)
    elevation_m: float | None  # from OSM "ele" tag, or None
    tags: dict             # raw OSM tag dict for further inspection


@dataclass
class FeaturesData:
    """Collection of geographic features near a query location."""

    latitude: float        # query centre
    longitude: float
    radius_m: float        # search radius used
    query_date: str        # YYYY-MM-DD (local date when query was made)
    fetched_at: str        # ISO 8601 UTC timestamp
    features: list[Feature] = field(default_factory=list)

    # Convenience groupings (populated by _group_features after fetch)
    peaks: list[Feature] = field(default_factory=list)
    viewpoints: list[Feature] = field(default_factory=list)
    towers: list[Feature] = field(default_factory=list)
    shelters: list[Feature] = field(default_factory=list)
    water_sources: list[Feature] = field(default_factory=list)
    parking: list[Feature] = field(default_factory=list)
    trailheads: list[Feature] = field(default_factory=list)
    other: list[Feature] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"FeaturesData  ({self.latitude:.4f}, {self.longitude:.4f})"
            f"  r={self.radius_m:.0f}m  {self.query_date}",
            f"  {len(self.features)} features total:",
            f"    peaks      : {len(self.peaks)}",
            f"    viewpoints : {len(self.viewpoints)}",
            f"    towers     : {len(self.towers)}",
            f"    shelters   : {len(self.shelters)}",
            f"    water      : {len(self.water_sources)}",
            f"    parking    : {len(self.parking)}",
            f"    trailheads : {len(self.trailheads)}",
            f"    other      : {len(self.other)}",
        ]
        # Show named items in each priority category
        for feat in (self.peaks + self.viewpoints + self.towers):
            name = feat.name or "(unnamed)"
            elev = f"  {feat.elevation_m:.0f}m" if feat.elevation_m is not None else ""
            lines.append(f"    [{feat.feature_type}] {name}{elev}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_tags(tags: dict) -> str:
    """Return a normalised feature_type string from raw OSM tags."""
    for (key, value), label in _TAG_TYPE_MAP:
        if tags.get(key) == value:
            return label
    return "other"


def _parse_elevation(tags: dict) -> float | None:
    """Extract a numeric elevation from OSM 'ele' tag, or return None."""
    raw = tags.get("ele")
    if raw is None:
        return None
    try:
        return float(str(raw).strip().split()[0])  # handle "145 m" or "145.3"
    except (ValueError, IndexError):
        return None


def _element_to_feature(el: dict) -> Feature | None:
    """Convert one Overpass JSON element dict to a Feature, or None if unusable."""
    osm_type = el.get("type", "")
    osm_id = el.get("id", 0)
    tags = el.get("tags", {})

    # Determine centroid lat/lng
    if osm_type == "node":
        lat = el.get("lat")
        lng = el.get("lon")
    else:
        # For ways/relations Overpass returns center when requested
        centre = el.get("center", {})
        lat = centre.get("lat")
        lng = centre.get("lon")

    if lat is None or lng is None:
        return None

    feature_type = _classify_tags(tags)
    name = tags.get("name") or tags.get("name:en") or None
    elevation_m = _parse_elevation(tags)

    return Feature(
        osm_id=osm_id,
        osm_type=osm_type,
        feature_type=feature_type,
        name=name,
        latitude=float(lat),
        longitude=float(lng),
        elevation_m=elevation_m,
        tags=tags,
    )


def _group_features(fd: FeaturesData) -> None:
    """Populate the convenience category lists from fd.features (in place)."""
    bucket_map: dict[str, list[Feature]] = {
        "peak":      fd.peaks,
        "saddle":    fd.peaks,
        "viewpoint": fd.viewpoints,
        "tower":     fd.towers,
        "shelter":   fd.shelters,
        "campsite":  fd.shelters,
        "water":     fd.water_sources,
        "toilets":   fd.other,
        "parking":   fd.parking,
        "trailhead": fd.trailheads,
        "information": fd.other,
        "picnic":    fd.other,
        "cave":      fd.other,
        "other":     fd.other,
    }
    for feat in fd.features:
        bucket = bucket_map.get(feat.feature_type, fd.other)
        bucket.append(feat)


def _build_overpass_query(lat: float, lng: float, radius_m: float) -> str:
    """Construct an Overpass QL query that fetches all relevant node/way types
    within *radius_m* metres of (*lat*, *lng*)."""
    r = int(radius_m)
    node_filters = " ".join([
        'node["natural"~"^(peak|saddle|spring|cave_entrance)$"]',
        'node["tourism"~"^(viewpoint|information|camp_site|picnic_site|wilderness_hut)$"]',
        'node["amenity"~"^(shelter|drinking_water|fountain|toilets|parking)$"]',
        'node["man_made"="tower"]',
        'node["historic"="tower"]',
        'node["highway"="trailhead"]',
        'node["information"~"^(board|map)$"]',
        'node["leisure"="picnic_table"]',
    ])
    way_filters = " ".join([
        'way["tourism"~"^(viewpoint|camp_site|picnic_site)$"]',
        'way["amenity"~"^(shelter|parking)$"]',
        'way["man_made"="tower"]',
    ])

    lines = [
        "[out:json][timeout:40];",
        f"(  // within {r}m of ({lat}, {lng})",
    ]
    for f in node_filters.split("node"):
        if f.strip():
            lines.append(f'  node{f.strip()}(around:{r},{lat},{lng});')
    for f in way_filters.split("way"):
        if f.strip():
            lines.append(f'  way{f.strip()}(around:{r},{lat},{lng});')
    lines += [
        ");",
        "out center tags;",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_features(
    lat: float,
    lng: float,
    radius_m: float = 3000.0,
    date: str | None = None,
) -> FeaturesData:
    """Fetch geographic features from the OpenStreetMap Overpass API.

    Parameters
    ----------
    lat, lng     : decimal degrees (WGS-84) — query centre
    radius_m     : search radius in metres (default 3 km)
    date         : YYYY-MM-DD label for the returned FeaturesData;
                   defaults to today's UTC date

    Returns
    -------
    FeaturesData
        Populated dataclass with features grouped by type.

    Raises
    ------
    requests.RequestException
        On any network-level failure.
    RuntimeError
        If the Overpass API returns an error response.
    """
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    query = _build_overpass_query(lat, lng, radius_m)

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
            f"Unexpected Overpass response (no 'elements' key): "
            f"{str(payload)[:200]}"
        )

    features: list[Feature] = []
    for el in payload["elements"]:
        feat = _element_to_feature(el)
        if feat is not None:
            features.append(feat)

    fd = FeaturesData(
        latitude=lat,
        longitude=lng,
        radius_m=radius_m,
        query_date=date,
        fetched_at=fetched_at,
        features=features,
    )
    _group_features(fd)
    return fd


def format_features_line(features: FeaturesData) -> str:
    """Return a compact one-liner for the map's info block.

    Example: "3 peaks  ·  2 viewpoints  ·  4 water sources"
    """
    parts: list[str] = []
    if features.peaks:
        label = "peak" if len(features.peaks) == 1 else "peaks"
        parts.append(f"{len(features.peaks)} {label}")
    if features.viewpoints:
        label = "viewpoint" if len(features.viewpoints) == 1 else "viewpoints"
        parts.append(f"{len(features.viewpoints)} {label}")
    if features.towers:
        label = "tower" if len(features.towers) == 1 else "towers"
        parts.append(f"{len(features.towers)} {label}")
    if features.water_sources:
        label = "water source" if len(features.water_sources) == 1 else "water sources"
        parts.append(f"{len(features.water_sources)} {label}")
    if features.shelters:
        label = "shelter" if len(features.shelters) == 1 else "shelters"
        parts.append(f"{len(features.shelters)} {label}")
    if not parts:
        return "No notable features found"
    return "  ·  ".join(parts)


# ---------------------------------------------------------------------------
# File I/O — cache / offline fallback
# ---------------------------------------------------------------------------


def save_features_to_file(features: FeaturesData, filepath: str | Path) -> None:
    """Serialise *features* to a JSON file at *filepath*.

    The grouped category lists (peaks, viewpoints, etc.) are excluded from the
    file — they are re-derived on load to avoid duplication.
    Parent directories are created automatically.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Only persist the canonical fields, not the derived groupings
    data = {
        "latitude": features.latitude,
        "longitude": features.longitude,
        "radius_m": features.radius_m,
        "query_date": features.query_date,
        "fetched_at": features.fetched_at,
        "features": [asdict(f) for f in features.features],
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_features_from_file(filepath: str | Path) -> FeaturesData:
    """Deserialise a :class:`FeaturesData` previously written by
    :func:`save_features_to_file`.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the JSON cannot be decoded into a FeaturesData.
    """
    path = Path(filepath)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    try:
        raw_features = data.pop("features", [])
        fd = FeaturesData(
            **data,
            features=[Feature(**f) for f in raw_features],
        )
    except (TypeError, KeyError) as exc:
        raise ValueError(
            f"Could not parse FeaturesData from {filepath}: {exc}"
        ) from exc
    _group_features(fd)
    return fd


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    LAT, LNG = 42.4491, -71.1140  # Middlesex Fells, MA
    RADIUS = 3000.0
    DATE = "2024-06-15"

    print(f"Fetching features for Middlesex Fells ({LAT}, {LNG}), r={RADIUS:.0f}m …")
    try:
        features = fetch_features(LAT, LNG, radius_m=RADIUS, date=DATE)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(features)
    print()
    print("Formatted line:")
    print(" ", format_features_line(features))
