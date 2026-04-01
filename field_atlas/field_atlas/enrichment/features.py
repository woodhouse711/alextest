"""
field_atlas/enrichment/features.py

Fetch named geographic features near a hiking route from OpenStreetMap via
the Overpass API.  Features become label candidates on the printed artifact.

API:  https://overpass-api.de/api/interpreter
No key required; public instance — be respectful of rate limits.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_REQUEST_TIMEOUT_S = 45

# Metres within which a feature is considered "near the route".
_NEAR_ROUTE_M = 500.0

# Base priority score per feature type — lower = higher priority.
# Used by rank_features(); near-route features are preferred within each band.
_TYPE_PRIORITY: dict[str, int] = {
    "peak":       0,
    "saddle":     1,
    "water":      2,
    "pond":       2,
    "reservoir":  2,
    "viewpoint":  3,
    "cliff":      4,
    "trail":      5,
    "shelter":    6,
    "parking":    7,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MapFeature:
    """A single named geographic feature from OpenStreetMap."""

    name: str
    feature_type: str   # one of: peak, water, trail, shelter, parking,
                        #         viewpoint, saddle, cliff, pond, reservoir
    lat: float          # WGS-84 decimal degrees
    lng: float
    elevation_m: float | None  # from OSM "ele" tag, or None
    osm_id: int


@dataclass
class FeatureSet:
    """Collection of named geographic features within a query bounding box."""

    bounds: dict        # {"min_lat": …, "max_lat": …, "min_lng": …, "max_lng": …}
    features: list[MapFeature] = field(default_factory=list)

    # Filtered convenience views — populated automatically after construction.
    peaks: list[MapFeature] = field(default_factory=list)
    water_features: list[MapFeature] = field(default_factory=list)
    trails: list[MapFeature] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"FeatureSet  bounds={self.bounds}",
            f"  {len(self.features)} named features total:",
            f"    peaks          : {len(self.peaks)}",
            f"    water features : {len(self.water_features)}",
            f"    trails         : {len(self.trails)}",
            f"    other          : {len(self.features) - len(self.peaks) - len(self.water_features) - len(self.trails)}",
        ]
        for feat in self.peaks + self.water_features:
            elev = f"  {feat.elevation_m:.0f}m" if feat.elevation_m is not None else ""
            lines.append(f"    [{feat.feature_type:10s}] {feat.name}{elev}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Overpass query
# ---------------------------------------------------------------------------

# Template — {{bbox}} is replaced with south,west,north,east (Overpass convention).
_QUERY_TEMPLATE = """\
[out:json][timeout:30];
(
  node["natural"="peak"]({{bbox}});
  node["natural"="saddle"]({{bbox}});
  node["natural"="cliff"]({{bbox}});
  node["natural"="water"]({{bbox}});
  way["natural"="water"]({{bbox}});
  node["water"="pond"]({{bbox}});
  node["water"="reservoir"]({{bbox}});
  way["water"="pond"]({{bbox}});
  way["water"="reservoir"]({{bbox}});
  node["tourism"="viewpoint"]({{bbox}});
  node["amenity"="shelter"]({{bbox}});
  node["amenity"="parking"]({{bbox}});
  way["highway"="path"]["name"]({{bbox}});
  way["highway"="track"]["name"]({{bbox}});
);
out center tags;"""


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


def _classify_tags(tags: dict) -> str | None:
    """Return a feature_type string from OSM tags, or None to skip the element."""
    nat     = tags.get("natural")
    water   = tags.get("water")
    tourism = tags.get("tourism")
    amenity = tags.get("amenity")
    highway = tags.get("highway")

    if nat == "peak":               return "peak"
    if nat == "saddle":             return "saddle"
    if nat == "cliff":              return "cliff"
    if nat == "water":
        # Refine if a water sub-type tag is present
        if water == "pond":         return "pond"
        if water == "reservoir":    return "reservoir"
        return "water"
    if water == "pond":             return "pond"
    if water == "reservoir":        return "reservoir"
    if tourism == "viewpoint":      return "viewpoint"
    if amenity == "shelter":        return "shelter"
    if amenity == "parking":        return "parking"
    if highway in ("path", "track"): return "trail"
    return None


def _parse_elevation(tags: dict) -> float | None:
    raw = tags.get("ele")
    if raw is None:
        return None
    try:
        return float(str(raw).strip().split()[0])
    except (ValueError, IndexError):
        return None


def _element_to_feature(el: dict) -> MapFeature | None:
    """Convert one Overpass JSON element to a MapFeature, or None to skip."""
    tags = el.get("tags", {})

    # Require a human-readable name.
    name = tags.get("name") or tags.get("name:en")
    if not name:
        return None

    feature_type = _classify_tags(tags)
    if feature_type is None:
        return None

    osm_type = el.get("type", "node")
    osm_id   = el.get("id", 0)

    if osm_type == "node":
        lat = el.get("lat")
        lng = el.get("lon")
    else:
        # Ways/relations: Overpass returns "center" when requested with "out center"
        center = el.get("center", {})
        lat = center.get("lat")
        lng = center.get("lon")

    if lat is None or lng is None:
        return None

    return MapFeature(
        name=str(name),
        feature_type=feature_type,
        lat=float(lat),
        lng=float(lng),
        elevation_m=_parse_elevation(tags),
        osm_id=int(osm_id),
    )


def _populate_groups(fs: FeatureSet) -> None:
    """Fill the peaks / water_features / trails convenience lists in place."""
    _WATER_TYPES = {"water", "pond", "reservoir"}
    for feat in fs.features:
        if feat.feature_type == "peak" or feat.feature_type == "saddle":
            fs.peaks.append(feat)
        elif feat.feature_type in _WATER_TYPES:
            fs.water_features.append(feat)
        elif feat.feature_type == "trail":
            fs.trails.append(feat)


# ---------------------------------------------------------------------------
# Public API — fetch
# ---------------------------------------------------------------------------


def fetch_features(
    bounds: dict,
    route_points: list[dict] | None = None,
) -> FeatureSet:
    """Fetch named geographic features from the Overpass API.

    Parameters
    ----------
    bounds:
        Dict with keys ``min_lat``, ``max_lat``, ``min_lng``, ``max_lng``
        (WGS-84 degrees).  Typically the padded route bounding box.
    route_points:
        Optional list of ``{"lat": …, "lng": …}`` dicts representing the
        hiking route.  Accepted here for interface consistency; use
        :func:`rank_features` to apply proximity-based ranking afterwards.

    Returns
    -------
    FeatureSet
        All named features within *bounds*, deduplicated by name, with
        convenience group lists populated.

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

    seen_names: set[str] = set()
    features: list[MapFeature] = []
    for el in payload["elements"]:
        feat = _element_to_feature(el)
        if feat is None:
            continue
        if feat.name in seen_names:
            continue            # deduplicate by name — keep first occurrence
        seen_names.add(feat.name)
        features.append(feat)

    fs = FeatureSet(bounds=bounds, features=features)
    _populate_groups(fs)
    return fs


# ---------------------------------------------------------------------------
# Public API — ranking
# ---------------------------------------------------------------------------


def _dist_approx_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Fast planar approximation of distance between two WGS-84 points (metres)."""
    mid_lat_rad = math.radians((lat1 + lat2) / 2.0)
    dlat = (lat2 - lat1) * 111_320.0
    dlng = (lng2 - lng1) * 111_320.0 * math.cos(mid_lat_rad)
    return math.hypot(dlat, dlng)


def rank_features(
    features: FeatureSet,
    route_points: list[dict],
    max_labels: int = 8,
) -> list[MapFeature]:
    """Rank features by relevance for map labelling and return the top N.

    Priority order
    --------------
    * Peaks near the route  (within 500 m of any route point)
    * Named water features near the route
    * Viewpoints near the route
    * Named trails near the route
    * Shelters (any distance)
    * Parking (any distance)
    * Other types fill remaining slots

    Within each priority band, features are sorted by proximity to the nearest
    route point (closer = higher).

    Parameters
    ----------
    features:
        A :class:`FeatureSet` as returned by :func:`fetch_features`.
    route_points:
        List of ``{"lat": …, "lng": …}`` dicts — the hiking route in
        geographic coordinates.
    max_labels:
        Maximum number of features to return.

    Returns
    -------
    list[MapFeature]
        Up to *max_labels* features in descending relevance order.
    """
    if not route_points:
        # No route to compare against — fall back to type-priority order only.
        sorted_feats = sorted(
            features.features,
            key=lambda f: _TYPE_PRIORITY.get(f.feature_type, 8),
        )
        return sorted_feats[:max_labels]

    def _nearest_dist(feat: MapFeature) -> float:
        return min(
            _dist_approx_m(feat.lat, feat.lng, pt["lat"], pt["lng"])
            for pt in route_points
        )

    def _score(feat: MapFeature) -> tuple[int, float]:
        base = _TYPE_PRIORITY.get(feat.feature_type, 8)
        dist = _nearest_dist(feat)
        # Near-route features get even-numbered bands; far ones get odd.
        band = base * 2 if dist <= _NEAR_ROUTE_M else base * 2 + 1
        return (band, dist)

    ranked = sorted(features.features, key=_score)
    return ranked[:max_labels]


# ---------------------------------------------------------------------------
# CLI summary helper
# ---------------------------------------------------------------------------


def format_features_line(features: FeatureSet) -> str:
    """Return a compact one-liner for the map's info block.

    Example: ``2 peaks  ·  3 water features  ·  4 trails``
    """
    parts: list[str] = []
    if features.peaks:
        n = len(features.peaks)
        parts.append(f"{n} {'peak' if n == 1 else 'peaks'}")
    if features.water_features:
        n = len(features.water_features)
        parts.append(f"{n} {'water feature' if n == 1 else 'water features'}")
    if features.trails:
        n = len(features.trails)
        parts.append(f"{n} {'trail' if n == 1 else 'trails'}")
    n_other = (
        len(features.features)
        - len(features.peaks)
        - len(features.water_features)
        - len(features.trails)
    )
    if n_other:
        parts.append(f"{n_other} other")
    return "  ·  ".join(parts) if parts else "No named features found"


# ---------------------------------------------------------------------------
# File I/O — cache / offline fallback
# ---------------------------------------------------------------------------


def save_features_to_file(features: FeatureSet, filepath: str | Path) -> None:
    """Serialise *features* to a JSON file at *filepath*.

    Only the canonical ``bounds`` and ``features`` fields are persisted;
    the derived group lists (peaks, water_features, trails) are re-built on
    load to avoid duplication.  Parent directories are created automatically.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "bounds": features.bounds,
        "features": [asdict(f) for f in features.features],
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_features_from_file(filepath: str | Path) -> FeatureSet:
    """Deserialise a :class:`FeatureSet` previously written by
    :func:`save_features_to_file`.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the JSON cannot be decoded into a FeatureSet.
    """
    path = Path(filepath)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    try:
        fs = FeatureSet(
            bounds=data["bounds"],
            features=[MapFeature(**f) for f in data.get("features", [])],
        )
    except (TypeError, KeyError) as exc:
        raise ValueError(
            f"Could not parse FeatureSet from {filepath}: {exc}"
        ) from exc
    _populate_groups(fs)
    return fs


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Middlesex Fells Reservation, MA — tight bounding box
    BOUNDS = {
        "min_lat": 42.420,
        "max_lat": 42.480,
        "min_lng": -71.150,
        "max_lng": -71.080,
    }

    print(
        f"Fetching features for Middlesex Fells "
        f"({BOUNDS['min_lat']}–{BOUNDS['max_lat']}N, "
        f"{BOUNDS['min_lng']}–{BOUNDS['max_lng']}E) …"
    )
    try:
        fs = fetch_features(BOUNDS)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{len(fs.features)} named features found\n")

    # Group and print by type
    by_type: dict[str, list[MapFeature]] = {}
    for feat in fs.features:
        by_type.setdefault(feat.feature_type, []).append(feat)

    for ftype in sorted(by_type):
        group = by_type[ftype]
        print(f"  {ftype} ({len(group)})")
        for feat in sorted(group, key=lambda f: f.name):
            elev = f"  {feat.elevation_m:.0f}m" if feat.elevation_m is not None else ""
            print(f"    {feat.name}{elev}")

    print()
    print("Formatted line:")
    print(" ", format_features_line(fs))
