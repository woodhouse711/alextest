"""
field_atlas/enrichment/models.py

EnrichmentData — combined environmental context for a single outing.

Aggregates weather, solar-position, and geographic-feature data into one
structure, and exposes helpers to fetch everything in one call and to
produce the pre-formatted strings the SVG renderer needs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from field_atlas.enrichment.features import (
    FeatureSet,
    fetch_features,
    format_features_line,
    rank_features,
)
from field_atlas.enrichment.solar import (
    SolarData,
    calculate_solar,
    format_solar_line,
)
from field_atlas.enrichment.weather import (
    WeatherData,
    fetch_weather,
    format_weather_line,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Combined dataclass
# ---------------------------------------------------------------------------


@dataclass
class EnrichmentData:
    """All environmental context for a single hiking outing.

    Every field except *date* may be None — a missing source is non-fatal.
    """

    date: str                        # YYYY-MM-DD
    weather: WeatherData | None
    solar: SolarData | None
    features: FeatureSet | None
    location_name: str | None = None # reverse-geocoded or manually set
    notes: str | None = None         # user-supplied notes about the hike

    def __str__(self) -> str:
        lines = [f"EnrichmentData  {self.date}"]
        if self.location_name:
            lines.append(f"  Location : {self.location_name}")
        lines.append(f"  Weather  : {'✓' if self.weather  else '—'}")
        lines.append(f"  Solar    : {'✓' if self.solar    else '—'}")
        lines.append(f"  Features : {'✓' if self.features else '—'}")
        if self.notes:
            lines.append(f"  Notes    : {self.notes}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Aggregate fetch
# ---------------------------------------------------------------------------


def enrich(
    lat: float,
    lng: float,
    date: str,
    bounds: dict,
    route_points: list[dict] | None = None,
    timezone_str: str = "America/New_York",
    notes: str | None = None,
) -> EnrichmentData:
    """Fetch weather, solar, and feature enrichment data for one outing.

    Each source is fetched independently — a failure in one does not prevent
    the others from completing.  Failed sources are set to ``None`` and a
    warning is emitted via ``logging``.

    Parameters
    ----------
    lat, lng:
        Decimal-degree centroid of the route (WGS-84).
    date:
        Calendar date as ``"YYYY-MM-DD"``.
    bounds:
        Dict with ``min_lat``, ``max_lat``, ``min_lng``, ``max_lng`` — the
        padded bounding box passed to the Overpass features query.
    route_points:
        Optional list of ``{"lat": …, "lng": …}`` dicts for the hiking
        route; forwarded to :func:`~field_atlas.enrichment.features.fetch_features`
        and used later by :func:`format_info_block` for proximity ranking.
    timezone_str:
        IANA timezone name used by the solar calculator for local-time output.
    notes:
        Optional user-supplied notes stored verbatim on the result.

    Returns
    -------
    EnrichmentData
        All fields populated where available; ``None`` where a source failed.
    """
    # ---- Weather (Open-Meteo Archive API) --------------------------------
    weather: WeatherData | None = None
    try:
        weather = fetch_weather(lat, lng, date)
        log.info("Weather: fetched from Open-Meteo")
    except Exception as exc:
        log.warning("Weather fetch failed: %s", exc)

    # ---- Solar (offline calculation — pysolar or NOAA equations) ---------
    solar: SolarData | None = None
    try:
        solar = calculate_solar(lat, lng, date, timezone_str)
        log.info("Solar: calculated for %s / %s", date, timezone_str)
    except Exception as exc:
        log.warning("Solar calculation failed: %s", exc)

    # ---- Geographic features (Overpass API) ------------------------------
    features: FeatureSet | None = None
    try:
        features = fetch_features(bounds, route_points=route_points)
        log.info("Features: fetched %d named features from Overpass", len(features.features))
    except Exception as exc:
        log.warning("Features fetch failed: %s", exc)

    return EnrichmentData(
        date=date,
        weather=weather,
        solar=solar,
        features=features,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# SVG renderer bridge
# ---------------------------------------------------------------------------


def format_info_block(
    enrichment: EnrichmentData,
    route_points: list[dict] | None = None,
) -> dict:
    """Return pre-formatted strings ready for the SVG renderer.

    Parameters
    ----------
    enrichment:
        A populated :class:`EnrichmentData` instance.
    route_points:
        Optional list of ``{"lat": …, "lng": …}`` dicts.  When provided,
        :func:`~field_atlas.enrichment.features.rank_features` uses proximity
        to the route to order the feature labels.  When omitted, features are
        ranked by type priority only.

    Returns
    -------
    dict
        ``weather_line``   — one-line weather summary string, or ``None``.

        ``solar_line``     — one-line solar summary string, or ``None``.

        ``feature_labels`` — list of ``{"name", "type", "lat", "lng"}``
                             dicts for up to 8 top-ranked label candidates,
                             or ``[]`` if features are unavailable.
    """
    # --- weather_line -------------------------------------------------------
    weather_line: str | None = None
    if enrichment.weather is not None:
        weather_line = format_weather_line(enrichment.weather)

    # --- solar_line ---------------------------------------------------------
    solar_line: str | None = None
    if enrichment.solar is not None:
        solar_line = format_solar_line(enrichment.solar)

    # --- feature_labels -----------------------------------------------------
    feature_labels: list[dict] = []
    if enrichment.features is not None:
        ranked = rank_features(
            enrichment.features,
            route_points or [],
            max_labels=8,
        )
        feature_labels = [
            {
                "name": feat.name,
                "type": feat.feature_type,
                "lat": feat.lat,
                "lng": feat.lng,
            }
            for feat in ranked
        ]

    return {
        "weather_line": weather_line,
        "solar_line": solar_line,
        "feature_labels": feature_labels,
    }


# ---------------------------------------------------------------------------
# __main__ — demonstrate with sample / cached data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging

    from field_atlas.enrichment.features import load_features_from_file
    from field_atlas.enrichment.weather import load_weather_from_file

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")

    DATE = "2024-06-15"
    LAT, LNG = 42.4491, -71.1140  # Middlesex Fells centroid
    BOUNDS = {"min_lat": 42.420, "max_lat": 42.480, "min_lng": -71.150, "max_lng": -71.080}
    ROUTE = [
        {"lat": 42.4471, "lng": -71.1072},  # Bear Hill
        {"lat": 42.4452, "lng": -71.1100},
        {"lat": 42.4430, "lng": -71.1130},
    ]

    # Load from sample files (live APIs blocked in dev sandbox)
    import sys
    from pathlib import Path

    sample_dir = Path(__file__).parents[2] / "sample_data"

    weather_path  = sample_dir / "weather_2024-06-15.json"
    features_path = sample_dir / "features_middlesex_fells.json"

    weather  = load_weather_from_file(weather_path)  if weather_path.exists()  else None
    features = load_features_from_file(features_path) if features_path.exists() else None

    if weather is None:
        print(f"Warning: {weather_path} not found — weather will be None", file=sys.stderr)
    if features is None:
        print(f"Warning: {features_path} not found — features will be None", file=sys.stderr)

    solar = calculate_solar(LAT, LNG, DATE, "America/New_York")

    enrichment = EnrichmentData(
        date=DATE,
        weather=weather,
        solar=solar,
        features=features,
        location_name="Middlesex Fells Reservation, MA",
        notes="Bear Hill loop via Rock Circuit Trail",
    )

    print(enrichment)
    print()

    info = format_info_block(enrichment, route_points=ROUTE)
    print(f"weather_line   : {info['weather_line']}")
    print(f"solar_line     : {info['solar_line']}")
    print(f"feature_labels : {len(info['feature_labels'])} items")
    for label in info["feature_labels"]:
        print(f"  [{label['type']:10s}] {label['name']}  ({label['lat']:.4f}, {label['lng']:.4f})")
