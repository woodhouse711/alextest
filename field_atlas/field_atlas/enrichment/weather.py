"""
weather.py — Fetch historical daily weather from the Open-Meteo Archive API.

Retrieves one day of meteorological data for a lat/lng point and packages it
into a WeatherData dataclass with derived fields (cardinal wind direction,
human-readable conditions summary, hourly arrays).

API:  https://open-meteo.com/en/docs/historical-weather-api
No key required; free tier covers archive queries.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import date as _date
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_REQUEST_TIMEOUT_S = 30

# Open-Meteo variable names used in this module.
_DAILY_VARS = ",".join([
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "windspeed_10m_max",
    "winddirection_10m_dominant",
    "cloudcover_mean",
])
_HOURLY_VARS = "temperature_2m,windspeed_10m,precipitation"

# Archive data becomes available roughly 5 days after the observation date.
_ARCHIVE_LAG_DAYS = 5


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class WeatherData:
    """One day of historical weather for a geographic point."""

    date: str                            # YYYY-MM-DD
    latitude: float
    longitude: float

    # Daily summaries
    temperature_min_c: float
    temperature_max_c: float
    temperature_mean_c: float
    precipitation_mm: float
    wind_speed_max_kmh: float
    wind_direction_dominant_deg: float   # 0–360
    wind_direction_cardinal: str         # e.g. "SW", "NNE"
    cloud_cover_mean_pct: float          # 0–100

    # Human-readable
    conditions_summary: str              # e.g. "Partly cloudy, warm (24°C high), moderate SW winds"

    # Hourly arrays (24 values, index 0 = midnight local time)
    hourly_temperatures: list[float]
    hourly_wind_speeds: list[float]
    hourly_precipitation: list[float]

    def __str__(self) -> str:
        lines = [
            f"WeatherData  {self.date}  ({self.latitude:.4f}, {self.longitude:.4f})",
            f"  Temp       : {self.temperature_min_c:.1f}°C → {self.temperature_max_c:.1f}°C"
            f"  (mean {self.temperature_mean_c:.1f}°C)",
            f"  Wind       : {self.wind_direction_cardinal} {self.wind_speed_max_kmh:.1f} km/h"
            f"  ({self.wind_direction_dominant_deg:.0f}°)",
            f"  Cloud cover: {self.cloud_cover_mean_pct:.0f}%",
            f"  Precip     : {self.precipitation_mm:.1f} mm",
            f"  Summary    : {self.conditions_summary}",
            f"  Hourly T   : {', '.join(f'{t:.1f}' for t in self.hourly_temperatures)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deg_to_cardinal(deg: float) -> str:
    """Convert a bearing in degrees (0–360) to an 8-point compass abbreviation.

    Uses equal 45° sectors centred on each cardinal/intercardinal direction.
    """
    sectors = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(deg / 45.0) % 8
    return sectors[idx]


def _cloud_description(pct: float) -> str:
    if pct < 20:
        return "Clear skies"
    if pct < 40:
        return "Mostly clear"
    if pct < 70:
        return "Partly cloudy"
    return "Overcast"


def _temp_description(max_c: float) -> str:
    if max_c < 0:
        return "freezing"
    if max_c < 8:
        return "cold"
    if max_c < 15:
        return "cool"
    if max_c < 22:
        return "mild"
    if max_c < 30:
        return "warm"
    return "hot"


def _wind_description(kmh: float, cardinal: str) -> str:
    if kmh < 5:
        return "calm"
    if kmh < 15:
        return f"light {cardinal} breeze"
    if kmh < 30:
        return f"moderate {cardinal} winds"
    if kmh < 50:
        return f"strong {cardinal} winds"
    return f"very strong {cardinal} winds"


def _build_conditions_summary(
    cloud_pct: float,
    temp_max: float,
    precip_mm: float,
    wind_kmh: float,
    wind_cardinal: str,
) -> str:
    """Compose a short human-readable conditions string."""
    parts = [_cloud_description(cloud_pct)]

    if precip_mm >= 0.5:
        parts.append(f"{precip_mm:.1f}mm rain")

    parts.append(f"{_temp_description(temp_max)} ({temp_max:.0f}°C high)")
    parts.append(_wind_description(wind_kmh, wind_cardinal))

    return ", ".join(parts)


def _validate_date(date_str: str) -> _date:
    """Parse and validate an ISO date string; raise ValueError if unusable."""
    try:
        d = _date.fromisoformat(date_str)
    except ValueError:
        raise ValueError(f"Invalid date format '{date_str}' — expected YYYY-MM-DD.")

    today = datetime.now(timezone.utc).date()

    if d > today:
        raise ValueError(
            f"Date {date_str} is in the future.  "
            "The Open-Meteo Archive only covers dates up to today."
        )

    lag = (today - d).days
    if lag < _ARCHIVE_LAG_DAYS:
        raise ValueError(
            f"Date {date_str} is only {lag} day(s) ago.  "
            f"The Open-Meteo Archive typically has a {_ARCHIVE_LAG_DAYS}-day lag; "
            "try a date at least 5 days in the past."
        )

    return d


def _extract_daily(data: dict, key: str) -> float:
    """Pull the first (and only) value from a daily array, with a NaN fallback."""
    try:
        val = data["daily"][key][0]
        return float(val) if val is not None else float("nan")
    except (KeyError, IndexError, TypeError):
        return float("nan")


def _extract_hourly(data: dict, key: str) -> list[float]:
    """Extract a full 24-value hourly array, padding/truncating as needed."""
    try:
        raw = data["hourly"][key]
    except KeyError:
        return [float("nan")] * 24

    values = [float(v) if v is not None else float("nan") for v in raw]
    # Pad to 24 if shorter; truncate if longer.
    values = (values + [float("nan")] * 24)[:24]
    return values


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_weather(lat: float, lng: float, date: str) -> WeatherData:
    """Fetch one day of historical weather from the Open-Meteo Archive.

    Parameters
    ----------
    lat:
        Latitude in decimal degrees (WGS84).
    lng:
        Longitude in decimal degrees (WGS84).
    date:
        Calendar date as ``"YYYY-MM-DD"``.

    Returns
    -------
    WeatherData
        Fully populated weather record for the requested day.

    Raises
    ------
    ValueError
        If *date* is in the future, too recent for the archive, or badly
        formatted.
    RuntimeError
        If the Open-Meteo API returns an error response.
    requests.RequestException
        On any network-level failure.
    """
    _validate_date(date)  # raises ValueError early if date is unusable

    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": date,
        "end_date": date,
        "daily": _DAILY_VARS,
        "hourly": _HOURLY_VARS,
        "timezone": "auto",
    }

    try:
        resp = requests.get(_ARCHIVE_URL, params=params, timeout=_REQUEST_TIMEOUT_S)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise requests.RequestException(
            f"Open-Meteo Archive request failed: {exc}"
        ) from exc

    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Open-Meteo returned non-JSON response: {resp.text[:200]}"
        ) from exc

    if data.get("error"):
        reason = data.get("reason", "unknown error")
        raise RuntimeError(f"Open-Meteo API error: {reason}")

    # --- Extract daily scalars -------------------------------------------
    temp_max   = _extract_daily(data, "temperature_2m_max")
    temp_min   = _extract_daily(data, "temperature_2m_min")
    temp_mean  = _extract_daily(data, "temperature_2m_mean")
    precip     = _extract_daily(data, "precipitation_sum")
    wind_max   = _extract_daily(data, "windspeed_10m_max")
    wind_deg   = _extract_daily(data, "winddirection_10m_dominant")
    cloud_pct  = _extract_daily(data, "cloudcover_mean")

    # --- Derived fields ---------------------------------------------------
    wind_cardinal = _deg_to_cardinal(wind_deg)
    conditions = _build_conditions_summary(
        cloud_pct, temp_max, precip, wind_max, wind_cardinal
    )

    # --- Hourly arrays ----------------------------------------------------
    hourly_temps  = _extract_hourly(data, "temperature_2m")
    hourly_winds  = _extract_hourly(data, "windspeed_10m")
    hourly_precip = _extract_hourly(data, "precipitation")

    return WeatherData(
        date=date,
        latitude=float(data.get("latitude", lat)),
        longitude=float(data.get("longitude", lng)),
        temperature_min_c=temp_min,
        temperature_max_c=temp_max,
        temperature_mean_c=temp_mean,
        precipitation_mm=precip,
        wind_speed_max_kmh=wind_max,
        wind_direction_dominant_deg=wind_deg,
        wind_direction_cardinal=wind_cardinal,
        cloud_cover_mean_pct=cloud_pct,
        conditions_summary=conditions,
        hourly_temperatures=hourly_temps,
        hourly_wind_speeds=hourly_winds,
        hourly_precipitation=hourly_precip,
    )


def format_weather_line(weather: WeatherData) -> str:
    """Return a compact one-line weather summary for the map's info block.

    Format::

        14°C → 22°C  ·  SW 12 km/h  ·  Partly cloudy  ·  2.1mm precip

    Precipitation is omitted when the total is 0 mm.

    Parameters
    ----------
    weather:
        A :class:`WeatherData` instance as returned by :func:`fetch_weather`.

    Returns
    -------
    str
        Single formatted line, no trailing newline.
    """
    temp_part  = f"{weather.temperature_min_c:.0f}°C → {weather.temperature_max_c:.0f}°C"
    wind_part  = f"{weather.wind_direction_cardinal} {weather.wind_speed_max_kmh:.0f} km/h"
    cloud_part = _cloud_description(weather.cloud_cover_mean_pct)

    parts = [temp_part, wind_part, cloud_part]

    if weather.precipitation_mm > 0:
        parts.append(f"{weather.precipitation_mm:.1f}mm precip")

    return "  ·  ".join(parts)


# ---------------------------------------------------------------------------
# File I/O — cache / offline fallback
# ---------------------------------------------------------------------------


def save_weather_to_file(weather: WeatherData, filepath: str | Path) -> None:
    """Serialise *weather* to a JSON file at *filepath*.

    Parent directories are created automatically.  Overwrites any existing
    file at that path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(weather), fh, indent=2)


def load_weather_from_file(filepath: str | Path) -> WeatherData:
    """Deserialise a :class:`WeatherData` previously written by
    :func:`save_weather_to_file`.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the JSON is present but cannot be decoded into a WeatherData.
    """
    path = Path(filepath)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    try:
        return WeatherData(**data)
    except (TypeError, KeyError) as exc:
        raise ValueError(
            f"Could not parse WeatherData from {filepath}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Middlesex Fells centroid, a known-good archive date.
    LAT, LNG = 42.4491, -71.1140
    DATE = "2024-06-15"

    print(f"Fetching weather for ({LAT}, {LNG}) on {DATE} …")
    try:
        weather = fetch_weather(LAT, LNG, DATE)
    except ValueError as exc:
        print(f"Date error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"API error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print()
    print(weather)
    print()
    print("Formatted line:")
    print(" ", format_weather_line(weather))
