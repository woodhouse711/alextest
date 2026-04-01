"""
field_atlas/enrichment/solar.py

Calculates sun position data for a specific date and location using pysolar.
Falls back to a built-in NOAA-equation implementation if pysolar is unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# pysolar import with fallback flag
# ---------------------------------------------------------------------------
try:
    import pysolar.solar as _pysolar  # type: ignore

    _HAVE_PYSOLAR = True
except ImportError:  # pragma: no cover
    _HAVE_PYSOLAR = False


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class SolarData:
    date: str                        # YYYY-MM-DD
    latitude: float
    longitude: float
    timezone_str: str                # IANA timezone name used for local times
    sunrise: str                     # HH:MM local time
    sunset: str                      # HH:MM local time
    solar_noon: str                  # HH:MM local time
    day_length_hours: float
    golden_hour_morning_end: str     # HH:MM local time
    golden_hour_evening_start: str   # HH:MM local time
    noon_altitude_deg: float         # sun's max elevation above horizon
    noon_azimuth_deg: float          # compass bearing of sun at solar noon
    sun_path: list[dict] = field(default_factory=list)
    # Each entry: {hour: int, altitude_deg: float, azimuth_deg: float}


# ---------------------------------------------------------------------------
# Low-level position helpers
# ---------------------------------------------------------------------------


def _utc_aware(dt: datetime) -> datetime:
    """Ensure a datetime is UTC-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _pysolar_position(lat: float, lng: float, dt_utc: datetime) -> tuple[float, float]:
    """Return (altitude_deg, azimuth_deg) via pysolar."""
    alt = _pysolar.get_altitude(lat, lng, dt_utc)
    az = _pysolar.get_azimuth(lat, lng, dt_utc)
    return alt, az


# ---------------------------------------------------------------------------
# NOAA fallback implementation
# ---------------------------------------------------------------------------
# Equations follow NOAA's Solar Calculator spreadsheet
# https://gml.noaa.gov/grad/solcalc/calcdetails.html


def _julian_day(dt_utc: datetime) -> float:
    """Convert a UTC datetime to Julian Day Number."""
    a = (14 - dt_utc.month) // 12
    y = dt_utc.year + 4800 - a
    m = dt_utc.month + 12 * a - 3
    jdn = (dt_utc.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045)
    frac = (dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0) / 24.0 - 0.5
    return jdn + frac


def _noaa_position(lat: float, lng: float, dt_utc: datetime) -> tuple[float, float]:
    """
    Return (altitude_deg, azimuth_deg) using NOAA solar-calculator equations.
    azimuth is clockwise from North (compass bearing).
    """
    jd = _julian_day(dt_utc)
    jc = (jd - 2451545.0) / 36525.0  # Julian century

    # Geometric mean longitude / anomaly of the sun
    gml = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    gma = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    eeo = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)

    # Equation of centre
    sc = (math.sin(math.radians(gma)) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
          + math.sin(math.radians(2 * gma)) * (0.019993 - 0.000101 * jc)
          + math.sin(math.radians(3 * gma)) * 0.000289)
    stl = gml + sc                            # Sun true longitude
    sal = stl - 0.00569 - 0.00478 * math.sin(math.radians(125.04 - 1934.136 * jc))

    # Obliquity of ecliptic
    moe = (23 + (26 + ((21.448 - jc * (46.8150 + jc * (0.00059 - jc * 0.001813)))) / 60) / 60)
    oe = moe + 0.00256 * math.cos(math.radians(125.04 - 1934.136 * jc))

    # Sun declination & right ascension
    decl = math.degrees(math.asin(math.sin(math.radians(oe)) * math.sin(math.radians(sal))))

    # Equation of time (minutes)
    y = math.tan(math.radians(oe / 2)) ** 2
    eot = (4 * math.degrees(
        y * math.sin(2 * math.radians(gml))
        - 2 * eeo * math.sin(math.radians(gma))
        + 4 * eeo * y * math.sin(math.radians(gma)) * math.cos(2 * math.radians(gml))
        - 0.5 * y * y * math.sin(4 * math.radians(gml))
        - 1.25 * eeo * eeo * math.sin(2 * math.radians(gma))
    ))

    # True solar time → hour angle
    minute_utc = dt_utc.hour * 60 + dt_utc.minute + dt_utc.second / 60.0
    tst = (minute_utc + eot + 4 * lng) % 1440
    ha = tst / 4 - 180 if tst / 4 < 0 else (tst / 4 - 180 if tst >= 0 else tst / 4 + 180)
    ha = tst / 4 + 180 if tst < 0 else tst / 4 - 180

    lat_r = math.radians(lat)
    decl_r = math.radians(decl)
    ha_r = math.radians(ha)

    # Solar zenith angle → altitude
    cos_z = (math.sin(lat_r) * math.sin(decl_r)
             + math.cos(lat_r) * math.cos(decl_r) * math.cos(ha_r))
    cos_z = max(-1.0, min(1.0, cos_z))
    zenith = math.degrees(math.acos(cos_z))
    altitude = 90.0 - zenith

    # Azimuth (compass bearing, 0=N, 90=E)
    cos_az = ((math.sin(lat_r) * math.cos(math.radians(zenith)) - math.sin(decl_r))
              / (math.cos(lat_r) * math.sin(math.radians(zenith))))
    cos_az = max(-1.0, min(1.0, cos_az))
    az_raw = math.degrees(math.acos(cos_az))
    azimuth = 180.0 + az_raw if ha > 0 else 180.0 - az_raw

    return altitude, azimuth


def _get_position(lat: float, lng: float, dt_utc: datetime) -> tuple[float, float]:
    """Dispatch to pysolar or fallback."""
    if _HAVE_PYSOLAR:
        return _pysolar_position(lat, lng, dt_utc)
    return _noaa_position(lat, lng, dt_utc)


# ---------------------------------------------------------------------------
# Sunrise / sunset binary search
# ---------------------------------------------------------------------------


def _find_crossing(
    lat: float,
    lng: float,
    date_utc_midnight: datetime,
    search_start_hour: float,
    search_end_hour: float,
    target_alt: float = 0.0,
    rising: bool = True,
) -> datetime | None:
    """
    Binary-search the UTC datetime within [start, end] hours after
    date_utc_midnight where altitude crosses target_alt.
    rising=True  → find where altitude crosses upward (sunrise)
    rising=False → find where altitude crosses downward (sunset)
    Returns None if no crossing is found.
    """
    lo = search_start_hour * 3600  # seconds from midnight UTC
    hi = search_end_hour * 3600

    def alt_at(secs: float) -> float:
        dt = date_utc_midnight + timedelta(seconds=secs)
        return _get_position(lat, lng, dt)[0]

    alt_lo = alt_at(lo)
    alt_hi = alt_at(hi)

    # Check that a crossing exists in the expected direction
    if rising and not (alt_lo < target_alt and alt_hi > target_alt):
        return None
    if not rising and not (alt_lo > target_alt and alt_hi < target_alt):
        return None

    for _ in range(50):  # ~50 iterations gives sub-second precision
        mid = (lo + hi) / 2.0
        alt_mid = alt_at(mid)
        if rising:
            if alt_mid < target_alt:
                lo = mid
            else:
                hi = mid
        else:
            if alt_mid > target_alt:
                lo = mid
            else:
                hi = mid
        if hi - lo < 0.5:
            break

    secs = (lo + hi) / 2.0
    return date_utc_midnight + timedelta(seconds=secs)


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------


def calculate_solar(
    lat: float,
    lng: float,
    date: str,
    timezone_str: str = "America/New_York",
) -> SolarData:
    """
    Calculate all solar position data for the given date and location.

    Parameters
    ----------
    lat, lng       : decimal degrees (WGS-84)
    date           : "YYYY-MM-DD"
    timezone_str   : IANA timezone name for local time output
    """
    tz = ZoneInfo(timezone_str)
    year, month, day = (int(x) for x in date.split("-"))

    # UTC equivalent of local midnight for the requested date
    local_midnight = datetime(year, month, day, 0, 0, 0, tzinfo=tz)
    utc_midnight = local_midnight.astimezone(timezone.utc)

    # ------------------------------------------------------------------ #
    # Sunrise / sunset (altitude = 0°, accounting for refraction ~0.833°) #
    # ------------------------------------------------------------------ #
    HORIZON = -0.833  # standard atmospheric refraction + solar disc radius

    sunrise_utc = _find_crossing(lat, lng, utc_midnight, 0, 16, HORIZON, rising=True)
    sunset_utc = _find_crossing(lat, lng, utc_midnight, 8, 24, HORIZON, rising=False)

    def _fmt_local(dt_utc: datetime | None) -> str:
        if dt_utc is None:
            return "--:--"
        return dt_utc.astimezone(tz).strftime("%H:%M")

    sunrise_str = _fmt_local(sunrise_utc)
    sunset_str = _fmt_local(sunset_utc)

    # Day length
    if sunrise_utc and sunset_utc:
        day_length_hours = (sunset_utc - sunrise_utc).total_seconds() / 3600.0
    else:
        day_length_hours = 0.0

    # ------------------------------------------------------------------ #
    # Solar noon — find UTC time of maximum altitude                       #
    # ------------------------------------------------------------------ #
    # Sample every 2 minutes across the local daytime window and pick peak
    best_noon_utc = utc_midnight + timedelta(hours=12)
    best_alt = -90.0
    for offset_min in range(0, 24 * 60, 2):
        candidate = utc_midnight + timedelta(minutes=offset_min)
        a, _ = _get_position(lat, lng, candidate)
        if a > best_alt:
            best_alt = a
            best_noon_utc = candidate

    noon_alt, noon_az = _get_position(lat, lng, best_noon_utc)
    solar_noon_str = _fmt_local(best_noon_utc)

    # ------------------------------------------------------------------ #
    # Golden hour (sun altitude between HORIZON and +10°)                 #
    # ------------------------------------------------------------------ #
    GOLDEN_LIMIT = 10.0

    gh_morning_end_utc = _find_crossing(
        lat, lng, utc_midnight, 0, 16, GOLDEN_LIMIT, rising=True
    )
    gh_evening_start_utc = _find_crossing(
        lat, lng, utc_midnight, 8, 24, GOLDEN_LIMIT, rising=False
    )

    gh_morning_end_str = _fmt_local(gh_morning_end_utc)
    gh_evening_start_str = _fmt_local(gh_evening_start_utc)

    # ------------------------------------------------------------------ #
    # Sun path (hourly local time, above horizon only)                     #
    # ------------------------------------------------------------------ #
    sun_path: list[dict] = []
    for hour in range(24):
        dt_local = datetime(year, month, day, hour, 0, 0, tzinfo=tz)
        dt_utc = dt_local.astimezone(timezone.utc)
        alt, az = _get_position(lat, lng, dt_utc)
        if alt > 0.0:
            sun_path.append(
                {
                    "hour": hour,
                    "altitude_deg": float(round(alt, 2)),
                    "azimuth_deg": float(round(az, 2)),
                }
            )

    return SolarData(
        date=date,
        latitude=lat,
        longitude=lng,
        timezone_str=timezone_str,
        sunrise=sunrise_str,
        sunset=sunset_str,
        solar_noon=solar_noon_str,
        day_length_hours=round(day_length_hours, 4),
        golden_hour_morning_end=gh_morning_end_str,
        golden_hour_evening_start=gh_evening_start_str,
        noon_altitude_deg=float(round(noon_alt, 2)),
        noon_azimuth_deg=float(round(noon_az, 2)),
        sun_path=sun_path,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_solar_line(solar: SolarData) -> str:
    """
    Return a formatted one-liner for the print info block.

    Example: "Sunrise 05:42  ·  Sunset 20:18  ·  14h 36m daylight"
    """
    total_minutes = round(solar.day_length_hours * 60)
    hours, minutes = divmod(total_minutes, 60)
    return f"Sunrise {solar.sunrise}  ·  Sunset {solar.sunset}  ·  {hours}h {minutes:02d}m daylight"


def sun_arc_points(solar: SolarData, num_points: int = 48) -> list[dict]:
    """
    Return a denser set of evenly-spaced points along the sun's arc.

    Each point:
        fraction     : 0.0 = sunrise, 1.0 = sunset
        altitude_deg : sun elevation above horizon
        azimuth_deg  : compass bearing (0=N, 90=E)
    """
    # Re-derive sunrise/sunset bounds using the same reference as calculate_solar
    year, month, day = (int(x) for x in solar.date.split("-"))
    local_midnight = datetime(year, month, day, 0, 0, 0,
                              tzinfo=ZoneInfo(solar.timezone_str))
    utc_midnight = local_midnight.astimezone(timezone.utc)
    HORIZON = -0.833

    sr = _find_crossing(solar.latitude, solar.longitude, utc_midnight, 0, 16, HORIZON, rising=True)
    ss = _find_crossing(solar.latitude, solar.longitude, utc_midnight, 8, 24, HORIZON, rising=False)

    if sr is None or ss is None:
        return []

    duration = (ss - sr).total_seconds()
    points: list[dict] = []

    for i in range(num_points):
        fraction = i / (num_points - 1) if num_points > 1 else 0.0
        dt_utc = sr + timedelta(seconds=fraction * duration)
        alt, az = _get_position(solar.latitude, solar.longitude, dt_utc)
        points.append(
            {
                "fraction": round(fraction, 6),
                "altitude_deg": float(round(alt, 3)),
                "azimuth_deg": float(round(az, 3)),
            }
        )

    return points


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    LAT, LNG = 42.4491, -71.1140  # Middlesex Fells, MA
    DATE = "2024-06-15"
    TZ = "America/New_York"

    print(f"Calculating solar data for Middlesex Fells ({LAT}, {LNG})")
    print(f"Date: {DATE}  |  Timezone: {TZ}")
    print(f"Using {'pysolar' if _HAVE_PYSOLAR else 'NOAA fallback equations'}")
    print("-" * 60)

    solar = calculate_solar(LAT, LNG, DATE, TZ)

    print(format_solar_line(solar))
    print(f"Solar noon    : {solar.solar_noon}")
    print(f"Day length    : {solar.day_length_hours:.4f} h")
    print(f"Noon altitude : {solar.noon_altitude_deg}°")
    print(f"Noon azimuth  : {solar.noon_azimuth_deg}°")
    print(f"Golden hour AM ends  : {solar.golden_hour_morning_end}")
    print(f"Golden hour PM starts: {solar.golden_hour_evening_start}")

    print()
    print(f"{'Hour (local)':>12}  {'Altitude°':>10}  {'Azimuth°':>10}")
    print("-" * 38)
    for p in solar.sun_path:
        print(f"{p['hour']:>12}  {p['altitude_deg']:>10.2f}  {p['azimuth_deg']:>10.2f}")

    print()
    print("Sun arc points (48 samples):")
    arc = sun_arc_points(solar, num_points=48)
    print(f"{'Fraction':>10}  {'Altitude°':>10}  {'Azimuth°':>10}")
    print("-" * 36)
    for pt in arc[::8]:  # print every 8th for brevity
        print(f"{pt['fraction']:>10.4f}  {pt['altitude_deg']:>10.3f}  {pt['azimuth_deg']:>10.3f}")
