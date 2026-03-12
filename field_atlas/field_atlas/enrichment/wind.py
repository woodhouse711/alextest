"""wind.py — Wind vector field generation and streamline tracing.

Generates a 2D wind vector field across the map extent, terrain-deflected by
the local gradient, then traces streamlines through it using 4th-order
Runge-Kutta integration.  The result is a set of smooth particle-like curves
suitable for rendering as a subtle static animation layer.

Coordinate system
-----------------
All x/y values used in the public API and returned streamlines are in the
projected coordinate system supplied as ``bounds_projected`` (metres, UTM).
Internally the wind grid is a regular N×N raster over the same bounds, with
row 0 at max_y (north) and col 0 at min_x (west).

Wind direction convention
-------------------------
Meteorological convention throughout: ``wind_direction_deg`` is the direction
the wind comes *from* (0° = N, 90° = E, 225° = SW, etc.).  The resulting flow
vector therefore points in the *opposite* direction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pos_to_grid(
    x: float,
    y: float,
    bounds: dict,
    N: int,
) -> tuple[float, float]:
    """Map a projected (x, y) position to fractional grid (row, col).

    Row 0 = max_y (north), row N-1 = min_y (south).
    Col 0 = min_x (west),  col N-1 = max_x (east).
    """
    col = (x - bounds["min_x"]) / (bounds["max_x"] - bounds["min_x"]) * (N - 1)
    row = (bounds["max_y"] - y) / (bounds["max_y"] - bounds["min_y"]) * (N - 1)
    return row, col


def _bilinear(field: np.ndarray, row: float, col: float) -> float:
    """Bilinear interpolation into a 2D field at fractional (row, col)."""
    nr, nc = field.shape
    r0 = max(0, min(int(math.floor(row)), nr - 1))
    c0 = max(0, min(int(math.floor(col)), nc - 1))
    r1 = max(0, min(r0 + 1, nr - 1))
    c1 = max(0, min(c0 + 1, nc - 1))
    dr = row - math.floor(row)
    dc = col - math.floor(col)
    return (
        (1 - dr) * (1 - dc) * field[r0, c0]
        + (1 - dr) * dc       * field[r0, c1]
        + dr       * (1 - dc) * field[r1, c0]
        + dr       * dc       * field[r1, c1]
    )


def _sample_wind(
    u_field: np.ndarray,
    v_field: np.ndarray,
    x: float,
    y: float,
    bounds: dict,
) -> tuple[float, float]:
    """Bilinearly sample (u, v) at projected position (x, y)."""
    N = u_field.shape[0]
    row, col = _pos_to_grid(x, y, bounds, N)
    return _bilinear(u_field, row, col), _bilinear(v_field, row, col)


def _rk4_step(
    u_field: np.ndarray,
    v_field: np.ndarray,
    x: float,
    y: float,
    bounds: dict,
    h: float,
) -> tuple[float, float, float]:
    """Advance (x, y) by *h* metres along the streamline using RK4.

    The wind field vectors are direction-normalized at each sub-step so the
    step size is a physical distance (metres), not a time increment.  Returns
    (new_x, new_y, local_speed_kmh) where local_speed is the un-normalized
    magnitude at the initial position.
    """
    def _dir(px: float, py: float) -> tuple[float, float, float]:
        u, v = _sample_wind(u_field, v_field, px, py, bounds)
        spd = math.sqrt(u * u + v * v)
        if spd < 1e-9:
            return 0.0, 0.0, 0.0
        return u / spd, v / spd, spd

    k1u, k1v, s1 = _dir(x, y)
    k2u, k2v, s2 = _dir(x + 0.5 * h * k1u, y + 0.5 * h * k1v)
    k3u, k3v, s3 = _dir(x + 0.5 * h * k2u, y + 0.5 * h * k2v)
    k4u, k4v, s4 = _dir(x + h * k3u, y + h * k3v)

    dx = h / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
    dy = h / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    speed = (s1 + 2 * s2 + 2 * s3 + s4) / 6.0

    return x + dx, y + dy, speed


# ---------------------------------------------------------------------------
# Public API — vector field construction
# ---------------------------------------------------------------------------

def build_wind_field(
    elevation: np.ndarray,
    wind_speed_kmh: float,
    wind_direction_deg: float,
    bounds_projected: dict,
    grid_resolution: int = 64,
    date_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 2D wind vector field deflected by terrain.

    Parameters
    ----------
    elevation:
        Raw DEM elevation array (any shape ≥ 2×2, float).  Resampled
        internally to *grid_resolution* × *grid_resolution*.
    wind_speed_kmh:
        Scalar base wind speed in km/h.
    wind_direction_deg:
        Meteorological convention: direction wind comes *from* in degrees
        (0° = N, 90° = E, 225° = SW, etc.).
    bounds_projected:
        ``{min_x, max_x, min_y, max_y}`` bounding box in projected metres.
    grid_resolution:
        Square grid side length.  Default 64 → 64 × 64 cells.
    date_seed:
        Integer seed for deterministic turbulence noise.  Pass the date as
        an integer (e.g. ``20240615``) to get consistent per-day variation.

    Returns
    -------
    (u_field, v_field)
        Two ``float32`` arrays of shape ``(N, N)`` giving wind velocity
        components in km/h.  *u* is the eastward component, *v* the northward.
    """
    from scipy.ndimage import zoom as _zoom, gaussian_filter as _gauss

    N = grid_resolution

    # --- Base wind flow vector (FROM dir_deg → flow points the other way) ---
    dir_rad = math.radians(wind_direction_deg)
    u_base = -wind_speed_kmh * math.sin(dir_rad)  # eastward component
    v_base = -wind_speed_kmh * math.cos(dir_rad)  # northward component

    # --- Resample elevation to N×N ----------------------------------------
    if elevation.ndim != 2 or min(elevation.shape) < 2:
        raise ValueError("elevation must be a 2D array with at least 2×2 cells")

    src_r, src_c = elevation.shape
    elev_raw = _zoom(elevation.astype(float), (N / src_r, N / src_c), order=1)
    # Fill NaN/inf with local median before smoothing.
    finite_mask = np.isfinite(elev_raw)
    fill_val = float(np.nanmedian(elev_raw[finite_mask])) if finite_mask.any() else 0.0
    elev_clean = np.where(finite_mask, elev_raw, fill_val)
    # Light smoothing reduces gradient noise from DEM quantisation artefacts.
    elev_smooth = _gauss(elev_clean, sigma=1.5)

    # --- Terrain gradient (physical units: metres / metre = dimensionless) --
    # np.gradient returns (d/drow, d/dcol) in array-index units.
    # Row increases downward (south); col increases rightward (east).
    proj_w = bounds_projected["max_x"] - bounds_projected["min_x"]
    proj_h = bounds_projected["max_y"] - bounds_projected["min_y"]
    cell_x = proj_w / (N - 1)   # metres per column step (eastward)
    cell_y = proj_h / (N - 1)   # metres per row step    (southward, i.e. −northward)

    dy_arr, dx_arr = np.gradient(elev_smooth)
    # Physical gradient: dz_dx = d(elevation)/d(easting)
    #                    dz_dy = d(elevation)/d(northing)
    # Moving one row up (+1 row) corresponds to −cell_y northing, so:
    dz_dx = dx_arr / cell_x
    dz_dy = -dy_arr / cell_y   # flip sign because row ↑ = northing ↓

    slope_magnitude = np.sqrt(dz_dx ** 2 + dz_dy ** 2)

    # --- Speed modification: faster over ridges, slower in valleys ----------
    if wind_speed_kmh > 1e-6:
        u_hat, v_hat = u_base / wind_speed_kmh, v_base / wind_speed_kmh
    else:
        u_hat = v_hat = 0.0

    # Dot product of terrain gradient with wind direction gives the slope that
    # opposes the wind: positive = uphill into wind = speed up; negative = lee.
    wind_slope = dz_dx * u_hat + dz_dy * v_hat
    p95_wind_slope = float(np.percentile(np.abs(wind_slope), 95))
    if p95_wind_slope > 1e-6:
        norm_wind_slope = np.clip(wind_slope / p95_wind_slope, -1.0, 1.0)
    else:
        norm_wind_slope = np.zeros((N, N))
    speed_factor = np.clip(1.0 + 0.3 * norm_wind_slope, 0.2, 2.0)

    # --- Direction deflection: wind curves to follow terrain contours --------
    p95_slope = float(np.percentile(slope_magnitude, 95))
    if p95_slope > 1e-6:
        deflection_strength = 0.4 * np.clip(slope_magnitude / p95_slope, 0.0, 1.0)
    else:
        deflection_strength = np.zeros((N, N))

    eps = 1e-8
    # Cross-slope unit vector: rotate gradient 90° CCW → (−dz_dy, dz_dx).
    # This points along terrain contours, tangent to the hillside.
    cross_x = -dz_dy / (slope_magnitude + eps)
    cross_y =  dz_dx / (slope_magnitude + eps)

    # --- Small-scale turbulence: smooth Gaussian noise, deterministic --------
    rng = np.random.default_rng(date_seed)
    raw_u = rng.standard_normal((N, N))
    raw_v = rng.standard_normal((N, N))
    turb_u = _gauss(raw_u, sigma=3.0) * wind_speed_kmh * 0.05
    turb_v = _gauss(raw_v, sigma=3.0) * wind_speed_kmh * 0.05

    # --- Compose final field -------------------------------------------------
    u_field = (
        u_base * speed_factor
        + deflection_strength * cross_x * wind_speed_kmh
        + turb_u
    )
    v_field = (
        v_base * speed_factor
        + deflection_strength * cross_y * wind_speed_kmh
        + turb_v
    )

    return u_field.astype(np.float32), v_field.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API — streamline tracing
# ---------------------------------------------------------------------------

def trace_streamlines(
    u_field: np.ndarray,
    v_field: np.ndarray,
    bounds_projected: dict,
    num_streamlines: int = 60,
    steps: int = 120,
    min_steps: int = 18,
    step_size: float | None = None,
    wind_direction_deg: float = 225.0,
    wind_speed_kmh: float = 19.0,
    seed: int = 42,
) -> list[list[tuple[float, float, float]]]:
    """Trace streamlines through the wind vector field using RK4 integration.

    Streamline length and density vary with local wind speed: fast zones
    produce longer, more densely seeded lines; calm zones produce short
    stubs.  This encodes wind intensity spatially in the same way the
    Apple Weather wind visualization does — the *pattern* of strokes
    communicates speed as clearly as any single-line property.

    Parameters
    ----------
    u_field, v_field:
        Wind component arrays from :func:`build_wind_field`, shape (N, N),
        values in km/h.
    bounds_projected:
        ``{min_x, max_x, min_y, max_y}`` bounding box in projected metres.
    num_streamlines:
        Total number of streamlines to trace.
    steps:
        *Maximum* integration steps for the fastest streamlines.
    min_steps:
        *Minimum* integration steps for the slowest streamlines.
        Length is interpolated between ``min_steps`` and ``steps``
        proportionally to the starting-point speed (relative to the
        field's 90th-percentile speed).  Default 18 → slow lines cover
        ~15 % of the map extent.
    step_size:
        Physical step length in metres.  Auto-computed when ``None``:
        ``max(extent_x, extent_y) / (steps * 2.5)``.
    wind_direction_deg:
        Meteorological direction (FROM) in degrees; used to identify the
        upwind edges for seed placement.
    wind_speed_kmh:
        Base wind speed used for the stagnation threshold (stop if local
        speed drops below 5 % of this value).
    seed:
        Random seed for reproducible seed-point jitter.

    Returns
    -------
    list of streamlines
        Each streamline is a list of ``(x, y, local_speed_kmh)`` tuples in
        projected metres.  Streamlines shorter than 3 points are omitted.
    """
    rng = np.random.default_rng(seed)

    min_x = bounds_projected["min_x"]
    max_x = bounds_projected["max_x"]
    min_y = bounds_projected["min_y"]
    max_y = bounds_projected["max_y"]
    proj_w = max_x - min_x
    proj_h = max_y - min_y

    if step_size is None:
        step_size = max(proj_w, proj_h) / (steps * 2.5)

    min_speed = max(0.05 * wind_speed_kmh, 0.5)  # stagnation threshold km/h

    # --- Speed magnitude field for density-weighted seeding -----------------
    speed_mag = np.sqrt(u_field.astype(float) ** 2 + v_field.astype(float) ** 2)
    # 90th-percentile speed for variable-length normalisation.
    global_p90 = float(np.percentile(speed_mag, 90))
    global_p90 = max(global_p90, wind_speed_kmh * 0.1, 1e-6)

    # --- Seed placement: 60% upwind edges, 40% speed-density interior -------
    dir_rad = math.radians(wind_direction_deg)
    u_base = -math.sin(dir_rad)   # eastward component of unit flow vector
    v_base = -math.cos(dir_rad)   # northward component

    n_edge = int(num_streamlines * 0.60)
    n_interior = num_streamlines - n_edge

    seeds: list[tuple[float, float]] = []

    # Distribute edge seeds between the two upwind sides, weighted by how
    # directly the wind hits each edge (|u| for west/east, |v| for south/north).
    abs_u = abs(u_base)
    abs_v = abs(v_base)
    total = abs_u + abs_v + 1e-9
    n_hori = max(1, round(n_edge * abs_v / total))   # south or north edge
    n_vert = n_edge - n_hori                          # west or east edge

    def _jitter(val: float, spacing: float) -> float:
        return float(val + rng.uniform(-0.1 * spacing, 0.1 * spacing))

    # Vertical upwind edge (west if u>0, east if u<0):
    edge_x = min_x if u_base >= 0 else max_x
    if n_vert > 1:
        spacing_v = proj_h / (n_vert - 1)
        for i in range(n_vert):
            y = _jitter(min_y + i * spacing_v, spacing_v)
            seeds.append((edge_x, float(np.clip(y, min_y, max_y))))
    else:
        seeds.append((edge_x, (min_y + max_y) / 2.0))

    # Horizontal upwind edge (south if v>0, north if v<0):
    edge_y = min_y if v_base >= 0 else max_y
    if n_hori > 1:
        spacing_h = proj_w / (n_hori - 1)
        for i in range(n_hori):
            x = _jitter(min_x + i * spacing_h, spacing_h)
            seeds.append((float(np.clip(x, min_x, max_x)), edge_y))
    else:
        seeds.append(((min_x + max_x) / 2.0, edge_y))

    # Interior seeds: speed-density weighted so high-velocity zones receive
    # proportionally more seeds and therefore more strokes.
    # Sub-linear exponent (0.55) keeps some coverage in calm areas.
    _N_grid = speed_mag.shape[0]
    flat_w = speed_mag.flatten() ** 0.55
    flat_w = flat_w / flat_w.sum()
    flat_idx = rng.choice(len(flat_w), size=n_interior, p=flat_w.astype(float))
    cell_sz = max(proj_w, proj_h) / max(1, _N_grid - 1)
    for idx in flat_idx:
        row_g = int(idx) // _N_grid
        col_g = int(idx) % _N_grid
        gx = min_x + (col_g / max(1, _N_grid - 1)) * proj_w
        gy = max_y - (row_g / max(1, _N_grid - 1)) * proj_h
        # Jitter within ±15 % of one grid cell to break up regularity.
        gx += float(rng.uniform(-0.15 * cell_sz, 0.15 * cell_sz))
        gy += float(rng.uniform(-0.15 * cell_sz, 0.15 * cell_sz))
        seeds.append((float(np.clip(gx, min_x, max_x)), float(np.clip(gy, min_y, max_y))))

    # --- Trace each seed with RK4 -------------------------------------------
    streamlines: list[list[tuple[float, float, float]]] = []

    for sx, sy in seeds:
        # Variable length: starting speed determines how many steps this
        # streamline gets.  Fast start → full budget; calm start → min_steps.
        u0, v0 = _sample_wind(u_field, v_field, sx, sy, bounds_projected)
        s0 = math.sqrt(u0 * u0 + v0 * v0)
        s0_norm = min(1.0, s0 / global_p90)
        this_steps = int(min_steps + s0_norm * (steps - min_steps))

        x, y = sx, sy
        trail: list[tuple[float, float, float]] = []

        for _ in range(this_steps):
            # Stagnation check at current position.
            u, v = _sample_wind(u_field, v_field, x, y, bounds_projected)
            spd = math.sqrt(u * u + v * v)
            if spd < min_speed:
                break

            trail.append((x, y, spd))

            # RK4 advance.
            nx, ny, _ = _rk4_step(u_field, v_field, x, y, bounds_projected, step_size)

            # Bounds check — stop if we've left the map.
            if not (min_x <= nx <= max_x and min_y <= ny <= max_y):
                # Record the last in-bounds point, then stop.
                trail.append((nx, ny, spd))
                break

            x, y = nx, ny

        if len(trail) >= 3:
            streamlines.append(trail)

    return streamlines


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def save_streamlines_to_file(
    streamlines: list[list[tuple[float, float, float]]],
    path: str | Path,
) -> None:
    """Write streamlines to a JSON file.

    Each streamline is serialised as a list of ``[x, y, speed]`` arrays.
    """
    data = [[[float(v) for v in pt] for pt in stream] for stream in streamlines]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))


def load_streamlines_from_file(
    path: str | Path,
) -> list[list[tuple[float, float, float]]]:
    """Load streamlines previously saved by :func:`save_streamlines_to_file`.

    Returns a list of streamlines, each a list of ``(x, y, speed)`` tuples.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return [
        [(float(pt[0]), float(pt[1]), float(pt[2])) for pt in stream]
        for stream in data
    ]
