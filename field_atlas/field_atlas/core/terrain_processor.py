"""
terrain_processor.py — Derive terrain layers from raw elevation grids.

Takes a 2-D numpy elevation array (as produced by dem_fetcher.load_dem) and
produces the processed layers needed by the rendering pipeline: contour lines,
hillshade, slope, and a smoothed elevation surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_contours(
    elevation: np.ndarray,
    transform,
    interval_m: float = 20.0,
    preserve_geology: bool = True,
) -> list[dict]:
    """Extract contour lines from an elevation grid.

    Parameters
    ----------
    elevation:
        2-D float array of elevation values in metres (NaN = no-data).
    transform:
        Affine transform (``rasterio.transform.Affine``) mapping pixel indices
        to CRS coordinates:
        ``x = transform.a * col + transform.b * row + transform.c``
        ``y = transform.d * col + transform.e * row + transform.f``
    interval_m:
        Vertical interval between contour levels in metres.
    preserve_geology:
        When ``True`` (default), only a minimal Gaussian blur (σ = 0.3 px) is
        applied before contouring — enough to suppress single-pixel DEM noise
        while keeping ridgeline V-shapes and drainage-valley inflections intact.
        When ``False``, a heavier σ = 1.0 blur is used, which rounds corners
        and produces balloon-like contours.

    Returns
    -------
    list[dict]
        Sorted ascending by elevation.  Each dict has:

        * ``elevation`` (float) – contour level in metres
        * ``paths`` (list[list[[x, y]]]) – one entry per disconnected segment;
          each entry is a list of ``[x, y]`` pairs in the input CRS.
    """
    valid = elevation[~np.isnan(elevation)]
    if valid.size == 0:
        return []

    z_min = float(valid.min())
    z_max = float(valid.max())

    # preserve_geology=True: use the raw grid so ridgeline V-shapes and
    # drainage inflections survive into the contour paths unchanged.
    # preserve_geology=False: σ=1.0 blur (old default, rounder/smoother).
    if preserve_geology:
        elev_work = elevation.copy().astype(np.float32)
    else:
        nan_mask = np.isnan(elevation)
        filled_for_smooth = np.where(nan_mask, 0.0, elevation)
        weights = np.where(nan_mask, 0.0, 1.0)
        smoothed_data = gaussian_filter(filled_for_smooth.astype(np.float64), sigma=1.0)
        smoothed_weights = gaussian_filter(weights.astype(np.float64), sigma=1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            elev_work = (smoothed_data / smoothed_weights).astype(np.float32)
        elev_work[smoothed_weights == 0.0] = np.nan

    # First level is the nearest interval boundary at or above z_min.
    first_level = np.ceil(z_min / interval_m) * interval_m
    levels = np.arange(first_level, z_max, interval_m)

    # Fill NaN with a sentinel below all levels so skimage treats those cells
    # as "outside" and doesn't generate spurious contours around no-data gaps.
    fill_val = z_min - 1.0
    elev_filled = np.where(np.isnan(elev_work), fill_val, elev_work)

    results = []
    for level in levels:
        raw_paths = find_contours(elev_filled, level=float(level))
        if not raw_paths:
            continue

        geo_paths = []
        for path in raw_paths:
            # path is (N, 2) of (row, col) float values from skimage.
            # Vectorised affine transform to CRS coordinates.
            rows_arr = path[:, 0]
            cols_arr = path[:, 1]
            xs = transform.a * cols_arr + transform.b * rows_arr + transform.c
            ys = transform.d * cols_arr + transform.e * rows_arr + transform.f
            geo_paths.append([[float(x), float(y)] for x, y in zip(xs, ys)])

        results.append({"elevation": float(level), "paths": geo_paths})

    return sorted(results, key=lambda c: c["elevation"])


def generate_hillshade(
    elevation: np.ndarray,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> np.ndarray:
    """Compute a shaded-relief (hillshade) array.

    Uses the standard cartographic formula::

        I = cos(z)*cos(S) + sin(z)*sin(S)*cos(A - aspect)

    where *z* is the solar zenith angle, *S* is the terrain slope angle, and
    *A* is the solar azimuth angle.  Both *A* and *aspect* are expressed in
    the same clockwise-from-north convention so the difference is meaningful.

    Parameters
    ----------
    elevation:
        2-D float array of elevation values in metres.
    azimuth:
        Solar azimuth in degrees, clockwise from north (default 315° = NW,
        the standard cartographic convention).
    altitude:
        Solar elevation angle above the horizon in degrees (0–90).

    Returns
    -------
    np.ndarray
        Float32 array, same shape as *elevation*, values in [0, 1].
        Cells that were NaN in the input remain NaN.
    """
    zenith_rad = np.radians(90.0 - altitude)
    azimuth_rad = np.radians(azimuth)

    # np.gradient returns (d/d_row, d/d_col); spacing=1 is fine because only
    # the ratio between axes matters for slope angle (and we assume square pixels).
    dy, dx = np.gradient(elevation)

    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))

    # Aspect in cartographic convention (clockwise from north).
    # arctan2(-dy, dx) gives math-convention bearing; subtracting from π/2
    # converts to clockwise-from-north.
    aspect_rad = np.pi / 2.0 - np.arctan2(-dy, dx)

    hillshade = (
        np.cos(zenith_rad) * np.cos(slope_rad)
        + np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    )

    hillshade = np.clip(hillshade, 0.0, 1.0)
    hillshade[np.isnan(elevation)] = np.nan
    return hillshade.astype(np.float32)


def generate_multidirectional_hillshade(
    elevation: np.ndarray,
) -> np.ndarray:
    """Compute a multi-directional hillshade with slope-based ambient occlusion.

    Combines three illumination angles so every slope face has texture rather
    than only the NW-facing aspects lit by a single source:

    * Primary  NW 315° / 45° — standard cartographic convention (weight 0.55)
    * Cross-light NE 45° / 25° — low-angle cross-lighting reveals rock texture
      on east-facing and ridge faces (weight 0.20)
    * Diffuse fill 315° / 80° — near-overhead sky, keeps shadows from pure
      black and simulates diffuse sky light (weight 0.25)

    A slope-based ambient occlusion term is then multiplied in: steep terrain
    (cliffs, rugged ridges) is darkened uniformly, simulating sky-light
    occlusion in deep valleys.  Flat areas are unaffected.

    Parameters
    ----------
    elevation:
        2-D float array of elevation values in metres (NaN = no-data).

    Returns
    -------
    np.ndarray
        Float32 array, same shape as *elevation*, values in [0, 1].
        Values near 0 are deep shadow; values near 1 are fully lit.
        NaN cells in the input remain NaN.
    """
    hs_primary    = generate_hillshade(elevation, azimuth=315.0, altitude=45.0)
    hs_crosslight = generate_hillshade(elevation, azimuth=45.0,  altitude=25.0)
    hs_fill       = generate_hillshade(elevation, azimuth=315.0, altitude=80.0)

    combined = (
        0.55 * np.where(np.isnan(hs_primary),    0.0, hs_primary)
        + 0.20 * np.where(np.isnan(hs_crosslight), 0.0, hs_crosslight)
        + 0.25 * np.where(np.isnan(hs_fill),       0.0, hs_fill)
    )

    # Slope-based ambient occlusion.
    # np.gradient returns dz/d_pixel; ratio of axes is what matters.
    dy, dx = np.gradient(np.where(np.isnan(elevation), 0.0, elevation))
    # Normalise slope to [0, 1] where 1 ≈ 45° and beyond.
    slope_norm = np.clip(np.arctan(np.sqrt(dx**2 + dy**2)) / (np.pi / 4.0), 0.0, 1.0)
    # AO darkens steep terrain by up to 30%; flat areas untouched.
    ao = 1.0 - 0.30 * slope_norm
    combined = combined * ao

    combined = np.clip(combined, 0.0, 1.0)
    combined[np.isnan(elevation)] = np.nan
    return combined.astype(np.float32)


def generate_hypsometric_rgba(
    elevation: np.ndarray,
    palette: str = "alpine",
) -> np.ndarray:
    """Build an RGBA elevation-tint (hypsometric) image.

    Maps normalised elevation to a colour ramp, producing a (rows, cols, 4)
    uint8 array that can be embedded as a PNG in the SVG renderer.  The alpha
    channel encodes opacity so the tint fades out at the lowest elevations,
    keeping valleys and flatlands clean while high terrain takes on colour.

    Palettes
    --------
    ``"alpine"``
        Cool blue-grey ramp as seen in East of Nowhere mountain relief prints.
        Lowlands are near-transparent cream; summits become deep slate-blue.
    ``"topo"``
        Warm earth tones: ochre lowlands → russet highlands, evocative of
        historical USGS topo overlays.

    Parameters
    ----------
    elevation:
        2-D float array of elevation in metres.
    palette:
        ``"alpine"`` (default) or ``"topo"``.

    Returns
    -------
    np.ndarray
        uint8 array of shape ``(rows, cols, 4)`` — RGBA.  NaN cells are fully
        transparent.
    """
    valid = elevation[~np.isnan(elevation)]
    if valid.size == 0:
        return np.zeros((*elevation.shape, 4), dtype=np.uint8)

    z_min = float(valid.min())
    z_max = float(valid.max())
    z_range = z_max - z_min if z_max > z_min else 1.0

    # t ∈ [0, 1]: 0 = lowest point, 1 = highest
    t = np.clip((elevation - z_min) / z_range, 0.0, 1.0)

    # --- colour ramps defined as (t_stop, R, G, B, A) ---
    if palette == "topo":
        stops = [
            (0.00,  245, 235, 215,   0),   # cream  — fully transparent at base
            (0.25,  230, 210, 170,  60),   # warm sand
            (0.50,  200, 170, 120,  90),   # ochre
            (0.75,  170, 120,  75, 115),   # russet
            (1.00,  130,  75,  40, 140),   # deep brown
        ]
    else:  # "alpine" — cool blue-grey
        stops = [
            (0.00,  240, 242, 245,   0),   # near-white — fully transparent at base
            (0.20,  210, 220, 230,  40),   # pale blue-grey
            (0.45,  160, 185, 205,  85),   # mid blue-grey
            (0.70,  100, 140, 175, 120),   # cool steel
            (1.00,   55,  90, 130, 150),   # deep slate-blue
        ]

    t_stops = np.array([s[0] for s in stops], dtype=np.float32)
    rgba_stops = np.array([[s[1], s[2], s[3], s[4]] for s in stops], dtype=np.float32)

    rows, cols = elevation.shape
    t_flat = t.ravel()

    result_flat = np.zeros((t_flat.size, 4), dtype=np.float32)
    for ch in range(4):
        result_flat[:, ch] = np.interp(t_flat, t_stops, rgba_stops[:, ch])

    result = result_flat.reshape(rows, cols, 4).astype(np.uint8)

    # NaN cells → fully transparent
    nan_mask = np.isnan(elevation)
    result[nan_mask, 3] = 0

    return result


def compute_slope(elevation: np.ndarray, cellsize: float) -> np.ndarray:
    """Compute terrain slope in degrees.

    Parameters
    ----------
    elevation:
        2-D float array of elevation values in metres.
    cellsize:
        Ground-sample distance in metres per pixel (assumed equal in x and y).

    Returns
    -------
    np.ndarray
        Float32 array of slope values in degrees, same shape as *elevation*.
        Cells that were NaN in the input remain NaN.
    """
    # Passing cellsize to np.gradient gives dz/dm (rise over run in m/m).
    dy, dx = np.gradient(elevation, cellsize)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)
    slope_deg[np.isnan(elevation)] = np.nan
    return slope_deg


def smooth_elevation(elevation: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing to reduce DEM noise.

    NaN cells are handled with a normalised-weight approach: the filter is
    applied to both the data (NaN→0) and a binary validity mask, then divided
    element-wise so that no-data regions do not bleed into valid areas.

    Parameters
    ----------
    elevation:
        2-D float array of elevation values in metres.
    sigma:
        Standard deviation of the Gaussian kernel in pixels.

    Returns
    -------
    np.ndarray
        Smoothed float32 array, same shape as *elevation*.  Cells with zero
        valid weight (entirely surrounded by NaN within the kernel) remain NaN.
    """
    nan_mask = np.isnan(elevation)
    filled = np.where(nan_mask, 0.0, elevation)
    weights = np.where(nan_mask, 0.0, 1.0)

    smoothed_data = gaussian_filter(filled.astype(np.float64), sigma=sigma)
    smoothed_weights = gaussian_filter(weights.astype(np.float64), sigma=sigma)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (smoothed_data / smoothed_weights).astype(np.float32)

    result[smoothed_weights == 0.0] = np.nan
    return result


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import rasterio

    if len(sys.argv) != 2:
        print(
            "Usage: python -m field_atlas.core.terrain_processor <dem.tif>",
            file=sys.stderr,
        )
        sys.exit(1)

    tiff_path = sys.argv[1]
    if not Path(tiff_path).exists():
        print(f"ERROR: file not found: {tiff_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading DEM : {tiff_path}")
    with rasterio.open(tiff_path) as src:
        elev = src.read(1).astype(np.float32)
        tf = src.transform
        nodata = src.nodata

    if nodata is not None:
        elev[elev == nodata] = np.nan

    valid = elev[~np.isnan(elev)]
    print(f"Grid shape  : {elev.shape}")
    print(f"Elev range  : {valid.min():.1f} m  to  {valid.max():.1f} m")
    print()

    contours = generate_contours(elev, tf, interval_m=20.0)
    print(f"Contours at 20 m intervals — {len(contours)} level(s):")
    for c in contours:
        n_paths = len(c["paths"])
        total_pts = sum(len(p) for p in c["paths"])
        print(
            f"  {c['elevation']:>8.1f} m  —  "
            f"{n_paths:>3} path(s), {total_pts:>6} point(s)"
        )
