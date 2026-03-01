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

    # Apply just enough smoothing to suppress single-pixel DEM noise.
    # preserve_geology=True → σ=0.3 (nearly raw); False → σ=1.0 (old default).
    sigma = 0.3 if preserve_geology else 1.0
    nan_mask = np.isnan(elevation)
    filled_for_smooth = np.where(nan_mask, 0.0, elevation)
    weights = np.where(nan_mask, 0.0, 1.0)
    smoothed_data = gaussian_filter(filled_for_smooth.astype(np.float64), sigma=sigma)
    smoothed_weights = gaussian_filter(weights.astype(np.float64), sigma=sigma)
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
