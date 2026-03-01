"""
dem_fetcher.py — Fetch and read USGS 3DEP elevation GeoTIFFs.

Downloads Digital Elevation Model (DEM) tiles from the USGS 3DEP Elevation
ImageServer and exposes helpers for reading pixel values back as numpy arrays.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.transform import rowcol


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USGS_ENDPOINT = (
    "https://elevation.nationalmap.gov/arcgis/rest/services"
    "/3DEPElevation/ImageServer/exportImage"
)
_REQUEST_TIMEOUT_S = 60


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_dem(
    bounds: dict,
    output_path: str,
    resolution: int = 1024,
) -> str:
    """Download a 3DEP elevation GeoTIFF for the given WGS84 bounding box.

    Parameters
    ----------
    bounds:
        Dict with keys ``min_lat``, ``max_lat``, ``min_lng``, ``max_lng``
        in decimal degrees (WGS84).
    output_path:
        File path where the downloaded GeoTIFF will be saved.
    resolution:
        Pixel width *and* height of the requested image.  Defaults to 1024.
        Higher values (1024, 2048) produce sharper contours at the cost of
        longer fetch and processing times.  2048 is the maximum the USGS
        3DEP ImageServer supports in a single request.

    Returns
    -------
    str
        The ``output_path`` that was written.

    Raises
    ------
    requests.RequestException
        On any network-level failure.
    ValueError
        If the server returns an HTML error page instead of a TIFF.
    """
    params = {
        "bbox": f"{bounds['min_lng']},{bounds['min_lat']},{bounds['max_lng']},{bounds['max_lat']}",
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": f"{resolution},{resolution}",
        "format": "tiff",
        "pixelType": "F32",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }

    try:
        response = requests.get(_USGS_ENDPOINT, params=params, timeout=_REQUEST_TIMEOUT_S)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise requests.RequestException(
            f"Failed to fetch DEM from USGS 3DEP: {exc}"
        ) from exc

    content_type = response.headers.get("Content-Type", "")
    if "tiff" not in content_type.lower():
        preview = response.text[:500].strip()
        raise ValueError(
            f"USGS returned unexpected Content-Type '{content_type}' "
            f"(expected image/tiff).\nResponse preview:\n{preview}"
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(response.content)
    return str(out)


def load_dem(tiff_path: str) -> tuple[np.ndarray, dict]:
    """Open a GeoTIFF and return the elevation array plus metadata.

    Parameters
    ----------
    tiff_path:
        Path to a GeoTIFF file as written by :func:`fetch_dem`.

    Returns
    -------
    elevation : np.ndarray
        2-D float32 array of elevation values in metres.  NoData cells are
        replaced with ``np.nan``.
    metadata : dict
        Keys:

        * ``transform`` – :class:`rasterio.transform.Affine`
        * ``crs`` – :class:`rasterio.crs.CRS`
        * ``bounds`` – :class:`rasterio.coords.BoundingBox`
        * ``shape`` – ``(rows, cols)`` tuple
        * ``resolution`` – ``(x_res, y_res)`` in CRS units per pixel

    Raises
    ------
    FileNotFoundError
        If *tiff_path* does not exist.
    rasterio.errors.RasterioIOError
        If the file cannot be opened as a raster.
    """
    path = Path(tiff_path)
    if not path.exists():
        raise FileNotFoundError(f"DEM file not found: {tiff_path}")

    with rasterio.open(path) as src:
        elevation = src.read(1).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        shape = (src.height, src.width)
        resolution = (abs(transform.a), abs(transform.e))

    if nodata is not None:
        elevation[elevation == nodata] = np.nan

    metadata = {
        "transform": transform,
        "crs": crs,
        "bounds": bounds,
        "shape": shape,
        "resolution": resolution,
    }
    return elevation, metadata


def get_elevation_at_point(
    elevation: np.ndarray,
    transform,
    lat: float,
    lng: float,
) -> float:
    """Return the elevation value at a geographic coordinate via bilinear interpolation.

    Parameters
    ----------
    elevation:
        2-D array as returned by :func:`load_dem`.
    transform:
        Affine transform from the same :func:`load_dem` metadata dict.
    lat:
        Latitude in decimal degrees.
    lng:
        Longitude in decimal degrees.

    Returns
    -------
    float
        Elevation in metres, or ``float('nan')`` if the point falls outside
        the raster extent or on a NoData cell.

    Raises
    ------
    ValueError
        If the coordinate is outside the raster bounds.
    """
    rows, cols = elevation.shape

    # Fractional pixel position (rasterio returns integer row/col via rowcol)
    col_f = (lng - transform.c) / transform.a
    row_f = (lat - transform.f) / transform.e

    # Bounds check
    if not (0 <= col_f <= cols - 1 and 0 <= row_f <= rows - 1):
        raise ValueError(
            f"Coordinate (lat={lat}, lng={lng}) is outside the raster extent."
        )

    # Bilinear interpolation
    row0, col0 = int(row_f), int(col_f)
    row1, col1 = min(row0 + 1, rows - 1), min(col0 + 1, cols - 1)
    dr, dc = row_f - row0, col_f - col0

    v00 = float(elevation[row0, col0])
    v01 = float(elevation[row0, col1])
    v10 = float(elevation[row1, col0])
    v11 = float(elevation[row1, col1])

    # Return nan if any contributing cell is nan
    if any(np.isnan(v) for v in (v00, v01, v10, v11)):
        return float("nan")

    return (
        v00 * (1 - dr) * (1 - dc)
        + v01 * (1 - dr) * dc
        + v10 * dr * (1 - dc)
        + v11 * dr * dc
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    # Rocky Mountain National Park, CO — representative high-elevation test area
    # (elevations roughly 1 240 – 4 345 m, giving a meaningful range printout)
    TEST_BOUNDS = {
        "min_lat": 40.30,
        "max_lat": 40.50,
        "min_lng": -105.80,
        "max_lng": -105.50,
    }
    # Spot-check: Longs Peak summit (4 346 m)
    SPOT_LAT, SPOT_LNG = 40.2550, -105.6153
    SPOT_LABEL = "Longs Peak"

    print("Fetching DEM from USGS 3DEP …")
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tiff_path = tmp.name

    try:
        fetch_dem(TEST_BOUNDS, tiff_path)

        elevation, meta = load_dem(tiff_path)
        valid = elevation[~np.isnan(elevation)]

        rows, cols = meta["shape"]
        print(f"Elevation grid : ({rows}, {cols}), "
              f"range: {valid.min():.1f}m to {valid.max():.1f}m")
        print(f"Mean elevation : {valid.mean():.1f} m")
        print(f"CRS            : {meta['crs']}")
        print(f"Resolution     : {meta['resolution'][0]:.6f} x {meta['resolution'][1]:.6f} deg/px")

        try:
            spot = get_elevation_at_point(elevation, meta["transform"], SPOT_LAT, SPOT_LNG)
            print(f"{SPOT_LABEL:<15}: {spot:.1f} m")
        except ValueError as exc:
            print(f"{SPOT_LABEL} outside raster bounds: {exc}")

    except (ValueError, requests.RequestException) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
