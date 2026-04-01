"""
dem_fetcher.py — Fetch and read elevation GeoTIFFs from multiple free sources.

Primary:  USGS 3DEP Elevation ImageServer (US coverage only).
Fallback: AWS Terrain Tiles / Mapzen Terrarium (global SRTM/NASADEM coverage,
          no authentication required).

Both sources are open-access and free.
"""

from __future__ import annotations

import io as _io
import math as _math
import sys
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.crs import CRS as _CRS
from rasterio.transform import from_bounds as _from_bounds
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


# ---------------------------------------------------------------------------
# AWS Terrain Tiles — global free DEM (SRTM/NASADEM, no auth required)
# ---------------------------------------------------------------------------
# Source:  https://registry.opendata.aws/terrain-tiles/
# Format:  Terrarium RGB PNG — elevation (m) = R×256 + G + B/256 − 32768
# Coverage: Global.  SRTM-derived below 60°N/S; NASADEM elsewhere.
# Zoom 13 gives ≈ 10 m/px at mid-latitudes — suitable for hikes up to ~20 km.
# ---------------------------------------------------------------------------

_AWS_TERRARIUM_URL = (
    "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png"
)
_TILE_PX = 256   # each tile is 256 × 256 pixels


def _slippy_tile(lng: float, lat: float, zoom: int) -> tuple[int, int]:
    """Convert WGS84 (lng, lat) to slippy-map tile (x, y) at *zoom*."""
    n = 2 ** zoom
    x = int((lng + 180.0) / 360.0 * n)
    lat_r = _math.radians(max(-85.051129, min(85.051129, lat)))
    y = int(
        (1.0 - _math.log(_math.tan(lat_r) + 1.0 / _math.cos(lat_r)) / _math.pi)
        / 2.0 * n
    )
    return x, y


def _tile_nw_corner(tx: int, ty: int, zoom: int) -> tuple[float, float]:
    """Return the (lng, lat) of a tile's north-west (top-left) corner."""
    n = 2 ** zoom
    lng = tx / n * 360.0 - 180.0
    lat = _math.degrees(_math.atan(_math.sinh(_math.pi * (1 - 2 * ty / n))))
    return lng, lat


def _fetch_dem_aws_terrarium(
    bounds: dict,
    output_path: str,
    zoom: int = 13,
) -> str:
    """Download and stitch AWS Terrarium tiles for *bounds* into a GeoTIFF.

    Parameters
    ----------
    bounds:
        WGS84 bounding box: ``min_lat``, ``max_lat``, ``min_lng``, ``max_lng``.
    output_path:
        Destination GeoTIFF path.
    zoom:
        Tile zoom level.  13 is a good default for hike-scale areas (< 20 km);
        use 12 for larger regions, 14 for very detailed small areas.

    Returns
    -------
    str
        The *output_path* that was written.
    """
    from PIL import Image as _PILImage  # lazy import — PIL is optional dep

    # Tile indices for NW and SE corners of the bounding box.
    tx0, ty0 = _slippy_tile(bounds["min_lng"], bounds["max_lat"], zoom)  # NW
    tx1, ty1 = _slippy_tile(bounds["max_lng"], bounds["min_lat"], zoom)  # SE
    # y increases southward in tile coords; clamp to valid range.
    tx1, ty1 = max(tx1, tx0), max(ty1, ty0)

    n_tx = tx1 - tx0 + 1
    n_ty = ty1 - ty0 + 1
    stitched = np.zeros((n_ty * _TILE_PX, n_tx * _TILE_PX), dtype=np.float32)

    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            url = _AWS_TERRARIUM_URL.format(z=zoom, x=tx, y=ty)
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            img = _PILImage.open(_io.BytesIO(resp.content)).convert("RGB")
            r, g, b = (np.array(ch, dtype=np.float32) for ch in img.split())
            elev = r * 256.0 + g + b / 256.0 - 32768.0
            row_off = (ty - ty0) * _TILE_PX
            col_off = (tx - tx0) * _TILE_PX
            stitched[row_off : row_off + _TILE_PX, col_off : col_off + _TILE_PX] = elev

    # Geographic extent of the stitched grid in WGS84.
    west, north = _tile_nw_corner(tx0,      ty0,      zoom)
    east, south = _tile_nw_corner(tx1 + 1,  ty1 + 1,  zoom)

    # Write GeoTIFF with an equirectangular transform.  Terrarium tiles are
    # internally Web Mercator, so pixel lat-spacing is slightly non-uniform;
    # the equirectangular approximation error is < 0.5% for areas < 50 km.
    transform = _from_bounds(west, south, east, north,
                             stitched.shape[1], stitched.shape[0])
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(out), "w",
        driver="GTiff",
        height=stitched.shape[0],
        width=stitched.shape[1],
        count=1,
        dtype="float32",
        crs=_CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(stitched, 1)

    return str(out)


def fetch_dem_global(
    bounds: dict,
    output_path: str,
    resolution: int = 1024,
) -> str:
    """Fetch a DEM for *bounds*, trying free sources in priority order.

    1. **USGS 3DEP** — highest quality for US coverage; fails gracefully
       for non-US bounds or network issues.
    2. **AWS Terrain Tiles** (Terrarium / SRTM/NASADEM) — global coverage,
       no authentication, ~10 m resolution at zoom 13.

    Parameters
    ----------
    bounds:
        WGS84 bounding box: ``min_lat``, ``max_lat``, ``min_lng``, ``max_lng``.
    output_path:
        Where to write the GeoTIFF.
    resolution:
        Pixel size hint passed to USGS (ignored for AWS tiles).

    Returns
    -------
    str
        ``(output_path, source)`` where *source* is ``"usgs"`` or ``"aws"``.

    Raises
    ------
    RuntimeError
        If all sources fail.
    """
    # 1 — USGS 3DEP (US only, but try first: higher resolution when available)
    try:
        fetch_dem(bounds, output_path, resolution=resolution)
        return "usgs"
    except Exception:
        pass

    # 2 — AWS Terrain Tiles (global SRTM/NASADEM, always free, no auth)
    # Auto-select zoom based on the shorter bounding-box dimension.
    span_deg = min(
        bounds["max_lng"] - bounds["min_lng"],
        bounds["max_lat"] - bounds["min_lat"],
    )
    # Target ~512 px across the shorter side; clamp to sane zoom range.
    zoom = max(10, min(14, round(_math.log2(360.0 / span_deg) - 1)))
    try:
        _fetch_dem_aws_terrarium(bounds, output_path, zoom=zoom)
        return "aws"
    except Exception as exc:
        raise RuntimeError(
            f"All DEM sources failed.  Last error: {exc}"
        ) from exc


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
