"""dem_features.py — DEM-derived geographic features: peaks and lake polygons.

Used as a fallback when OSM data is unavailable (e.g. offline / no network).
All returned coordinates are in WGS-84 (lat, lng decimal degrees).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------


def detect_peaks(
    elevation: np.ndarray,
    bounds_projected: dict,
    transformer,
    min_prominence_m: float = 60.0,
    neighborhood_cells: int = 8,
    max_peaks: int = 10,
    dem_transform=None,
) -> list:
    """Return prominent local peaks as :class:`~field_atlas.enrichment.features.MapFeature` objects.

    Parameters
    ----------
    elevation:
        2-D float DEM array, values in metres.
    bounds_projected:
        ``{min_x, max_x, min_y, max_y}`` projected bounds (metres).
    transformer:
        WGS-84 → UTM pyproj Transformer (used in reverse).
    min_prominence_m:
        Minimum required prominence (metres above surrounding minimum).
    neighborhood_cells:
        Half-size of the local-maximum search window in grid cells.
    max_peaks:
        Cap on the number of peaks returned (highest first).
    """
    from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
    from .features import MapFeature

    elev = gaussian_filter(elevation.astype(float), sigma=1.5)
    N, M = elev.shape
    pw = bounds_projected["max_x"] - bounds_projected["min_x"]
    ph = bounds_projected["max_y"] - bounds_projected["min_y"]

    size = 2 * neighborhood_cells + 1
    local_max = maximum_filter(elev, size=size)
    is_max = elev == local_max

    # Remove map-edge artefacts
    m = neighborhood_cells + 2
    is_max[:m, :] = is_max[-m:, :] = is_max[:, :m] = is_max[:, -m:] = False

    # Prominence = elevation above minimum within a 3× larger window
    prom_size = size * 3
    local_min = minimum_filter(elev, size=prom_size)
    prominence = elev - local_min

    peaks = []
    rows, cols = np.where(is_max)
    for row, col in zip(rows, cols):
        if float(prominence[row, col]) < min_prominence_m:
            continue
        if dem_transform is not None:
            lng, lat = dem_transform * (float(col), float(row))
        else:
            x = bounds_projected["min_x"] + col * pw / max(M - 1, 1)
            y = bounds_projected["max_y"] - row * ph / max(N - 1, 1)
            lat, lng = transformer.transform(x, y, direction="INVERSE")
        elev_m = float(elev[row, col])
        peaks.append(MapFeature(
            name=f"{int(round(elev_m))} m",
            feature_type="peak",
            lat=float(lat),
            lng=float(lng),
            elevation_m=elev_m,
            osm_id=-(900_000 + len(peaks)),
        ))

    peaks.sort(key=lambda p: -(p.elevation_m or 0))
    return peaks[:max_peaks]


# ---------------------------------------------------------------------------
# Lake detection
# ---------------------------------------------------------------------------


def detect_lakes(
    elevation: np.ndarray,
    bounds_projected: dict,
    transformer,
    min_area_m2: float = 25_000.0,
    max_area_m2: float = 20_000_000.0,
    dem_transform=None,
) -> list[list[list[float]]]:
    """Return water-body polygons detected from flat depressions in the DEM.

    Each polygon is a closed ring of ``[lat, lng]`` pairs (WGS-84), matching
    the ``water_areas`` format expected by :class:`~field_atlas.enrichment.osm_vectors.OSMVectors`.

    The algorithm:
    1. Compute slope magnitude from the smoothed DEM.
    2. Label connected flat pixels (slope below an adaptive threshold).
    3. Keep only regions that are basin-like: surrounding terrain is at least
       8 m higher than the region mean.
    4. Fit a convex hull and convert to lat/lng.

    Parameters
    ----------
    elevation:
        2-D float DEM array, values in metres.
    bounds_projected:
        ``{min_x, max_x, min_y, max_y}`` in projected metres.
    transformer:
        WGS-84 → UTM pyproj Transformer (used in reverse).
    min_area_m2:
        Minimum lake area to include (default 25 000 m² ≈ 5 acres).
    max_area_m2:
        Maximum area; excludes oceans and large coastal flats.
    """
    from scipy.ndimage import gaussian_filter, label, binary_dilation
    from scipy.spatial import ConvexHull

    elev = gaussian_filter(elevation.astype(float), sigma=2.0)
    N, M = elev.shape
    pw = bounds_projected["max_x"] - bounds_projected["min_x"]
    ph = bounds_projected["max_y"] - bounds_projected["min_y"]
    cell_w = pw / max(M - 1, 1)
    cell_h = ph / max(N - 1, 1)
    cell_area = cell_w * cell_h

    dy_arr, dx_arr = np.gradient(elev)
    slope = np.sqrt((dx_arr / cell_w) ** 2 + (-dy_arr / cell_h) ** 2)

    # Adaptive flat threshold: 3 % of the 75th-percentile slope
    thresh = max(0.003, 0.03 * float(np.percentile(slope, 75)))
    flat = slope < thresh

    # Trim map edges (DEM boundary artefacts are always flat)
    flat[:4, :] = flat[-4:, :] = flat[:, :4] = flat[:, -4:] = False

    labeled, n_regions = label(flat)
    water_polys: list[list[list[float]]] = []

    for i in range(1, n_regions + 1):
        region = labeled == i
        area = float(region.sum()) * cell_area
        if not (min_area_m2 <= area <= max_area_m2):
            continue

        rows, cols = np.where(region)
        region_mean = float(elev[rows, cols].mean())

        # Require a genuine basin: surrounding band is clearly higher
        border = binary_dilation(region, iterations=6) & ~region
        br, bc = np.where(border)
        if len(br) < 6:
            continue
        if float(elev[br, bc].mean()) - region_mean < 4.0:
            continue

        # Convex hull polygon → lat/lng using DEM affine if available
        if dem_transform is not None:
            # Use actual DEM pixel→WGS84 transform for correct positioning
            lngs_arr = np.array([dem_transform * (float(c), float(r)) for c, r in zip(cols, rows)])
            hull_pts = lngs_arr  # shape (N, 2) with (lng, lat) pairs
            if len(hull_pts) < 4:
                continue
            try:
                hull = ConvexHull(hull_pts)
            except Exception:
                continue
            poly: list[list[float]] = []
            for hi in hull.vertices:
                lng_v, lat_v = float(hull_pts[hi, 0]), float(hull_pts[hi, 1])
                poly.append([lat_v, lng_v])
        else:
            xs = bounds_projected["min_x"] + cols * pw / max(M - 1, 1)
            ys = bounds_projected["max_y"] - rows * ph / max(N - 1, 1)
            pts = np.column_stack([xs, ys])
            if len(pts) < 4:
                continue
            try:
                hull = ConvexHull(pts)
            except Exception:
                continue
            poly: list[list[float]] = []
            for px, py in pts[hull.vertices]:
                lat, lng = transformer.transform(px, py, direction="INVERSE")
                poly.append([float(lat), float(lng)])
        poly.append(poly[0])  # close the ring
        water_polys.append(poly)

    return water_polys
