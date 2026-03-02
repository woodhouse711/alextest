# Field Atlas — Development Report

**Prepared:** March 2, 2026
**Repository:** woodhouse711/alextest (`field_atlas/`)
**Branch:** `claude/setup-field-atlas-TOebe`

---

## 1. Project Overview

Field Atlas is a command-line tool that converts GPX hiking tracks into print-quality
cartographic artifacts. Given a GPX file, the tool fetches elevation data, processes
terrain geometry, and renders a poster-sized map with contour lines, hillshade, OSM
vector layers, and a speed-gradient route line. Output is an 18×24 in (or 24×18 in
landscape) SVG with a simultaneous PDF export.

The stated output formats are SVG, PDF, and STL (3D mesh). Only SVG and PDF are
currently functional; STL is a placeholder.

### Core Pipeline

```
GPX file
  ↓ parse_gpx()             — haversine distance, elevation gain, segment speeds
  ↓ padded_bounds()         — WGS84 bounding box + 20% padding
  ↓ [optional] enrich()     — weather (Open-Meteo), solar (pysolar), features (Overpass)
  ↓ fetch_osm_vectors()     — roads, waterways, water areas, trails from Overpass
  ↓ get_projection()        — auto-select UTM zone from track centroid
  ↓ fetch_dem() / load_dem()— USGS 3DEP GeoTIFF (or local file)
  ↓ generate_hillshade()    — standard cartographic shaded-relief array
  ↓ generate_contours()     — marching-squares + D-P simplification
  ↓ project_points/bounds() — WGS84 → UTM metres
  ↓ [optional] build_wind_field() + trace_streamlines() — RK4 wind paths
  ↓ render_terrain_svg()    — full SVG compositor
  ↓ cairosvg.svg2pdf()      — timestamped PDF in docs/preview/
```

---

## 2. Module Inventory

### `field_atlas/core/`

| Module | Status | Description |
|--------|--------|-------------|
| `gpx_parser.py` | Complete | Parses GPX tracks and routes; haversine distance; 2m-threshold elevation gain/loss; per-segment speed with 5-point rolling average |
| `projection.py` | Complete | WGS84 → UTM via pyproj; auto-selects zone from centroid |
| `dem_fetcher.py` | Complete | USGS 3DEP elevation tile fetch; `load_dem()` returns numpy array + rasterio Affine transform |
| `terrain_processor.py` | Complete | `generate_contours()` (marching squares, `preserve_geology=True`); `generate_hillshade()` (standard formula); `compute_slope()`; `smooth_elevation()` |

### `field_atlas/enrichment/`

| Module | Status | Description |
|--------|--------|-------------|
| `weather.py` | Complete | Open-Meteo Archive API; `WeatherData` dataclass; file I/O for offline use |
| `solar.py` | Complete | pysolar-backed `SolarData`; sunrise/sunset times; `sun_arc_points()` for the diagram inset |
| `features.py` | Complete | Overpass API; `MapFeature` / `FeatureSet`; proximity + type ranking; up to 8 labeled features |
| `osm_vectors.py` | Complete | Overpass full-geometry query; roads, waterways, water areas, trails; JSON cache I/O |
| `wind.py` | Complete | 2D wind field (terrain-deflected, with noise); RK4 streamline tracing; JSON I/O |
| `models.py` | Complete | `EnrichmentData` aggregate; `enrich()` pipeline; `format_info_block()`; `derive_location_name()` |
| `tides.py` | **Stub** | Empty file; tidal enrichment not implemented |

### `field_atlas/render/`

| Module | Status | Description |
|--------|--------|-------------|
| `svg_composer.py` | Complete (~2,400 lines) | Entire rendering engine: contours, hillshade, OSM layers, route, neatline, graticule, north arrow, scale bar, title block, speed legend, environmental insets |
| `color_palettes.py` | Complete | 5 palettes (`coastal`, `ember`, `ocean`, `mono`, `nordic`); HSL-space interpolation |
| `pdf_export.py` | **Stub** | Empty file; PDF export is done inline in `cli/main.py` via `cairosvg` |
| `stl_export.py` | **Stub** | Empty file; 3D mesh export not implemented |

### `field_atlas/cli/`

| Module | Status | Description |
|--------|--------|-------------|
| `main.py` | Complete | Click-based CLI; `render` subcommand wires the full pipeline; auto-orientation; inline PDF export |

---

## 3. Development Timeline

Development ran from February 28 to March 2, 2026 — approximately three days of
concentrated work. The history below summarizes the major development phases as
reconstructed from git commits.

### Phase 0 — Scaffolding (Feb 28, early)

The project was initialized with a complete module structure before any code existed.
The foundational commit sequence:

- `b076480` — `dem_fetcher.py` for USGS 3DEP
- `510ebc4` — `projection.py` for WGS84 → UTM
- `563b698` — `gpx_parser.py` extended to handle route files and directories
- `cb90ec4` — `svg_composer.py` Phase 1 (contour + route SVG renderer)
- `c758d7e` — `terrain_processor.py`
- `576c3df` — `cli/main.py` full GPX→DEM→SVG pipeline (initial version)
- `ca9c2e6` — white background rect added to canvas

### Phase 1 — Enrichment Pipeline (Feb 28, mid)

Weather, solar, and geographic feature enrichment was added as a parallel layer
activated by the `--date` flag:

- `c91ccec` — `weather.py` (Open-Meteo Archive)
- `3dc5b26` — `solar.py` (pysolar)
- `a5d18ed` — enrichment file I/O, features module, CLI fallback
- `a2665f9` — `features.py` redesign: `MapFeature`/`FeatureSet`, bounds-based fetch
- `1f4a338` — `models.py`: `EnrichmentData`, `enrich()`, `format_info_block()`
- `aaf04bd` — `--date` / `--notes` CLI flags wired into pipeline
- `ed74eb1` — Phase 2 cleanup: numpy casting, `derive_location_name()`
- `cf2bd58` — feature label rendering: markers, styles, AABB collision avoidance

### Phase 2 — Terrain Pipeline Fix (Mar 1, early)

A critical bug cluster was found and fixed after the first end-to-end renders showed
incorrect or missing output:

- `97acc16` — **Three bugs fixed** (see §5.1, §5.2, §5.3 below)
- `a7ef285` — **Title block not wired** (see §5.4)

### Phase 3 — Contour Rendering Overhaul (Mar 1, mid)

The contour visual quality went through an extended refinement cycle:

- `faf8bb6` — 3-tier stroke weights, Chaikin smoothing, hillshade opacity modulation
- `926c570` — `add_title_block()` rewritten with full 6-row info block
- `00260f6` — 3× font scale for legibility; 3" dedicated bottom margin
- `fdb8dd2` — route casing, benchmark markers, Catmull-Rom smoothing, hillshade colour fix
- `c0db690` — D-P simplification + index labels; `preserve_geology` parameter
- `91d8098` — D-P tolerance raised to 3m (see §5.7)
- `c316e9d` — D-P tolerance **reverted** to 0.5m
- `7a0c1de` — default DEM resolution raised to 1024px

### Phase 4 — Real Data + OSM (Mar 1, mid-late)

The move from synthetic to real DEMs, and the addition of map context layers:

- `9bd5559` — SRTM data via AWS Terrain Tiles replacing synthetic DEMs
- `59c92ec` — **OSM vector layers**: roads, waterways, water areas, trail network
- `a58eb51` — output format switched from PNG to PDF
- `0b135ea` — contour tone polish, sheet auto-orient, margin update, water area fix
- `5f08a15` — contour labels on major lines; peak markers; lake fill attempt (later paused)

### Phase 5 — Route Rendering (Mar 1, evening)

The route coloring system went through several complete redesigns:

- `a7dddda` — speed-to-color routing with palette system
- `db90c58` — per-segment `linearGradient` strokes
- `584c3a8` — Catmull-Rom Bézier curves for speed-colored route
- `7239217` — continuous gradient, HSL color, coastal palette
- `49e021a` — scipy spline + per-segment along-path gradients (current approach)
- `77df2f6` — unified start/end route markers (white circle, black outline, 3× larger)
- `bc1c0ed` — end marker 30% smaller than start

### Phase 6 — Wind Streamlines (Mar 1, late evening)

- `48d4831` — wind vector field (terrain-deflected) + RK4 streamline tracing + rendering

### Phase 7 — Cartographic Elements (Mar 2)

- `96c55ae` — USGS-style checkered neatline border + coordinate graticule
- `16757e6` — seconds-level labels, hairlines, rotated side text, wider border
- `e79a9fc` — refined neatline, grid, nautical north arrow, title block breathing room
- `acfe243` — **auto-scaling dual-unit cartographic scale bar** (metric + imperial + RF)

---

## 4. Current Product State

### 4.1 Visual Layers (z-order, bottom to top)

1. **White background** — full canvas fill
2. **OSM water areas** — pale blue polygon fills *(fetched but **disabled** — see §5.8)*
3. **OSM waterways** — blue-gray stroked lines; rivers 0.5mm, streams 0.2mm
4. **OSM roads** — gray stroked lines; major 0.6mm, minor 0.35mm
5. **Coordinate graticule** — 33% black hairlines at subdivision interval; 50% opacity
6. **Wind streamlines** — tapered dark-gray paths; bell-curve width; 6–12% opacity *(only with `--date` + weather data)*
7. **Contour lines** — 3-tier hierarchy:
   - Major (every 10th level): `#808080`, 0.20mm
   - Index (every 5th level): `#BFBFBF`, 0.10mm
   - Minor (base interval): `#D9D9D9`, 0.05mm
   - Elevation labels on major contours; white knockout rectangle
8. **OSM trail network** — dashed gray `#BBBBBB`, 0.2mm; above contours, below route
9. **Route line** — black casing + per-segment HSL gradient (speed) or solid blue fallback
10. **Route start/end markers** — white circle, black outline; start r=3.0mm, end r=2.1mm

### 4.2 Marginalia

- **Neatline border** — 5mm wide (two 2.5mm bands) USGS-style checkered frame; 25% opacity bands; subdivision boundaries aligned to graticule ticks
- **Coordinate labels** — lat labels rotated 90° on side margins; lng labels horizontal top/bottom
- **North arrow** — 8-point nautical compass rose; 25.4mm total diameter; serif "N" label
- **Scale bar** — dual-unit (metric + imperial) checkered bar; auto-selected ground distance (35–85mm target width); representative fraction (RF) below
- **Title block** (bottom margin):
  - Row 1: Location name, uppercase, letterspaced (10.59mm / 30pt equivalent)
  - Row 2: Date, long-form human format (8.46mm)
  - Row 3: 120mm hairline rule
  - Row 4: Distance · Gain · Duration · Coordinates (7.41mm)
  - Row 5: Weather + sunrise (6.87mm) — *only with enrichment*
  - Row 6: "FIELD ATLAS" wordmark, right-aligned (5.28mm)
- **Speed legend** — 20×2mm gradient swatch + SLOW/FAST labels — *only with speed data*
- **Sun arc inset** — upper-right margin; gold polyline showing solar path; E/W labels; sunrise–sunset time — *only with enrichment*
- **Wind compass inset** — upper-left margin; compass rose + directional arrow + speed label — *only with enrichment*

### 4.3 Canvas and Format

- **Portrait:** 18×24 in (457.2×609.6mm) — selected when route height ≥ width
- **Landscape:** 24×18 in (609.6×457.2mm) — selected when route width > height
- **Margins:** 38.1mm (1.5 in) sides + top; 76.2mm (3 in) bottom
- **Output:** SVG written to `output/` (gitignored); timestamped PDF archived to `docs/preview/`

### 4.4 CLI Interface

```
python -m field_atlas render <gpx_file>
  --output PATH         (default: output/<slug>.svg)
  --resolution INT      DEM pixels (default: 1024)
  --contour-interval M  vertical interval (default: 20m)
  --padding FRAC        bounding box padding (default: 0.2)
  --dem-file PATH       local GeoTIFF bypass
  --weather-file PATH   pre-fetched weather JSON
  --features-file PATH  pre-fetched features JSON
  --vectors-file PATH   pre-fetched OSM vectors JSON
  --date YYYY-MM-DD     activates enrichment pipeline
  --notes TEXT          user notes stored in enrichment record
  --route-palette NAME  coastal|ember|ocean|mono|nordic (default: coastal)
  --route-width MM      base route line width (default: 1.8)
  --wind-streamlines N  streamline count; 0=disable (default: 60)
  --streamlines-file PATH pre-computed streamlines JSON
```

### 4.5 Sample Data

| GPX file | Distance | Terrain | Notes |
|----------|----------|---------|-------|
| North Cascades High Route | 170.9 km | Alpine, Washington | No timestamps; elevation gain reports 0m |
| Botany Bay on AllTrails | ~5 km | Coastal, BC | Has some elevation data |
| Hidden Lake Peak | ~16 km | Alpine, WA | — |
| Sun Mountain via Sunnyside Trail | ~25 km | Okanogan, WA | — |

Accompanying pre-fetched data files: `dem_north_cascades_high_route.tif`, `vectors_north_cascades_high_route.json` for offline/cached rendering.

---

## 5. Bugs and Errors Encountered

### 5.1 Projection Axis Order (Fixed — commit `97acc16`)

**Severity: Critical (rendered map was wrong)**

The initial implementation of `project_points()` and `project_bounds()` in
`core/projection.py` assigned the `(easting, northing)` pair returned by pyproj
as `(northing, easting)`. Since EPSG:326XX defines axes as (Easting, Northing),
every projected coordinate was transposed, rotating the entire rendered map 90°.
The fix was straightforward once diagnosed — correcting the variable assignment — but
no output from before this commit is geographically correct.

### 5.2 Contour Paths Not Projected to UTM (Fixed — commit `97acc16`)

**Severity: Critical**

`generate_contours()` returns `[lng, lat]` coordinate pairs in the WGS84 CRS of the
source DEM. The CLI was passing these raw geographic coordinates directly to
`render_terrain_svg()`, which expects UTM metres. This caused contour lines to render
in the wrong coordinate space entirely (or to scale to effectively zero size). The
fix added the missing projection loop in `cli/main.py`.

### 5.3 Title Block Not Called (Fixed — commit `a7ef285`)

**Severity: High (entire bottom third of map was blank)**

`add_title_block()` was implemented in `svg_composer.py` but never invoked from
`render_terrain_svg()`. All early renders have a blank bottom margin. This was
diagnosed and fixed by wiring the call into the composition function.

### 5.4 Hillshade Colour Fix (Fixed — commit `fdb8dd2`)

**Severity: Medium**

The hillshade layer was rendering with incorrect colours. This was corrected as part
of the route casing and smoothing commit.

### 5.5 Elevation Gain Reporting 0m for North Cascades (Active)

**Severity: Medium**

The North Cascades High Route GPX is a *route* file (planning geometry) rather than
a *track* file with embedded elevation data. `parse_gpx()` reads elevation from
`pt.elevation`, which is `None` for route waypoints without embedded altitudes. The
elevation gain calculation filters out `None` values and operates on the remainder,
leaving `elevation_gain_m = 0.0`. The printed output reads "Gain: 0m" for a 170 km
alpine traverse — visually wrong for the title block.

A robust fix would fall back to sampling the DEM raster at each route point's
coordinates when GPX elevation data is absent.

### 5.6 Timezone Hardcoded to Eastern Time (Active)

**Severity: Medium**

In `cli/main.py`, the call to `enrich()` always passes
`timezone_str="America/New_York"` regardless of where the route is located. For the
North Cascades (UTC-8), Hidden Lake Peak (UTC-8), and any international route, the
solar enrichment will compute sunrise/sunset times using an incorrect timezone,
potentially off by 3+ hours. The timezone should be inferred from the route
centroid's longitude (or from a timezone lookup library).

### 5.7 Douglas-Peucker Tolerance Oscillation (Resolved by reversion — commits `c0db690`, `91d8098`, `c316e9d`)

**Severity: Low (design debate, not a correctness bug)**

Three consecutive commits changed the D-P simplification tolerance in
`svg_composer.py`:
- `c0db690`: Set to 0.5m, added D-P simplification
- `91d8098`: Raised to 3m — rationale: "22% of points retained at 0.5m still looks soft"
- `c316e9d`: Reverted to 0.5m — revised rationale: "full density IS the geological texture"

The final position (0.5m) is sound for preserving terrain character, but the oscillation
suggests that the visual expectations for contour rendering were not established before
implementation began. The remaining softness at lower-resolution DEMs (10m SRTM) is a
data resolution constraint, not an algorithmic one.

### 5.8 Water Area Fill Disabled (Active — paused by design)

**Severity: Low**

OSM water area polygons (lakes, reservoirs) are fetched and parsed into
`OSMVectors.water_areas` but are deliberately not rendered. A comment in
`svg_composer.py` reads: *"PAUSED: lake fill polygons disabled pending design review."*
This means lakes appear as white voids on the map rather than with their characteristic
pale-blue fill, which reduces map legibility in lake-dense terrain like the Pacific
Northwest.

### 5.9 Large SVG/PDF File Size for Dense Routes (Active)

**Severity: Medium (performance)**

The speed-gradient route rendering creates one `<linearGradient>` element in the SVG
`<defs>` section per resampled route segment. With the default `_RESAMPLE_N = 600`,
each route produces 599 gradient definitions. For the North Cascades High Route
(170 km), this contributes to a 13 MB PDF. The size grows with both route point count
and the resampling density.

A more efficient approach would use a single `<linearGradient>` with many color stops
per visual chunk, or consolidate adjacent segments with similar colors into one path
element.

### 5.10 `pdf_export.py` and `stl_export.py` Are Empty Stubs (Active)

**Severity: Low (structural debt)**

Two module files in `render/` exist only as empty placeholders. The PDF export
functionality is implemented inline in `cli/main.py` (a 4-line `cairosvg.svg2pdf()`
call). If the CLI is the only intended entry point this is acceptable, but it
contradicts the project structure, which suggests these should be importable modules.

### 5.11 `tides.py` Is an Empty Stub (Active)

**Severity: Low**

The tidal enrichment module exists in the enrichment package but contains no code.
Tidal data is listed in the README as a feature. The module was scaffolded but never
implemented.

### 5.12 No Unit Tests (Active)

**Severity: Medium (maintainability risk)**

The `tests/` directory contains only `__init__.py` and `mock_speed_render.py` (a
manual rendering test, not an automated test suite). There are no pytest-compatible
unit tests for `gpx_parser`, `terrain_processor`, `projection`, `dem_fetcher`,
`color_palettes`, or any enrichment module. The lack of test coverage means regressions
in the computation pipeline — coordinate handling, elevation processing, contour
generation — will not be caught automatically.

### 5.13 Requirements Are Unversioned (Active)

**Severity: Low (reproducibility)**

`requirements.txt` lists nine bare package names without version constraints:

```
gpxpy
rasterio
numpy
scipy
svgwrite
pyproj
requests
matplotlib
click
cairosvg
```

No `pyproject.toml` or `setup.cfg` exists. Because numpy, scipy, and rasterio
frequently introduce breaking changes between minor versions, a fresh install may
produce different results or import errors over time.

### 5.14 Cache Directory Relative to CWD (Active)

**Severity: Low (UX)**

`_CACHE_DIR = Path("cache")` in `cli/main.py` is relative to the working directory
at invocation time, not to a predictable location like the project root or
`~/.field_atlas/`. Running the CLI from different directories produces cache files
scattered across the filesystem.

### 5.15 Speed Coloring Silently Falls Back on Length Mismatch (Active)

**Severity: Low**

The check `len(_speeds) == len(route_points) - 1` in `svg_composer.py` is strict. If
any projection or deduplication step changes the effective route point count, speed
coloring silently falls back to solid blue without any warning to the user. This could
explain unexpected plain-blue renders when speed data was expected to be active.

---

## 6. Architecture Observations

### Rendering Monolith

`svg_composer.py` has grown to approximately 2,400 lines and handles every visual
element from contour lines to the scale bar. While functionally cohesive, this makes
it difficult to test individual rendering components in isolation and creates a high
merge risk for any parallel development.

### Inline PDF Generation

PDF export via `cairosvg.svg2pdf()` is a one-line call embedded at the end of the
CLI `render` command. The dedicated `render/pdf_export.py` module is empty. This is
a structural inconsistency but not a functional problem.

### Enrichment Independence

The enrichment pipeline is well-architected for graceful degradation. Each source
(weather, solar, features) is fetched independently with `try/except`; failures
degrade to `None` rather than aborting the render. This is one of the stronger design
decisions in the codebase.

### Local Data Override Pattern

Every API-backed data source (`--dem-file`, `--weather-file`, `--features-file`,
`--vectors-file`, `--streamlines-file`) has a corresponding CLI flag to substitute a
pre-fetched local file. This enables fully offline rendering and deterministic
re-renders, which is important for iteration speed.

---

## 7. Output Summary

As of March 2, 2026, the following renders have been produced and archived:

| Route | Render count | File sizes | Notes |
|-------|-------------|------------|-------|
| North Cascades High Route | 9 renders (3 PNG + 6 PDF) | 13 MB each (latest) | No timestamps; gain = 0m |
| Botany Bay on AllTrails | 17 renders (all PDF) | 23K–376K | Primary development testbed |
| Hidden Lake Peak | 8 renders (4 PNG + 4 PDF) | 277K–647K | — |
| Sun Mountain via Sunnyside Trail | 5 renders (all PDF) | 41K each | — |
| Bellevue (sample) | 9 renders (8 PNG + 1 PDF) | 23K–35K | Small synthetic test |
| Mock speed render | 1 PDF | 23K | Artificial speed data test |

The Botany Bay route has the most render history because it was the primary
iterative testbed throughout visual development — small enough to render quickly
(~1 second vs ~5 seconds for North Cascades) but rich enough in contour variation
to evaluate rendering changes.

---

## 8. Summary of Known Open Issues

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 5.5 | Elevation gain = 0 for routes without embedded elevation | Medium | Open |
| 5.6 | Timezone hardcoded to Eastern, wrong for WA/international routes | Medium | Open |
| 5.8 | Lake fill polygons disabled | Low | Paused by design |
| 5.9 | Large PDF size (13 MB) due to per-segment gradient defs | Medium | Open |
| 5.10 | `pdf_export.py` and `stl_export.py` are empty | Low | Structural debt |
| 5.11 | `tides.py` is empty; tidal feature not implemented | Low | Unstarted |
| 5.12 | No automated unit tests | Medium | Unstarted |
| 5.13 | Requirements unversioned | Low | Open |
| 5.14 | Cache dir relative to CWD | Low | Open |
| 5.15 | Speed coloring silently falls back on point-count mismatch | Low | Open |
