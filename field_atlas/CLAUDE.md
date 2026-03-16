# Field Atlas — Claude Instructions

## Preview render after every update

After every Claude-initiated code change (bug fix, new feature, refactor), run a
preview render using the Hidden Lake Peak sample route and commit the output.
This keeps `docs/preview/` as a visual changelog that makes regressions obvious.

### Command

```bash
cd /home/user/alextest/field_atlas
python -m field_atlas render sample_data/hidden-lake-peak.gpx \
    --dem-file sample_data/dem_hidden_lake_peak.tif
```

The CLI automatically writes a timestamped PNG and PDF to
`docs/preview/YYYY-MM-DD/` (today's date folder, created if needed).

### After rendering, commit the preview

```bash
git add docs/preview/
git commit -m "preview: <short description of what changed>"
```

### Folder structure

```
docs/preview/
├── YYYY-MM-DD/          ← one folder per calendar day
│   ├── <slug>_HHMMSS.png
│   └── <slug>_HHMMSS.pdf
└── archive/             ← older renders before the dated-folder convention
```

## Sample data & caches

| File | Purpose |
|---|---|
| `sample_data/hidden-lake-peak.gpx` | Primary render test (alpine, complete elevation) |
| `sample_data/Botany_Bay.gpx` | Fast coastal test (small bbox, quick fetch) |
| `sample_data/dem_hidden_lake_peak.tif` | Pre-fetched USGS DEM — use with `--dem-file` to avoid network |
| `sample_data/vectors_north_cascades_high_route.json` | Pre-fetched OSM vectors (North Cascades route) |

## Key rendering files

| File | What it does |
|---|---|
| `field_atlas/core/terrain_processor.py` | Contour generation, hillshade, hypsometric tint |
| `field_atlas/render/svg_composer.py` | SVG composition — layer order, blend modes, styles |
| `field_atlas/cli/main.py` | Pipeline orchestration, DEM fetch, preview export |
| `field_atlas/render/color_palettes.py` | Route colour gradients |

## Branch

All development goes on `claude/add-realistic-map-depth-igX62`.
Push with: `git push -u origin claude/add-realistic-map-depth-igX62`
