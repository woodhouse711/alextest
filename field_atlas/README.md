# Field Atlas

Field Atlas is a tool that converts GPX hiking data into terrain-based cartographic artifacts.

Given a GPX track, Field Atlas fetches elevation data, processes terrain geometry, and produces
richly detailed outputs including SVG maps, printable PDFs, and 3D-printable STL meshes.
Enrichment modules add weather, solar angle, tidal, and geographic feature data to produce
maps that capture not just where you went, but what the land looked and felt like.

## Features

- Parse GPX tracks and waypoints
- Fetch and process Digital Elevation Model (DEM) data
- Project and align geographic coordinates
- Enrich routes with weather, solar, tidal, and feature data
- Export to SVG, PDF, and STL formats

## Usage

```bash
field-atlas --help
```

## Project Structure

- `field_atlas/core/` — GPX parsing, DEM fetching, terrain processing, projection
- `field_atlas/enrichment/` — Weather, solar, tidal, and geographic feature enrichment
- `field_atlas/render/` — SVG composition, PDF export, STL export
- `field_atlas/cli/` — Command-line interface
