"""
color_palettes.py — Route color palette definitions and speed interpolation.

Palettes map hiking speed (slow → fast) to a three-stop color gradient.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class RoutePalette:
    """Three-stop color gradient for encoding movement speed along a route."""

    slow_color: str   # hex color for the slowest segments
    mid_color: str    # hex color for median-pace segments
    fast_color: str   # hex color for the fastest segments
    name: str         # human-readable palette name


# ---------------------------------------------------------------------------
# Preset palettes
# ---------------------------------------------------------------------------

PALETTES: dict[str, RoutePalette] = {
    "maroon_turquoise": RoutePalette("#6B2039", "#8C7A6B", "#2B8C8C", "Maroon to Turquoise"),
    "ember":            RoutePalette("#8B2500", "#CC7A29", "#E8B84D", "Ember"),
    "ocean":            RoutePalette("#1A3A4A", "#3A7CA5", "#5BC4B0", "Ocean"),
    "mono":             RoutePalette("#222222", "#666666", "#AAAAAA", "Monochrome"),
    "nordic":           RoutePalette("#2E3440", "#5E81AC", "#88C0D0", "Nordic"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def get_palette(name: str) -> RoutePalette:
    """Return a :class:`RoutePalette` by name.

    Raises
    ------
    KeyError
        If *name* is not in :data:`PALETTES`.
    """
    if name not in PALETTES:
        raise KeyError(f"Unknown palette {name!r}. Available: {list(PALETTES)}")
    return PALETTES[name]


def interpolate_color(speed_normalized: float, palette: RoutePalette) -> str:
    """Return a hex color for a 0.0–1.0 normalised speed value.

    Interpolates linearly in RGB through three stops:

    * ``0.0`` → ``palette.slow_color``
    * ``0.5`` → ``palette.mid_color``
    * ``1.0`` → ``palette.fast_color``

    Parameters
    ----------
    speed_normalized:
        Clamped to [0.0, 1.0].
    palette:
        A :class:`RoutePalette` instance.

    Returns
    -------
    str
        Upper-case hex color string, e.g. ``"#2B8C8C"``.
    """
    t = max(0.0, min(1.0, speed_normalized))
    if t <= 0.5:
        u = t * 2.0
        c0 = _hex_to_rgb(palette.slow_color)
        c1 = _hex_to_rgb(palette.mid_color)
    else:
        u = (t - 0.5) * 2.0
        c0 = _hex_to_rgb(palette.mid_color)
        c1 = _hex_to_rgb(palette.fast_color)

    r = int(round(c0[0] + u * (c1[0] - c0[0])))
    g = int(round(c0[1] + u * (c1[1] - c0[1])))
    b = int(round(c0[2] + u * (c1[2] - c0[2])))
    return f"#{r:02X}{g:02X}{b:02X}"
