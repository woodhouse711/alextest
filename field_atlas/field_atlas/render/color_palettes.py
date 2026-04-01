"""
color_palettes.py — Route color palette definitions and speed interpolation.

Palettes map hiking speed (slow → fast) to a three-stop color gradient.
Interpolation is performed in HSL space to preserve saturation across the
white midpoint without producing muddy RGB pastels.
"""

from __future__ import annotations

import colorsys
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
    "coastal":          RoutePalette("#0B678E", "#FFFFFF", "#46BAB4", "Coastal"),
    "ember":            RoutePalette("#8B2500", "#FFFFFF", "#E8B84D", "Ember"),
    "ocean":            RoutePalette("#1A3A4A", "#3A7CA5", "#5BC4B0", "Ocean"),
    "mono":             RoutePalette("#222222", "#888888", "#DDDDDD", "Monochrome"),
    "nordic":           RoutePalette("#2E3440", "#5E81AC", "#88C0D0", "Nordic"),
}


# ---------------------------------------------------------------------------
# Color conversion helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex_to_hsl(hex_color: str) -> tuple[float, float, float]:
    """Return (h, s, l) each in [0.0, 1.0] for a hex color string."""
    r, g, b = _hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    return h, s, l


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    """Convert HSL (each in [0.0, 1.0]) back to an uppercase hex string."""
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    ri = max(0, min(255, int(round(r * 255))))
    gi = max(0, min(255, int(round(g * 255))))
    bi = max(0, min(255, int(round(b * 255))))
    return f"#{ri:02X}{gi:02X}{bi:02X}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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

    Interpolates in HSL space through three stops:

    * ``0.0`` → ``palette.slow_color``
    * ``0.5`` → ``palette.mid_color``
    * ``1.0`` → ``palette.fast_color``

    When the midpoint is achromatic (white or gray), the donor hue from the
    slow or fast end is preserved so saturation fades cleanly without muddy
    RGB mixing.

    Parameters
    ----------
    speed_normalized:
        Clamped to [0.0, 1.0].
    palette:
        A :class:`RoutePalette` instance.

    Returns
    -------
    str
        Upper-case hex color string, e.g. ``"#46BAB4"``.
    """
    t = max(0.0, min(1.0, speed_normalized))

    if t <= 0.5:
        u = t * 2.0
        h0, s0, l0 = _hex_to_hsl(palette.slow_color)
        h1, s1, l1 = _hex_to_hsl(palette.mid_color)
        if s1 < 0.02:
            # Mid is achromatic (white/gray): hold slow hue, interpolate S and L
            h = h0
            s = s0 * (1.0 - u)
            l = l0 + (l1 - l0) * u
        else:
            h = h0 + (h1 - h0) * u
            s = s0 + (s1 - s0) * u
            l = l0 + (l1 - l0) * u
    else:
        u = (t - 0.5) * 2.0
        h0, s0, l0 = _hex_to_hsl(palette.mid_color)
        h1, s1, l1 = _hex_to_hsl(palette.fast_color)
        if s0 < 0.02:
            # Mid is achromatic: hold fast hue, interpolate S and L
            h = h1
            s = s1 * u
            l = l0 + (l1 - l0) * u
        else:
            h = h0 + (h1 - h0) * u
            s = s0 + (s1 - s0) * u
            l = l0 + (l1 - l0) * u

    return _hsl_to_hex(h, s, l)
