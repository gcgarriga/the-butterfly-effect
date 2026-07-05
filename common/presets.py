"""Shared output-size presets and figure helpers.

Every concept in this repo renders resolution-independent (continuous fields or
vector line art), so any preset or custom WIDTHxHEIGHT works without distortion.
Field-based concepts use `aspect()` to widen their sampling domain to match.
"""

import argparse

import matplotlib.pyplot as plt

# name -> (width, height) in pixels
SIZES = {
    "wallpaper_4k": (3840, 2160),  # 16:9 desktop / slides / video call background
    "wallpaper_2k": (2560, 1440),  # 16:9 smaller desktop
    "ultrawide": (3440, 1440),  # 21:9 monitors, website hero
    "square": (2048, 2048),  # 1:1 social / album / OG image
    "banner": (1584, 396),  # 4:1 profile header (e.g. LinkedIn)
    "wide": (1920, 480),  # 4:1 slimmer header
    "phone": (1290, 2796),  # portrait phone wallpaper
    "print_a4": (3508, 2480),  # A4 @ 300 dpi, landscape poster
}
DEFAULT_SIZE = "wallpaper_4k"
DEFAULT_BG = "#04060d"


def resolve(size: str) -> tuple[int, int]:
    """Accept a preset name or a 'WIDTHxHEIGHT' string -> (width, height) in px."""
    if size in SIZES:
        return SIZES[size]
    if "x" in size.lower():
        w, h = size.lower().split("x")
        return int(w), int(h)
    raise ValueError(f"Unknown size {size!r}; use a preset {list(SIZES)} or WxH, e.g. 2560x1440")


def aspect(w: int, h: int) -> float:
    return w / h


def new_fig(w: int, h: int, bg: str = DEFAULT_BG, dpi: int = 100):
    """A full-bleed figure of exactly (w, h) pixels with a single borderless axes."""
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)
    ax.axis("off")
    return fig, ax


def base_parser(description: str) -> argparse.ArgumentParser:
    """Common CLI: --size, --out, --dpi. Concepts add --seed/--palette/etc."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--size", default=DEFAULT_SIZE,
                   help=f"preset {list(SIZES)} or WIDTHxHEIGHT (default: {DEFAULT_SIZE})")
    p.add_argument("--out", default=None, help="output PNG path (default: auto-named)")
    p.add_argument("--dpi", type=int, default=100, help="raster dpi (default: 100)")
    return p
