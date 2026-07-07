"""Shared visual theme — canonical palettes, colormap resolution, line scaling.

The ten concept generators share one curated look, but historically each generator
re-declared its palettes, its background colour and the size->line-weight magic
number. That copy-paste let the SAME palette name drift to DIFFERENT colours
across concepts (e.g. "ice" resolved to five different ramps). This module holds
the shared pieces so the identity is defined once.

`PALETTES` holds only the multi-stop ramps that were verified byte-identical
across every concept that uses them; genuinely concept-specific ramps stay local
to their generator (so this refactor changes no rendered output). `resolve_cmap`
collapses the several palette-resolution idioms into one call, and `line_scale`
replaces the diagonal/1632 factor that was pasted into most generators.

Matplotlib is imported lazily inside `resolve_cmap` to match the repo's
offline-friendly, lazy-import style (importing this module stays cheap).
"""

# name -> hex colour stops, low..high. Each name below was duplicated verbatim
# across the listed concepts before this refactor; now it has one authority.
PALETTES = {
    "aurora": ["#02110d", "#0fbf8f", "#3aa0ff", "#9b5cff", "#eafff7"],  # 5, 8, 9, 10
    "ice": ["#06122a", "#0b3a6b", "#1b8fd0", "#5fd3f3", "#ffffff"],  # 4, 5
    "glow": ["#1b3a6b", "#3aa0ff", "#9be7ff", "#ffffff"],  # 8, 10
    "violet": ["#241046", "#7b2ff7", "#c77dff", "#ffffff"],  # 2, 8, 10
    "gold": ["#2a1c02", "#b8860b", "#ffd60a", "#fffbe6"],  # 2, 8
    "ember": ["#2a0606", "#ff5a36", "#ffae00", "#fffaf0"],  # 8, 10
}


def resolve_cmap(spec):
    """Return a matplotlib Colormap for `spec`.

    `spec` may be any of the idioms the generators used:
      * a Colormap instance            -> returned unchanged
      * a registered PALETTES name     -> LinearSegmentedColormap from its stops
      * a matplotlib builtin name      -> plt.get_cmap(name)
      * a list/tuple of hex stops      -> LinearSegmentedColormap from those stops

    Custom ramps are built with matplotlib's default 256-sample resolution, so
    the result is identical to the previous per-generator `LSC.from_list(...)`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Colormap, LinearSegmentedColormap

    if isinstance(spec, Colormap):
        return spec
    if isinstance(spec, (list, tuple)):
        return LinearSegmentedColormap.from_list("custom", list(spec))
    if spec in PALETTES:
        return LinearSegmentedColormap.from_list(spec, PALETTES[spec])
    return plt.get_cmap(spec)


def line_scale(w: int, h: int) -> float:
    """Size->line-weight factor that keeps strokes proportional across outputs.

    Every line-art generator multiplied its base line widths by this so a 4K
    render and a slim banner keep the same visual stroke weight. 1632 is the
    shared reference diagonal the widths were tuned against.
    """
    return (w**2 + h**2) ** 0.5 / 1632
