"""Spectral interference — a luminous moire whose wave amplitudes are the real
singular spectrum of a GPT-2 weight matrix.

Renders at any size/aspect. Two styles, many palettes, optional studio finish.

Examples:
    python generate.py                                   # 4K, ice, connected
    python generate.py --size banner --palette steel --studio
    python generate.py --size 2560x1440 --style lines --palette pastel_sky --seed 5
"""

import os
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap as LSC
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import presets  # noqa: E402

BG = "#04060d"

PALETTES = {
    # cool / mono / serious
    "ice": ["#06122a", "#0b3a6b", "#1b8fd0", "#5fd3f3", "#ffffff"],
    "steel": ["#05070d", "#17293c", "#3f6180", "#7fa8c8", "#cfe6f7", "#ffffff"],
    "graphite": ["#08090c", "#20262e", "#4a5561", "#8a99a6", "#dfe8ef", "#ffffff"],
    "ice_slate": ["#070d16", "#16324f", "#2f6f9e", "#7fb8e0", "#eaf5ff"],
    # pastels
    "pastel_dawn": ["#241a33", "#6d5a8c", "#c9a0dc", "#ffc6d9", "#ffe8d6", "#fffaf0"],
    "cotton_candy": ["#151030", "#5a4a9c", "#9b8cff", "#ffb3de", "#c8f0ff", "#ffffff"],
    "pastel_mint": ["#0e2626", "#2f6f6a", "#7fd1c1", "#bff3e0", "#eafff8"],
    "pastel_sky": ["#101a33", "#3b5a8c", "#8fb8e8", "#c9e0ff", "#f2f8ff"],
    "peach_lilac": ["#1c1430", "#6a5390", "#b99fe0", "#ffc9b0", "#ffe9d6"],
    # rich
    "jade": ["#03130f", "#0a3a2a", "#1f8f6a", "#6fd8a8", "#dfffe8"],
    "copper": ["#160a06", "#5a2a12", "#c96a2a", "#ffab5e", "#ffe4c0"],
    "sunset": ["#1a0a2a", "#7a2a6a", "#ff5a8a", "#ffb066", "#ffe9a8"],
    "magma": ["#06010a", "#3a0ca3", "#b5179e", "#ff8800", "#ffe08a"],
    "iridescent": ["#0b1f3a", "#3aa0ff", "#9b5cff", "#ff5ca8", "#ffd86b"],
}


def spectrum(k):
    here = os.path.dirname(__file__)
    path = os.path.join(here, "sval128.npy")
    if os.path.exists(path):
        return np.load(path)[:k]
    import torch
    from transformers import GPT2LMHeadModel

    m = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    with torch.no_grad():
        w = m.transformer.h[6].mlp.c_fc.weight.numpy()
    sval = np.linalg.svd(w, compute_uv=False)[:128]
    np.save(path, sval)
    return sval[:k]


def field(seed, w, h, k=18, fmax=2.8, exp=0.55):
    """Sum of plane waves; sampling domain widens with aspect so features stay round."""
    amp = spectrum(k)
    amp = (amp / amp[0]) ** exp
    rng = np.random.default_rng(seed)
    dirs = rng.uniform(0, np.pi, k)
    phase = rng.uniform(0, 2 * np.pi, k)
    freqs = np.linspace(0.3, fmax, k)
    a = presets.aspect(w, h)
    gy = int(np.clip(h * 0.5, 360, 1100))
    gx = int(gy * a)
    X, Y = np.meshgrid(np.linspace(-2 * a, 2 * a, gx), np.linspace(-2, 2, gy))
    Z = np.zeros_like(X)
    for amp_k, f, th, ph in zip(amp, freqs, dirs, phase):
        Z += amp_k * np.sin(f * (np.cos(th) * X + np.sin(th) * Y) + ph)
    return X, Y, Z


def _vignette(ax, X, Y, cap=0.5, start=0.72):
    nx, ny = 400, 120
    gx, gy = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    r = np.sqrt(gx**2 + (gy * 0.55) ** 2)
    rgba = np.zeros((ny, nx, 4))
    rgba[..., 3] = np.clip((r - start) * 2.0, 0, cap)
    ax.imshow(
        rgba, extent=(X.min(), X.max(), Y.min(), Y.max()), aspect="auto", zorder=5
    )


def draw(ax, X, Y, Z, cmap, style, lw):
    ax.set_facecolor(BG)
    if style == "connected":
        ax.contourf(X, Y, Z, levels=30, cmap=cmap, alpha=0.40, zorder=0)
        strokes = [(6.0, 0.06), (3.0, 0.16), (1.4, 0.95)]
        levels = 30
    else:  # lines: finer moire, no fill
        strokes = [(5.0, 0.05), (2.4, 0.12), (0.7, 1.0)]
        levels = 50
    for width, a in strokes:
        ax.contour(
            X, Y, Z, levels=levels, cmap=cmap, linewidths=width * lw, alpha=a, zorder=1
        )
    _vignette(ax, X, Y)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.axis("off")


def studio_finish(rgb):
    """Soft spotlight + elliptical vignette + gentle contrast/saturation lift."""
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rx = (xx / (w - 1) - 0.5) * 2.0
    ry = (yy / (h - 1) - 0.5) * 2.0
    r = np.sqrt((rx * 0.85) ** 2 + ry**2)
    out = rgb * (1 - np.clip((r - 0.55) / 0.75, 0, 1) * 0.55)[..., None]
    spot = np.exp(-(rx**2 / (2 * 0.70**2) + (ry + 0.15) ** 2 / (2 * 0.55**2)))
    out = 1 - (1 - out) * (1 - spot[..., None] * np.array([0.80, 0.88, 1.0]) * 0.14)
    out = np.clip((out - 0.5) * 1.09 + 0.5, 0, 1)
    luma = out @ np.array([0.2126, 0.7152, 0.0722])
    return np.clip(luma[..., None] + (out - luma[..., None]) * 1.14, 0, 1)


def main():
    p = presets.base_parser(__doc__)
    p.add_argument("--palette", default="ice", choices=list(PALETTES))
    p.add_argument("--style", default="connected", choices=["connected", "lines"])
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--studio", action="store_true", help="apply studio lighting finish")
    args = p.parse_args()

    w, h = presets.resolve(args.size)
    cmap = LSC.from_list(args.palette, PALETTES[args.palette])
    lw = (w**2 + h**2) ** 0.5 / 1632  # keep line weight proportional across sizes

    fig, ax = presets.new_fig(w, h, BG, args.dpi)
    draw(ax, *field(args.seed, w, h), cmap, args.style, lw)

    out = args.out or f"spectral_{args.palette}_{args.style}_s{args.seed}_{w}x{h}.png"
    if args.studio:
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255
        Image.fromarray((studio_finish(buf) * 255).astype(np.uint8)).save(out)
    else:
        fig.savefig(out, dpi=args.dpi, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
