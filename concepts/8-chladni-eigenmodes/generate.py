"""Chladni eigenmodes — vibrating-plate nodal patterns.

A plate's standing-wave field is a sum of modes sin(nπx/Lx)·sin(mπy/Ly); sand
collects where the field vanishes (the nodal lines). Here the mode amplitudes are
the real singular spectrum of a GPT-2 weight matrix, so the network's learned
frequencies choose which modes ring loudest. We render the glowing nodal band.

Examples:
    python generate.py --size wallpaper_4k --palette ice
    python generate.py --size banner --palette ultraviolet --seed 3
"""

import os
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap as LSC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import presets  # noqa: E402

BG = "#04060d"

PALETTES = {
    "ice": ["#06122a", "#0b3a6b", "#1b8fd0", "#5fd3f3", "#ffffff"],
    "aurora": ["#02110d", "#0fbf8f", "#3aa0ff", "#9b5cff", "#eafff7"],
    "magma": ["#06010a", "#3a0ca3", "#b5179e", "#ff8800", "#ffe08a"],
    "ultraviolet": ["#05030f", "#3a0ca3", "#7b2ff7", "#ff2d95", "#ffd6f2"],
    "ember": ["#0a0402", "#7a1f0a", "#ff5a36", "#ffae00", "#fff0d0"],
    "gold": ["#0a0802", "#2a1c02", "#b8860b", "#ffd60a", "#fffbe6"],
}


def spectrum(k):
    path = os.path.join(os.path.dirname(__file__), "sval128.npy")
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


def field(seed, w, h, k=44):
    """Plate aspect follows the output; mode ceilings scale so cells stay square-ish."""
    a = presets.aspect(w, h)
    lx, ly = a, 1.0
    nmax, mmax = int(round(3 * a)) + 2, 6
    amp = spectrum(k)
    rng = np.random.default_rng(seed)
    amp = (amp / amp[0]) ** 0.5 * rng.choice([-1, 1], k)
    ns = rng.integers(1, nmax + 1, k)
    ms = rng.integers(1, mmax + 1, k)
    gy = int(np.clip(h * 0.5, 360, 1100))
    gx = int(gy * a)
    X, Y = np.meshgrid(np.linspace(0, lx, gx), np.linspace(0, ly, gy))
    Z = np.zeros_like(X)
    for amp_k, n, m in zip(amp, ns, ms):
        Z += amp_k * np.sin(n * np.pi * X / lx) * np.sin(m * np.pi * Y / ly)
    return Z


def draw(ax, Z, cmap):
    sigma = np.abs(Z).max() * 0.045
    inten = np.exp(-(Z**2) / (2 * sigma**2)) + 0.4 * np.exp(-(Z**2) / (2 * (3 * sigma) ** 2))
    inten /= inten.max()
    ax.imshow(inten, origin="lower", aspect="auto", cmap=cmap, interpolation="bilinear")
    ax.axis("off")


def main():
    p = presets.base_parser(__doc__)
    p.add_argument("--palette", default="ice", choices=list(PALETTES))
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args()

    w, h = presets.resolve(args.size)
    fig, ax = presets.new_fig(w, h, BG, args.dpi)
    draw(ax, field(args.seed, w, h), LSC.from_list(args.palette, PALETTES[args.palette]))
    out = args.out or f"chladni_{args.palette}_s{args.seed}_{w}x{h}.png"
    fig.savefig(out, dpi=args.dpi, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
