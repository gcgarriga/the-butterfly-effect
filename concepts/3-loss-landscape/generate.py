"""Loss landscape — GPT-2's real optimization terrain.

Using the filter-normalized random-directions method (Li et al. 2018): snapshot the
trained weights, pick two random directions in weight space (each scaled per-tensor
to the weights' norm), and evaluate the real language-model loss on a grid of
perturbations. Rendered top-down as a log-scaled topographic contour with a glowing
gradient-descent trajectory diving into the minimum.

Examples:
    python generate.py --size wallpaper_4k --seed 4 --palette magma
    python generate.py --size banner --seed 4 --palette abyss
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import presets, style  # noqa: E402

BG = presets.DEFAULT_BG
GRID, RANGE = 81, 1.0
EVAL_TEXT = (
    "The history of artificial intelligence is a story of ambition and breakthroughs."
)

PALETTES = {
    "magma": "magma",
    "inferno": "inferno",
    "viridis": "viridis",
    "abyss": ["#05030f", "#1b1450", "#7209b7", "#f72585", "#ff8800", "#ffe08a"],
    "ember": ["#06010a", "#3a0606", "#c81d25", "#ff7b00", "#ffe08a", "#fffaf0"],
    "twilight": "twilight_shifted",
}


def compute_terrain(seed):
    cache = os.path.join(os.path.dirname(__file__), f"landscape_s{seed}_g{GRID}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    ids = tok(EVAL_TEXT, return_tensors="pt").input_ids.to(dev)
    params = list(model.parameters())
    orig = [p.detach().clone() for p in params]

    def direction():
        dirs = []
        for p in orig:
            d = torch.randn_like(p)
            if p.dim() >= 2:
                df, pf = d.flatten(1), p.flatten(1)
                d = (df * (pf.norm(dim=1) / (df.norm(dim=1) + 1e-10))[:, None]).view_as(
                    p
                )
            else:
                d = torch.zeros_like(p)
            dirs.append(d)
        return dirs

    torch.manual_seed(seed)
    d1, d2 = direction(), direction()
    axis = np.linspace(-RANGE, RANGE, GRID)
    Z = np.empty((GRID, GRID), np.float32)
    with torch.no_grad():
        for iy, b in enumerate(axis):
            for ix, a in enumerate(axis):
                for p, o, e1, e2 in zip(params, orig, d1, d2):
                    p.copy_(o + a * e1 + b * e2)
                Z[iy, ix] = model(ids, labels=ids).loss.item()
        for p, o in zip(params, orig):
            p.copy_(o)
    np.save(cache, Z)
    return Z


def _smooth(path, w=9):
    k = np.ones(w) / w
    out = path.copy()
    for c in (0, 1):
        out[:, c] = np.convolve(np.pad(path[:, c], w, "edge"), k, "same")[w:-w]
    return out


def descent_path(Z, steps=320):
    gy, gx = np.gradient(Z)
    corners = [(0, 0), (0, GRID - 1), (GRID - 1, 0), (GRID - 1, GRID - 1)]
    sy, sx = max(corners, key=lambda c: Z[c])
    y, x, pts = float(sy), float(sx), []
    for k in range(steps):
        pts.append((x, y))
        i, j = np.clip(int(round(y)), 0, GRID - 1), np.clip(int(round(x)), 0, GRID - 1)
        dx, dy = gx[i, j], gy[i, j]
        n = np.hypot(dx, dy) + 1e-9
        step = 1.3 * (1 - k / steps) + 0.04
        x = np.clip(x - step * dx / n, 0, GRID - 1)
        y = np.clip(y - step * dy / n, 0, GRID - 1)
    my, mx = np.unravel_index(np.argmin(Z), Z.shape)
    return _smooth(np.array(pts + [(mx, my)]))


def draw(ax, Z, cmap, s, descent=True):
    ax.set_facecolor(BG)
    Zc = np.log(Z)  # loss spans ~3..150; log reveals the basin's rings
    ax.contourf(Zc, levels=44, cmap=cmap)
    ax.contour(Zc, levels=22, colors="white", linewidths=0.22 * s, alpha=0.16)
    if descent:  # the glowing gradient-descent trajectory diving into the minimum
        path = descent_path(Z)
        for lw, a in [(9, 0.10), (5, 0.20), (2.4, 0.45), (1.1, 0.97)]:
            ax.plot(
                path[:, 0],
                path[:, 1],
                color="#9be7ff",
                lw=lw * s,
                alpha=a,
                solid_capstyle="round",
            )
        ex, ey = path[-1]
        for sz, a in [(520, 0.06), (240, 0.12), (110, 0.30)]:
            ax.scatter(
                [ex],
                [ey],
                s=sz * s * s,
                color="#9be7ff",
                alpha=a,
                edgecolors="none",
                zorder=6,
            )
        ax.scatter([ex], [ey], s=46 * s * s, color="white", edgecolors="none", zorder=7)
    ax.set_xlim(0, GRID - 1)
    ax.set_ylim(0, GRID - 1)
    ax.axis("off")


def main():
    p = presets.base_parser(__doc__)
    p.add_argument("--palette", default="magma", choices=list(PALETTES))
    p.add_argument("--seed", type=int, default=4)
    p.add_argument(
        "--no-descent",
        action="store_true",
        help="hide the gradient-descent trajectory (terrain only)",
    )
    args = p.parse_args()

    w, h = presets.resolve(args.size)
    cmap = style.resolve_cmap(PALETTES[args.palette])
    fig, ax = presets.new_fig(w, h, BG, args.dpi)
    draw(
        ax,
        compute_terrain(args.seed),
        cmap,
        style.line_scale(w, h),
        descent=not args.no_descent,
    )
    out = args.out or f"landscape_{args.palette}_s{args.seed}_{w}x{h}.png"
    fig.savefig(out, dpi=args.dpi, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
