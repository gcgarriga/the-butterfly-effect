"""Embedding constellation — the shape of language.

A k-nearest-neighbour graph over real GPT-2 token embeddings (cosine similarity),
force-directed into 2D (Fruchterman-Reingold) and drawn as a glowing node-edge
network. Semantically similar tokens pull together into luminous clusters; a
high-degree hub reads as a bright convergence burst.

Examples:
    python generate.py --size wallpaper_4k --palette turbo
    python generate.py --size ultrawide --palette ice --seed 3
"""

import os
import sys

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap as LSC
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import presets  # noqa: E402

BG = "#04060d"
N, KNN, ITERS = 520, 4, 220

PALETTES = {
    "turbo": plt.get_cmap("turbo"),
    "ice": LSC.from_list("ice", ["#0b3a6b", "#1b8fd0", "#5fd3f3", "#ffffff"]),
    "aurora": LSC.from_list("aurora", ["#0fbf8f", "#3aa0ff", "#9b5cff", "#ff5ca8"]),
    "ember": LSC.from_list("ember", ["#7a1f0a", "#ff5a36", "#ffae00", "#fff0d0"]),
}


def build(seed):
    cache = os.path.join(os.path.dirname(__file__), f"layout_s{seed}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["pos"], d["edges"], d["hue"]
    import torch
    from transformers import GPT2LMHeadModel

    m = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    with torch.no_grad():
        wte = m.transformer.wte.weight.numpy()
    rng = np.random.default_rng(seed)
    E = wte[rng.choice(5000, N, replace=False)]
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sim = E @ E.T
    np.fill_diagonal(sim, -1)
    nbr = np.argsort(-sim, axis=1)[:, :KNN]
    edges = np.array(sorted({(min(i, j), max(i, j)) for i in range(N) for j in nbr[i]}))
    A = np.zeros((N, N))
    A[edges[:, 0], edges[:, 1]] = 1
    A += A.T

    Ec = E - E.mean(0)
    _, _, Vt = np.linalg.svd(Ec, full_matrices=False)
    pos = Ec @ Vt[:2].T
    pos /= np.abs(pos).max()
    k, temp = np.sqrt(1.0 / N) * 2.0, 0.12
    for _ in range(ITERS):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((diff**2).sum(-1)) + 1e-6
        rep = (k**2 / dist**2)[..., None] * diff
        attr = (A * dist / k)[..., None] * (-diff) / dist[..., None]
        disp = rep.sum(1) + attr.sum(1)
        dl = np.sqrt((disp**2).sum(-1)) + 1e-9
        pos += disp / dl[:, None] * np.minimum(dl, temp)[:, None]
        temp *= 0.985
    hue = np.argsort(np.argsort(pos[:, 0])) / (N - 1)  # even spectrum spread along x
    np.savez(cache, pos=pos, edges=edges, hue=hue)
    return pos, edges, hue


def main():
    p = presets.base_parser(__doc__)
    p.add_argument("--palette", default="turbo", choices=list(PALETTES))
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()

    w, h = presets.resolve(args.size)
    pos, edges, hue = build(args.seed)
    cmap = PALETTES[args.palette]
    s = (w**2 + h**2) ** 0.5 / 1632  # scale glyph sizes with output

    fig, ax = presets.new_fig(w, h, BG, args.dpi)
    ecol = cmap(hue[edges].mean(1))
    for lw, a in [(2.4, 0.05), (1.0, 0.16), (0.4, 0.5)]:
        c = ecol.copy()
        c[:, 3] = a
        ax.add_collection(LineCollection(pos[edges], colors=c, linewidths=lw * s))
    ncol = cmap(hue)
    for sz, a in [(90, 0.10), (34, 0.25), (10, 0.9)]:
        ax.scatter(pos[:, 0], pos[:, 1], s=sz * s * s, c=ncol, alpha=a, edgecolors="none")
    ax.scatter(pos[:, 0], pos[:, 1], s=2.5 * s * s, c="white", alpha=0.9, edgecolors="none")
    ax.set_xlim(*np.percentile(pos[:, 0], [1, 99]))
    ax.set_ylim(*np.percentile(pos[:, 1], [1, 99]))
    ax.axis("off")

    out = args.out or f"constellation_{args.palette}_s{args.seed}_{w}x{h}.png"
    fig.savefig(out, dpi=args.dpi, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
