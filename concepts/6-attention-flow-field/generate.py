"""Attention flow field (an explored dead-end).

Reimagine attention not as discrete arcs but as bundled currents: place token nodes on
a circle and route each strong link through the centre (edge bundling) into glowing
rivers. The attention sink made everything funnel into one thin vortex — too sparse
and monochrome, which is why this was abandoned.

Example:
    python generate.py --size wallpaper_4k --layer 6
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import presets  # noqa: E402

BG = "#04060d"
TEXT = "Intelligence emerges from simple rules repeated at scale, layer upon layer, until meaning appears"


def attention(layer):
    cache = os.path.join(os.path.dirname(__file__), "attention.npy")
    if os.path.exists(cache):
        return np.load(cache)[layer]
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    with torch.no_grad():
        out = model(tok(TEXT, return_tensors="pt").input_ids, output_attentions=True)
    attn = np.stack([a[0].mean(0).numpy() for a in out.attentions])  # [12, seq, seq]
    np.save(cache, attn)
    return attn[layer]


def main():
    p = presets.base_parser(__doc__)
    p.add_argument("--layer", type=int, default=6)
    args = p.parse_args()
    attn = attention(args.layer)
    seq = attn.shape[0]
    ang = np.linspace(0, 2 * np.pi, seq, endpoint=False)
    node = np.stack([np.cos(ang), np.sin(ang)], 1)
    links = [(i, j, attn[i, j]) for i in range(seq) for j in range(1, i)]
    thr = np.percentile([w for *_, w in links], 70)
    links = sorted([lk for lk in links if lk[2] >= thr], key=lambda t: t[2])
    smax = max(w for *_, w in links)

    w, h = presets.resolve(args.size)
    lw = (w**2 + h**2) ** 0.5 / 1632
    cmap = plt.get_cmap("hsv")
    fig, ax = presets.new_fig(w, h, BG, args.dpi)
    tb = np.linspace(0, 1, 80)[:, None]
    for i, j, wt in links:
        p0, p3 = node[j], node[i]
        p1, p2 = p0 * 0.15, p3 * 0.15  # pull toward centre -> bundle
        curve = (
            (1 - tb) ** 3 * p0
            + 3 * (1 - tb) ** 2 * tb * p1
            + 3 * (1 - tb) * tb**2 * p2
            + tb**3 * p3
        )
        col = cmap(ang[j] / (2 * np.pi))
        for width, a in [(7, 0.06), (3, 0.15), (1.3, 0.5 + 0.5 * wt / smax)]:
            ax.plot(
                curve[:, 0], curve[:, 1], color=col, lw=width * lw, alpha=min(a, 1.0)
            )
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")
    out = args.out or f"flow_field_L{args.layer}_{w}x{h}.png"
    fig.savefig(out, dpi=args.dpi, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
