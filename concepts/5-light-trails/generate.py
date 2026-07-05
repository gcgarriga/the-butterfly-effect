"""Light-trails (an explored dead-end).

Trace each token's vector as it moves through GPT-2's 12 layers, project to 2D, and
draw a glowing tapering ribbon per token — the model "thinking" as silk threads.
Vectors are unit-normalized before PCA to tame GPT-2's massive-activation outliers
(which otherwise collapse every trail into one streak — why this was abandoned).

Example:
    python generate.py --size wallpaper_4k
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import presets  # noqa: E402

BG = "#04060d"
TEXT = "Intelligence emerges from simple rules repeated at scale, layer upon layer, until meaning appears"


def hidden_states():
    cache = os.path.join(os.path.dirname(__file__), "hidden_states.npy")
    if os.path.exists(cache):
        return np.load(cache)
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    with torch.no_grad():
        out = model(tok(TEXT, return_tensors="pt").input_ids, output_hidden_states=True)
    hs = np.stack([h[0].numpy() for h in out.hidden_states])  # [13, seq, 768]
    np.save(cache, hs)
    return hs


def smooth(x, y, factor=16, w=9):
    t, tf = np.linspace(0, 1, len(x)), np.linspace(0, 1, len(x) * factor)
    xf, yf = np.interp(tf, t, x), np.interp(tf, t, y)
    k = np.ones(w) / w
    xf = np.convolve(np.pad(xf, w, "edge"), k, "same")[w:-w]
    yf = np.convolve(np.pad(yf, w, "edge"), k, "same")[w:-w]
    return xf, yf


def main():
    p = presets.base_parser(__doc__)
    args = p.parse_args()
    hs = hidden_states()
    L, seq, _ = hs.shape
    hn = hs / (np.linalg.norm(hs, axis=2, keepdims=True) + 1e-9)
    flat = hn.reshape(L * seq, -1)
    flat = flat - flat.mean(0)
    _, _, Vt = np.linalg.svd(flat, full_matrices=False)
    proj = (flat @ Vt[:2].T).reshape(L, seq, 2)

    w, h = presets.resolve(args.size)
    lw = (w**2 + h**2) ** 0.5 / 1632
    cmap = plt.get_cmap("turbo")
    fig, ax = presets.new_fig(w, h, BG, args.dpi)
    for t in range(seq):
        xf, yf = smooth(proj[:, t, 0], proj[:, t, 1])
        pts = np.array([xf, yf]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        ramp = np.linspace(0.08, 1.0, len(segs))
        rgb = cmap((t + 0.5) / seq)[:3]
        for width, a in [(7, 0.05), (3.5, 0.13), (1.6, 0.6), (0.7, 1.0)]:
            col = np.zeros((len(segs), 4))
            col[:, :3] = rgb
            col[:, 3] = np.clip(ramp * a, 0, 1)
            ax.add_collection(LineCollection(segs, colors=col, linewidths=width * lw))
    ax.autoscale()
    ax.axis("off")
    out = args.out or f"light_trails_{w}x{h}.png"
    fig.savefig(out, dpi=args.dpi, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
