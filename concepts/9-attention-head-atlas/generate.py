"""Attention-head atlas — a map of GPT-2's 144 attention heads.

One forward pass exposes every head's attention matrix: 12 layers x 12 heads. We lay
them out as a 12x12 contact sheet of small glowing glyphs (rows = layers top-to-bottom,
cols = heads). Insiders can read the repertoire at a glance: diagonal "previous-token"
heads, bright first-column attention-sink heads, striped induction heads, diffuse heads.

The default atlas is cached (`attn_heads.npy`) so it renders offline; passing a different
--text recomputes it from GPT-2 (downloaded on demand). --palette/--gamma/--size never
recompute.

Examples:
    python generate.py --size square --palette magma
    python generate.py --size 3000x3000 --palette ice --gamma 0.5
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import presets  # noqa: E402

BG = "#04060d"
DEFAULT_TEXT = ("The mind is a pattern of patterns, a wave folding back on itself; "
                "every thought echoes another thought, every echo a thought again.")
PALETTES = ["magma", "inferno", "ice", "aurora", "viridis"]
CUSTOM = {
    "ice": ["#02030a", "#0b2545", "#1b6ca8", "#5fd3f3", "#ffffff"],
    "aurora": ["#02110d", "#0fbf8f", "#3aa0ff", "#9b5cff", "#eafff7"],
}


def compute(text):
    """Return attention as [layers, heads, seq, seq]. Needs GPT-2 (eager attention)."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        attns = model(ids, output_attentions=True).attentions
    return np.stack([a[0].numpy() for a in attns])


def load(text):
    cache = os.path.join(os.path.dirname(__file__), "attn_heads.npy")
    if text == DEFAULT_TEXT and os.path.exists(cache):
        return np.load(cache)
    data = compute(text)
    if text == DEFAULT_TEXT:
        np.save(cache, data)
    return data


def main():
    p = presets.base_parser(__doc__)
    p.add_argument("--palette", default="magma", choices=PALETTES)
    p.add_argument("--gamma", type=float, default=0.7, help="contrast; <1 lifts faint attention")
    p.add_argument("--text", default=DEFAULT_TEXT)
    args = p.parse_args()

    from matplotlib.colors import LinearSegmentedColormap as LSC
    cmap = LSC.from_list(args.palette, CUSTOM[args.palette]) if args.palette in CUSTOM else args.palette

    attn = load(args.text)
    n_layer, n_head = attn.shape[:2]
    w, h = presets.resolve(args.size)
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=args.dpi)
    fig.patch.set_facecolor(BG)
    g = 0.006
    for L in range(n_layer):
        for H in range(n_head):
            ax = fig.add_axes([H / n_head + g, (n_layer - 1 - L) / n_layer + g,
                               1 / n_head - 2 * g, 1 / n_layer - 2 * g])
            ax.imshow(attn[L, H] ** args.gamma, cmap=cmap, interpolation="nearest")
            ax.set_facecolor(BG)
            ax.axis("off")
    out = args.out or f"head_atlas_{args.palette}_{w}x{h}.png"
    fig.savefig(out, dpi=args.dpi, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
