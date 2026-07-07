"""Forking-paths tree — the butterfly effect of language.

From a prompt, recursively expand the tree of likely next-token continuations: at each
node keep every token within RATIO of the most likely one, so the tree branches where
the model is uncertain and runs straight where it is confident. A single root explodes
into hundreds of possible futures — deterministic model, diverging paths.

Branching responds to temperature: a flatter distribution (higher --temp) puts more
tokens within RATIO of the top, thickening the tree.

The default tree is cached (`tree_default.npz`) so it renders offline; any change to a
generation parameter (--prompt/--temp/--ratio/--k/--max-depth/--cap) recomputes it from
GPT-2 (downloaded on demand). Rendering options (--palette/--size) never recompute.

Examples:
    python generate.py --size wallpaper_4k --palette glow
    python generate.py --prompt "In the beginning" --temp 2.0 --ratio 0.3
"""

import os
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap as LSC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import presets  # noqa: E402

BG = "#04060d"
DEFAULTS = dict(prompt="The", temp=1.6, ratio=0.4, k=4, max_depth=8, cap=600)

PALETTES = {
    "glow": ["#1b3a6b", "#3aa0ff", "#9be7ff", "#ffffff"],
    "ember": ["#2a0606", "#ff5a36", "#ffae00", "#fffaf0"],
    "violet": ["#241046", "#7b2ff7", "#c77dff", "#ffffff"],
    "aurora": ["#02110d", "#0fbf8f", "#3aa0ff", "#9b5cff", "#eafff7"],
    "gold": ["#2a1c02", "#b8860b", "#ffd60a", "#fffbe6"],
}


def compute_tree(prompt, temp, ratio, k, max_depth, cap):
    """Return per-node (parent, depth, y, prob) arrays. Needs GPT-2."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    parent, depth, prob = [-1], [0], [1.0]
    kids = {0: []}
    seqs = {0: tok(prompt, return_tensors="pt").input_ids}

    def expand(nid, d):
        if d >= max_depth or len(parent) >= cap:
            return
        with torch.no_grad():
            logits = model(seqs[nid]).logits[0, -1]
        probs = torch.softmax(logits / temp, -1)
        tp, ti = probs.topk(k)
        top1 = tp[0].item()
        for p, t in zip(tp.tolist(), ti.tolist()):
            if p < ratio * top1 or len(parent) >= cap:
                continue
            cid = len(parent)
            parent.append(nid)
            depth.append(d + 1)
            prob.append(p)
            kids[nid].append(cid)
            kids[cid] = []
            seqs[cid] = torch.cat([seqs[nid], torch.tensor([[t]])], 1)
            expand(cid, d + 1)

    expand(0, 0)
    y = [0.0] * len(parent)
    leaf = [0.0]

    def assign(nid):
        if not kids[nid]:
            yy = leaf[0]
            leaf[0] += 1
        else:
            yy = float(np.mean([assign(c) for c in kids[nid]]))
        y[nid] = yy
        return yy

    assign(0)
    return np.array(parent), np.array(depth), np.array(y), np.array(prob)


def load_tree(args):
    """Use the shipped cache for default generation params; else recompute from GPT-2."""
    gen = dict(
        prompt=args.prompt,
        temp=args.temp,
        ratio=args.ratio,
        k=args.k,
        max_depth=args.max_depth,
        cap=args.cap,
    )
    cache = os.path.join(os.path.dirname(__file__), "tree_default.npz")
    if gen == DEFAULTS and os.path.exists(cache):
        d = np.load(cache)
        return d["parent"], d["depth"], d["y"], d["prob"]
    tree = compute_tree(**gen)
    if gen == DEFAULTS:
        np.savez(cache, parent=tree[0], depth=tree[1], y=tree[2], prob=tree[3])
    return tree


def draw(ax, parent, depth, y, prob, cmap, lw):
    ax.set_facecolor(BG)
    maxd = max(depth.max(), 1)
    th = np.linspace(0, 1, 40)
    smooth = 3 * th**2 - 2 * th**3
    for i in range(1, len(parent)):
        p = parent[i]
        xs = depth[p] + (depth[i] - depth[p]) * th
        ys = y[p] + (y[i] - y[p]) * smooth
        col = cmap(0.15 + 0.85 * depth[i] / maxd)  # brighten outward with depth
        for width, a in [(11, 0.05), (5.0, 0.14), (2.2, 0.9)]:
            ax.plot(xs, ys, color=col, lw=width * lw, alpha=a, solid_capstyle="round")
    ax.set_xlim(-0.3, maxd + 0.3)
    ax.set_ylim(y.min() - 1, y.max() + 1)
    ax.axis("off")


def main():
    p = presets.base_parser(__doc__)
    p.add_argument("--palette", default="glow", choices=list(PALETTES))
    p.add_argument("--prompt", default=DEFAULTS["prompt"])
    p.add_argument(
        "--temp", type=float, default=DEFAULTS["temp"], help="higher = bushier"
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=DEFAULTS["ratio"],
        help="keep tokens >= ratio*top; lower = bushier",
    )
    p.add_argument("--k", type=int, default=DEFAULTS["k"], help="max branches per node")
    p.add_argument("--max-depth", type=int, default=DEFAULTS["max_depth"])
    p.add_argument(
        "--cap", type=int, default=DEFAULTS["cap"], help="max nodes (the frontier size)"
    )
    args = p.parse_args()

    w, h = presets.resolve(args.size)
    lw = (w**2 + h**2) ** 0.5 / 1632
    cmap = LSC.from_list(args.palette, PALETTES[args.palette])
    fig, ax = presets.new_fig(w, h, BG, args.dpi)
    draw(ax, *load_tree(args), cmap, lw)
    out = args.out or f"forking_{args.palette}_{w}x{h}.png"
    fig.savefig(out, dpi=args.dpi, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
