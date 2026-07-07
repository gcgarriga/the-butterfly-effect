"""Attention arcs — where two runs attend differently.

Run the prompt through GPT-2 twice, take |A - B| of a layer's attention matrix, keep
only the strongest divergent links (top few percent), and draw them as thin glowing
arcs on black. The attention-sink (key 0) is dropped so the delicate web among later
tokens shows. Left stays empty (shared prompt); a web grows where the runs attended
to different context.

Examples:
    python generate.py --size wallpaper_4k --layer 8 --palette ice
    python generate.py --size banner --layer 5 --palette gold
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import presets, style  # noqa: E402

BG = presets.DEFAULT_BG
PROMPT = "The fundamental nature of chaos and reality is"
N_NEW, TEMP, SEED_A, SEED_B, KEEP_PCT = 30, 1.5, 42, 1337, 97.0

PALETTES = {
    # concept-local "ice" differs from the shared style.PALETTES["ice"] ramp
    "ice": ["#1b3a6b", "#3aa0ff", "#9be7ff", "#ffffff"],
    "gold": style.PALETTES["gold"],
    "violet": style.PALETTES["violet"],
    "emerald": ["#06302a", "#0fbf8f", "#7cf9c8", "#ffffff"],
    "magenta": ["#3a0826", "#ff2d95", "#ff8fce", "#ffffff"],
}


def attention_delta():
    """[n_layers, seq, seq] absolute attention difference between the two runs."""
    cache = os.path.join(os.path.dirname(__file__), "attn_all.npy")
    if os.path.exists(cache):
        return np.load(cache)
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()

    def run(seed):
        torch.manual_seed(seed)
        ids = tok(PROMPT, return_tensors="pt").input_ids
        with torch.no_grad():
            gen = model.generate(
                ids,
                do_sample=True,
                temperature=TEMP,
                min_new_tokens=N_NEW,
                max_new_tokens=N_NEW,
                pad_token_id=tok.eos_token_id,
            )
            out = model(gen, output_attentions=True)
        return np.stack([a[0].mean(0).numpy() for a in out.attentions])

    delta = np.abs(run(SEED_A) - run(SEED_B))
    np.save(cache, delta)
    return delta


def main():
    p = presets.base_parser(__doc__)
    p.add_argument("--layer", type=int, default=5, help="transformer block 0-11")
    p.add_argument("--palette", default="ice", choices=list(PALETTES))
    args = p.parse_args()

    attn = attention_delta()[args.layer]
    seq = attn.shape[0]
    i_all, j_all = np.tril_indices(seq, k=-1)
    keep = j_all >= 1  # drop the attention sink at key 0
    i_all, j_all = i_all[keep], j_all[keep]
    strength = attn[i_all, j_all]
    mask = strength >= np.percentile(strength, KEEP_PCT)
    links = sorted(zip(i_all[mask], j_all[mask], strength[mask]), key=lambda t: t[2])
    smax = strength[mask].max()

    w, h = presets.resolve(args.size)
    lw = style.line_scale(w, h)
    cmap = style.resolve_cmap(PALETTES[args.palette])
    fig, ax = presets.new_fig(w, h, BG, args.dpi)

    theta = np.linspace(0, np.pi, 200)
    for i, j, s in links:
        t = s / smax
        cx, r = (i + j) / 2, (i - j) / 2
        x, y = cx + r * np.cos(theta), r * np.sin(theta)
        col = cmap(0.25 + 0.75 * t)
        for width, a in [
            (18, 0.05 * t),
            (10, 0.10 * t),
            (5.5, 0.25 * t),
            (2.8, 0.45 + 0.4 * t),
        ]:
            ax.plot(
                x,
                y,
                color=col,
                lw=width * lw,
                alpha=min(a, 1.0),
                solid_capstyle="round",
            )
        ax.plot(
            x,
            y,
            color="white",
            lw=1.1 * lw,
            alpha=min(0.5 + 0.5 * t, 1.0),
            solid_capstyle="round",
        )

    max_r = max((i - j) / 2 for i, j, _ in links)
    ax.set_xlim(-1, seq)
    ax.set_ylim(-max_r * 0.06, max_r * 1.12)
    ax.axis("off")

    out = args.out or f"attention_L{args.layer}_{args.palette}_{w}x{h}.png"
    fig.savefig(out, dpi=args.dpi, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
