"""Hidden-state divergence — the original butterfly effect.

Feed the same prompt through GPT-2 twice at high temperature (different seeds), take
the absolute difference of the final-layer hidden states, and plot it. Because GPT-2
is causal, the shared-prompt columns are identical (delta = 0) and divergence erupts
at the first sampled token.

Two styles:
  raw    - the plain delta heatmap (neurons x tokens)
  banner - rows sorted by magnitude + blurred into flowing bands

Examples:
    python generate.py --style banner --size wallpaper_4k --palette turbo
    python generate.py --style raw --size banner --palette magma
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap as LSC, PowerNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import presets  # noqa: E402

BG = "#000000"
PROMPT = "The fundamental nature of chaos and reality is"
N_NEW, TEMP, SEED_A, SEED_B = 30, 1.5, 42, 1337

PALETTES = {
    "magma": "magma", "inferno": "inferno", "turbo": "turbo",
    "viridis": "viridis", "plasma": "plasma", "twilight": "twilight_shifted",
    "butterfly": LSC.from_list("butterfly",
        ["#08001f", "#3a0ca3", "#7209b7", "#b5179e", "#f72585", "#ff8800", "#ffd60a"]),
}


def compute_delta():
    cache = os.path.join(os.path.dirname(__file__), "delta.npy")
    if os.path.exists(cache):
        return np.load(cache)
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

    def run(seed):
        torch.manual_seed(seed)
        ids = tok(PROMPT, return_tensors="pt").input_ids
        with torch.no_grad():
            gen = model.generate(ids, do_sample=True, temperature=TEMP,
                                 min_new_tokens=N_NEW, max_new_tokens=N_NEW,
                                 pad_token_id=tok.eos_token_id)
            out = model(gen, output_hidden_states=True)
        return out.hidden_states[-1].squeeze(0)

    delta = torch.abs(run(SEED_A) - run(SEED_B)).cpu().numpy()
    np.save(cache, delta)
    return delta


def blur(a, passes=2):
    for _ in range(passes):
        for axis in (0, 1):
            a = 0.5 * a + 0.25 * np.roll(a, 1, axis) + 0.25 * np.roll(a, -1, axis)
    return a


def main():
    p = presets.base_parser(__doc__)
    p.add_argument("--style", default="banner", choices=["raw", "banner"])
    p.add_argument("--palette", default="turbo", choices=list(PALETTES))
    args = p.parse_args()

    delta = compute_delta()
    cmap = PALETTES[args.palette]
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    w, h = presets.resolve(args.size)
    fig, ax = presets.new_fig(w, h, BG, args.dpi)

    if args.style == "raw":
        ax.imshow(delta.T, aspect="auto", cmap=cmap, interpolation="nearest")
    else:
        img = blur(delta.T[np.argsort(delta.mean(0))])  # sort neurons by mean divergence
        norm = PowerNorm(0.55, vmin=0, vmax=float(np.percentile(delta, 98)))
        ax.imshow(img, aspect="auto", cmap=cmap, norm=norm, interpolation="bilinear")
    ax.axis("off")

    out = args.out or f"divergence_{args.style}_{args.palette}_{w}x{h}.png"
    fig.savefig(out, dpi=args.dpi, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
