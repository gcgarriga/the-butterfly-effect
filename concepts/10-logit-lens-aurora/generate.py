"""Logit-lens aurora — how GPT-2 makes up its mind, layer by layer.

The logit lens applies the model's output head to *every* layer's hidden state, not
just the last, revealing the next-token distribution the model is "considering" at each
depth. We take that distribution's sorted top probabilities per layer and stack them as
glowing ridgelines: early layers are flat and uncertain, deeper layers spike as the
prediction sharpens — the model committing to an answer as depth increases.

The default is cached (`logit_lens.npy`) so it renders offline; a different --prompt
recomputes from GPT-2 (downloaded on demand). --palette/--top/--size never recompute.

Examples:
    python generate.py --size wallpaper_4k --palette glow
    python generate.py --prompt "Once upon a time" --top 40
"""

import os
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap as LSC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import presets  # noqa: E402

BG = "#04060d"
DEFAULT_PROMPT = "The future of artificial intelligence will"
DEFAULT_TOP = 30
PALETTES = {
    "glow": ["#1b3a6b", "#3aa0ff", "#9be7ff", "#ffffff"],
    "ember": ["#2a0606", "#ff5a36", "#ffae00", "#fffaf0"],
    "aurora": ["#02110d", "#0fbf8f", "#3aa0ff", "#9b5cff", "#eafff7"],
    "violet": ["#241046", "#7b2ff7", "#c77dff", "#ffffff"],
}


def compute(prompt, top):
    """Return [n_layers+1, top] sorted top probabilities per layer. Needs GPT-2."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
        ln_f, head = model.transformer.ln_f, model.lm_head
        rows = []
        for h in hs:
            probs = torch.softmax(head(ln_f(h[0, -1])), -1)
            rows.append(torch.sort(probs, descending=True).values[:top].numpy())
    return np.stack(rows)


def load(prompt, top):
    cache = os.path.join(os.path.dirname(__file__), "logit_lens.npy")
    if prompt == DEFAULT_PROMPT and top == DEFAULT_TOP and os.path.exists(cache):
        return np.load(cache)
    data = compute(prompt, top)
    if prompt == DEFAULT_PROMPT and top == DEFAULT_TOP:
        np.save(cache, data)
    return data


def draw(ax, rows, cmap, lw):
    ax.set_facecolor(BG)
    n, top = rows.shape
    x = np.arange(top)
    step = 0.7
    for L, row in enumerate(rows):
        y = L * step + np.sqrt(row) * 3.2  # sqrt lifts the tail; spikes = confidence
        col = cmap(L / n)
        ax.fill_between(x, L * step, y, color=col, alpha=0.12)
        for width, a in [(10, 0.06), (5, 0.18), (2.6, 0.95)]:
            ax.plot(x, y, color=col, lw=width * lw, alpha=a, solid_capstyle="round")
    ax.set_xlim(-1, top)
    ax.set_ylim(-0.4, n * step + 3.4)
    ax.axis("off")


def main():
    p = presets.base_parser(__doc__)
    p.add_argument("--palette", default="glow", choices=list(PALETTES))
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument(
        "--top", type=int, default=DEFAULT_TOP, help="tokens per layer ridgeline"
    )
    args = p.parse_args()

    w, h = presets.resolve(args.size)
    lw = (w**2 + h**2) ** 0.5 / 1632
    cmap = LSC.from_list(args.palette, PALETTES[args.palette])
    fig, ax = presets.new_fig(w, h, BG, args.dpi)
    draw(ax, load(args.prompt, args.top), cmap, lw)
    out = args.out or f"logit_lens_{args.palette}_{w}x{h}.png"
    fig.savefig(out, dpi=args.dpi, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
