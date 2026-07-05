"""Concept sketches for an AI-themed LinkedIn banner.

Renders four candidate directions at banner ratio (1584x396) so we can pick the
most beautiful before committing to a final design:
  1. Embedding galaxy      - PCA of GPT-2 token embeddings (real)
  2. Forking-paths tree     - branching next-token continuations (real)
  3. Loss landscape         - synthetic surface + gradient-descent trajectory
  4. Positional interference- canonical sinusoidal positional encodings
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

torch.manual_seed(0)
np.random.seed(0)
BG = "#04060d"
W, H, DPI = 1584, 396, 150
PROMPT = "The fundamental nature of chaos and reality is"

_tok = _model = None


def model():
    global _tok, _model
    if _model is None:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        _tok = GPT2Tokenizer.from_pretrained("gpt2")
        _model = GPT2LMHeadModel.from_pretrained("gpt2")
        _model.eval()
    return _tok, _model


def fig_ax():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(name, dpi=DPI, facecolor=BG)
    plt.close(fig)
    print("Saved", name)


# 1. Embedding galaxy ----------------------------------------------------------
def embedding_galaxy():
    _, m = model()
    wte = m.transformer.wte.weight.detach().numpy()
    idx = np.random.choice(wte.shape[0], 6000, replace=False)
    X = wte[idx]
    X = X - X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:3].T  # top-3 principal components
    cmap = LinearSegmentedColormap.from_list(
        "g", ["#3aa0ff", "#9b5cff", "#ff5ca8", "#ffd86b"]
    )
    c = (P[:, 2] - P[:, 2].min()) / (np.ptp(P[:, 2]) + 1e-9)

    fig, ax = fig_ax()
    ax.scatter(P[:, 0], P[:, 1], s=22, c=c, cmap=cmap, alpha=0.12, edgecolors="none")
    ax.scatter(P[:, 0], P[:, 1], s=2.5, c=c, cmap=cmap, alpha=0.85, edgecolors="none")
    lo, hi = np.percentile(P[:, 0], [1, 99])
    ax.set_xlim(lo, hi)
    ax.set_ylim(*np.percentile(P[:, 1], [1, 99]))
    save(fig, "concept_galaxy.png")


# 2. Forking-paths tree --------------------------------------------------------
def forking_tree():
    tok, m = model()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    K, THRESH, MAXD, CAP = 3, 0.08, 8, 220
    nodes = [{"depth": 0, "p": 1.0, "parent": None, "kids": []}]
    seqs = {0: ids}

    def expand(nid, depth):
        if depth >= MAXD or len(nodes) >= CAP:
            return
        with torch.no_grad():
            logits = m(seqs[nid]).logits[0, -1]
        probs = torch.softmax(logits, -1)
        top_p, top_i = probs.topk(K)
        for p, t in zip(top_p.tolist(), top_i.tolist()):
            if p < THRESH or len(nodes) >= CAP:
                continue
            cid = len(nodes)
            nodes.append({"depth": depth + 1, "p": p, "parent": nid, "kids": []})
            nodes[nid]["kids"].append(cid)
            seqs[cid] = torch.cat([seqs[nid], torch.tensor([[t]])], 1)
            expand(cid, depth + 1)

    expand(0, 0)

    leaf_y = [0.0]

    def assign(nid):
        kids = nodes[nid]["kids"]
        if not kids:
            y = leaf_y[0]
            leaf_y[0] += 1
        else:
            ys = [assign(k) for k in kids]
            y = sum(ys) / len(ys)
        nodes[nid]["y"] = y
        return y

    assign(0)
    cmap = LinearSegmentedColormap.from_list(
        "t", ["#1b3a6b", "#3aa0ff", "#9be7ff", "#ffffff"]
    )

    fig, ax = fig_ax()
    th = np.linspace(0, 1, 40)
    for n in nodes[1:]:
        par = nodes[n["parent"]]
        x0, y0, x1, y1 = par["depth"], par["y"], n["depth"], n["y"]
        xs = x0 + (x1 - x0) * th
        ys = y0 + (y1 - y0) * (3 * th**2 - 2 * th**3)  # smoothstep curve
        col = cmap(0.2 + 0.8 * n["p"])
        for lw, a in [(7, 0.06), (3, 0.15), (1.3, 0.5 + 0.5 * n["p"])]:
            ax.plot(xs, ys, color=col, lw=lw, alpha=min(a, 1), solid_capstyle="round")
    ax.set_xlim(-0.2, MAXD + 0.2)
    ax.set_ylim(-1, leaf_y[0])
    save(fig, "concept_tree.png")


# 3. Loss landscape (synthetic) ------------------------------------------------
def loss_landscape():
    x = np.linspace(-3, 3, 600)
    y = np.linspace(-1.6, 1.6, 300)
    Xg, Yg = np.meshgrid(x, y)
    Z = (
        1.6 * np.exp(-((Xg + 1.5) ** 2 + (Yg + 0.6) ** 2) / 0.8)
        + 1.2 * np.exp(-((Xg - 1.7) ** 2 + (Yg - 0.5) ** 2) / 0.5)
        - 1.5 * np.exp(-((Xg - 0.2) ** 2 + (Yg + 0.1) ** 2) / 1.2)
        + 0.25 * np.sin(2.2 * Xg) * np.cos(2.6 * Yg)
    )

    def grad(px, py):
        e = 1e-3
        gx = (zval(px + e, py) - zval(px - e, py)) / (2 * e)
        gy = (zval(px, py + e) - zval(px, py - e)) / (2 * e)
        return gx, gy

    def zval(px, py):
        return (
            1.6 * np.exp(-((px + 1.5) ** 2 + (py + 0.6) ** 2) / 0.8)
            + 1.2 * np.exp(-((px - 1.7) ** 2 + (py - 0.5) ** 2) / 0.5)
            - 1.5 * np.exp(-((px - 0.2) ** 2 + (py + 0.1) ** 2) / 1.2)
            + 0.25 * np.sin(2.2 * px) * np.cos(2.6 * py)
        )

    px, py, path = -2.4, 1.2, []
    for _ in range(160):
        path.append((px, py))
        gx, gy = grad(px, py)
        px -= 0.05 * gx
        py -= 0.05 * gy
    path = np.array(path)

    fig, ax = fig_ax()
    ax.contourf(Xg, Yg, Z, levels=40, cmap="magma")
    ax.contour(Xg, Yg, Z, levels=18, colors="white", linewidths=0.25, alpha=0.18)
    for lw, a in [(6, 0.12), (3, 0.3), (1.4, 0.95)]:
        ax.plot(path[:, 0], path[:, 1], color="#9be7ff", lw=lw, alpha=a)
    ax.scatter([path[-1, 0]], [path[-1, 1]], s=60, color="white", zorder=5)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    save(fig, "concept_landscape.png")


# 4. Positional interference ---------------------------------------------------
def positional_interference():
    L, d = 396, 320
    pos = np.arange(L)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d)
    pe = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
    cmap = LinearSegmentedColormap.from_list(
        "p", ["#0b1f3a", "#3a0ca3", "#f72585", "#ffd60a"]
    )

    fig, ax = fig_ax()
    ax.imshow(pe.T, aspect="auto", cmap=cmap, interpolation="bilinear")
    save(fig, "concept_positional.png")


if __name__ == "__main__":
    embedding_galaxy()
    forking_tree()
    loss_landscape()
    positional_interference()
