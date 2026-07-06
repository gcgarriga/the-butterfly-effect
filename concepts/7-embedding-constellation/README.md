# Embedding constellation

![constellation turbo](gallery/turbo_2k.png)

The **shape of language**. We take real GPT-2 token embeddings, build a k-nearest-
neighbour graph by cosine similarity, and lay it out in 2D with a Fruchterman-Reingold
force simulation (springs pull neighbours together, all nodes repel). Rendered as a
glowing node-edge network. Semantically related tokens settle into clusters; a token
that is many others' nearest neighbour becomes a high-degree hub — the bright
convergence burst.

**What it represents:** the geometry of the model's learned vocabulary — which words
the network considers close, made visible as a constellation.

## Usage

```bash
python generate.py --size wallpaper_4k --palette turbo
python generate.py --size ultrawide --palette ice --seed 3
python generate.py --size 2048x2048 --palette aurora --seed 7
```

- `--palette` — `turbo` (full spectrum), `ice`, `aurora`, `ember`.
- `--seed` — samples a different set of ~520 tokens and layout; `--size`, `--out`.
- Coloring is rank-based along x for an even spectrum spread. Tuning: `N` (nodes),
  `KNN` (neighbours), `ITERS`, repulsion scale `k` in `build()` (larger = less hub
  clumping).

Layout cached per seed as `layout_s{seed}.npz` (recomputed from GPT-2 if missing).
`gallery/` holds sample renders.
