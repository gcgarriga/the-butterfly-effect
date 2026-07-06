# Concepts

Each folder is a self-contained generator: `generate.py` (same CLI everywhere),
a `gallery/` of samples, a README explaining the technique, and small model-data
caches. See the [repository README](../README.md) for the full overview, install and
sizing details.

| # | Concept | What it visualizes |
|---|---------|--------------------|
| 1 | [hidden-state-divergence](1-hidden-state-divergence) | two high-temp runs' hidden states diverging after the shared prompt |
| 2 | [attention-arcs](2-attention-arcs) | strongest links where two runs attend differently |
| 3 | [loss-landscape](3-loss-landscape) | GPT-2's real filter-normalized loss surface + descent path |
| 4 | [spectral-interference](4-spectral-interference) | wave interference from a weight matrix's singular spectrum |
| 5 | [chladni-eigenmodes](5-chladni-eigenmodes) | vibrating-plate nodal patterns from the spectrum |
| 6 | [strange-attractor](6-strange-attractor) | Lorenz & friends — the literal butterfly effect (procedural) |
| 7 | [embedding-constellation](7-embedding-constellation) | k-NN graph of token embeddings — the "shape of language" |

Run any concept the same way:

```bash
cd <concept-folder>
python generate.py --size wallpaper_4k          # + per-concept --palette/--seed/etc.
python generate.py --help
```
