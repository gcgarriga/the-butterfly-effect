# Concepts

Each folder is a self-contained generator: `generate.py` (same CLI everywhere),
a `gallery/` of samples, a README explaining the technique, and small model-data
caches. See the [repository README](../README.md) for the full overview, install and
sizing details.

| # | Concept | What it visualizes | Status |
|---|---------|--------------------|--------|
| 1 | [hidden-state-divergence](1-hidden-state-divergence) | two high-temp runs' hidden states diverging after the shared prompt | ✅ |
| 2 | [attention-arcs](2-attention-arcs) | strongest links where two runs attend differently | ✅ |
| 3 | [exploration-sketches](3-exploration-sketches) | early scouting sketches | 🔎 record |
| 4 | [loss-landscape](4-loss-landscape) | GPT-2's real filter-normalized loss surface + descent | ✅ |
| 5 | [light-trails](5-light-trails) | token vectors through 12 layers | ⚰️ dead-end |
| 6 | [attention-flow-field](6-attention-flow-field) | attention as edge-bundled currents | ⚰️ dead-end |
| 7 | [spectral-interference](7-spectral-interference) | wave interference from a weight matrix's singular spectrum | ✅ |
| 8 | [chladni-eigenmodes](8-chladni-eigenmodes) | vibrating-plate nodal patterns from the spectrum | ✅ |
| 9 | [strange-attractor](9-strange-attractor) | Lorenz & friends — the literal butterfly effect | ✅ |
| 10 | [embedding-constellation](10-embedding-constellation) | k-NN graph of token embeddings | ✅ |

Run any concept the same way:

```bash
cd <concept-folder>
python generate.py --size wallpaper_4k          # + per-concept --palette/--seed/etc.
python generate.py --help
```
