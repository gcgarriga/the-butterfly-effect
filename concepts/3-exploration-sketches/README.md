# Exploration sketches

![sketches](gallery/concept_landscape.png)

A single scouting pass that renders four quick concept sketches side by side, used to
choose a direction early on. Kept as a record of the exploration.

- `concept_galaxy.png` — PCA of GPT-2 token embeddings (a semantic starfield); matured
  into concept **10** (embedding constellation).
- `concept_tree.png` — a branching tree of next-token continuations under sampling.
- `concept_landscape.png` — a synthetic loss-surface contour; matured into concept
  **4** (real loss landscape).
- `concept_positional.png` — sinusoidal positional-encoding interference; the wave
  idea matured into concept **7** (spectral interference).

**What it represents:** the fork in the road — several AI-visualization ideas tried
cheaply before committing.

## Usage

```bash
python generate.py     # renders all four sketches (fresh; no cache)
```

Unlike the other concepts this script has no CLI — it is a fixed exploratory artifact.
The matured directions live in concepts 4, 7 and 10. `gallery/` holds the outputs.
