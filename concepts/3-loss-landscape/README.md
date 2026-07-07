# Loss landscape

![landscape magma](gallery/magma_2k.png)

GPT-2's **real** optimization terrain, via the filter-normalized random-directions
method (Li et al. 2018, *Visualizing the Loss Landscape of Neural Nets*):

1. Snapshot the trained weights `θ`.
2. Draw two random directions `d₁, d₂` matching `θ`'s shapes; scale each weight
   tensor's direction to that tensor's own norm (filter normalization — the step that
   makes the surface meaningful rather than arbitrary).
3. Evaluate the real LM loss on a grid of `θ + a·d₁ + b·d₂` (81×81 ≈ 6.5k forward
   passes per seed).

Rendered top-down as a **log-scaled** topographic contour (log reveals the basin's
concentric rings), with a glowing gradient-descent trajectory diving from the
highest-loss corner into the minimum.

**What it represents:** the actual shape of the terrain the model sits in — the dark
basin is the trained minimum; the ridges are where loss rises as you perturb weights.

## Usage

```bash
python generate.py --size wallpaper_4k --seed 4 --palette magma
python generate.py --size banner --seed 4 --palette abyss
```

- `--seed` — a different pair of directions (`0`–`5` precomputed, others compute on
  demand and cache). `--palette` — `magma`, `inferno`, `viridis`, `abyss`, `ember`,
  `twilight`. `--no-descent` — hide the gradient-descent trajectory (terrain only).
  `--size`, `--out`.
- Computing a new seed needs `torch`/`transformers` and a few minutes (MPS/CUDA
  recommended); precomputed grids are shipped as `landscape_s{0..5}_g81.npy`.

`gallery/` holds sample renders.
