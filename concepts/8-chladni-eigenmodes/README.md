# Chladni eigenmodes

![chladni ice](gallery/ice_2k.png)

Standing-wave **nodal patterns** of a vibrating plate. A plate's displacement is a
sum of modes `sin(nπx/Lx)·sin(mπy/Ly)`; scatter sand on it and the sand gathers where
the field is zero — the nodal lines (Chladni figures). Here the **mode amplitudes are
the real singular spectrum of a GPT-2 weight matrix**, so the network's learned
frequencies decide which plate modes ring loudest.

We render the *nodal band* (a glowing Gaussian around `Z = 0`), giving luminous
electric veins on black.

**What it represents:** the same learned spectrum as concept 7, but excited as plate
resonances instead of travelling waves — structured, netted, organic.

## Usage

```bash
python generate.py --size wallpaper_4k --palette ice
python generate.py --size banner --palette ultraviolet --seed 3
python generate.py --size 2048x2048 --palette gold --seed 1
```

- `--palette` — `ice`, `aurora`, `magma`, `ultraviolet`, `ember`, `gold`.
- `--seed` — selects the mode mix; `--size` (preset or `WxH`); `--out`.
- Vein thickness = the `sigma` factor in `draw()`; mode density scales with `--seed`
  via `k` and the `nmax/mmax` ceilings in `field()`.

Spectrum cached in `sval128.npy`. `gallery/` holds sample renders.
