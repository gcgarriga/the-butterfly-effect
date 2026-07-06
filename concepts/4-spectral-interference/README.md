# Spectral interference

![steel studio banner](gallery/steel_studio_banner.png)

A luminous moiré of contour lines. The image is a superposition of plane waves

```
Z(x, y) = Σ_k  a_k · sin( f_k (cosθ_k · x + sinθ_k · y) + φ_k )
```

whose **amplitudes `a_k` are the real singular values** of a GPT-2 weight matrix
(`transformer.h[6].mlp.c_fc`, via SVD). The directions `θ_k`, phases `φ_k` and
frequencies `f_k` are seeded pseudo-randomly, so a `--seed` selects a "terrain"
while the wave energy distribution stays grounded in the model's learned spectrum.
Rendering the iso-contours of that field yields the topographic/fingerprint pattern.

**What it represents:** the frequency content of what the network learned, turned
into interference — structure from a real model, rendered as abstract art.

## Usage

```bash
python generate.py                                        # 4K, ice, connected
python generate.py --size banner --palette steel --studio # profile header
python generate.py --size 2560x1440 --palette pastel_sky --seed 5
python generate.py --size square --style lines --palette iridescent
```

- `--style connected` — filled + glowing lines (soft, flowing). `--style lines` —
  fine moiré line-work only.
- `--palette` — cool/mono (`ice`, `steel`, `graphite`, `ice_slate`), pastels
  (`pastel_dawn`, `cotton_candy`, `pastel_mint`, `pastel_sky`, `peach_lilac`),
  rich (`jade`, `copper`, `sunset`, `magma`, `iridescent`).
- `--studio` — soft spotlight + vignette + contrast/saturation for a lit-backdrop finish.
- `--seed`, `--size` (preset or `WxH`), `--out`. Tuning: `field(k, fmax, exp)` in the
  source — lower `k`/`fmax` = larger, more connected swirls.

Spectrum cached in `sval128.npy` (recomputed from GPT-2 if missing). `gallery/` holds
sample renders.
