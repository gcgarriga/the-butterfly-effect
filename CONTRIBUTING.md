# Contributing

Thanks for your interest in **The Butterfly Effect** — generative backgrounds from
the real internals of GPT-2. Contributions are welcome: new concepts, better palettes,
bug fixes, and docs.

## Project layout

```
common/presets.py        shared sizes + figure helpers (base_parser, resolve, new_fig)
concepts/<n>-<name>/
    generate.py          one self-contained CLI per concept
    gallery/             a few sample renders
    README.md            the technique + what it represents
    *.npy / *.npz        small precomputed model-data caches (shipped)
tests/test_smoke.py      offline smoke suite (runs every generator tiny)
.github/workflows/ci.yml lint + offline smoke on every push/PR
```

Each concept is standalone: `generate.py` builds its CLI from
`common.presets.base_parser` (giving `--size` / `--out` / `--dpi`) and adds its own
flags. Model-derived data is cached to `.npy` / `.npz` and only recomputed on a cache
miss or non-default parameters, so everything renders offline from the shipped caches.

## Dev setup (offline, no torch)

The smoke suite and every shipped cache render **without** downloading GPT-2, so you can
develop and test with just the lightweight stack:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt      # pytest + ruff
pip install numpy matplotlib pillow      # the runtime deps the generators use
```

If you need the **full model path** (recomputing a cache from GPT-2 with a new `--seed`,
`--prompt`, etc.), install the complete stack instead:

```bash
pip install -r requirements.txt          # adds torch + transformers (~500 MB download on first run)
```

## Running checks locally

```bash
ruff check .              # lint
ruff format .            # auto-format (use --check in CI)
MPLBACKEND=Agg pytest    # offline smoke suite (no model download)
```

CI runs `ruff check .`, `ruff format --check .`, and the offline smoke suite on every
push and pull request. Please make sure all three are green before opening a PR.

## Adding a new concept

1. Create `concepts/N-name/` (next number, short kebab-case name).
2. Add `generate.py` that builds its parser from `common.presets.base_parser(...)` and
   writes a correctly-sized PNG honoring `--size` / `--out` / `--dpi`.
3. If it derives from GPT-2, cache the model-derived array to a **small** `.npy` / `.npz`
   and load from that cache by default, recomputing only on a cache miss or non-default
   generation parameters — so it renders offline.
4. Ship that small cache plus a couple of `gallery/` sample renders.
5. Write a `README.md` explaining the technique and what it represents.
6. Add the folder name to the `CONCEPTS` list in `tests/test_smoke.py` so it's covered
   by the smoke suite.
7. Run the checks above and confirm your generator passes at `--size 240x160`.

## Commit style

Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`,
`refactor:`, `docs:`, `test:`, `chore:`. Keep each PR to one logical change.

## Reporting issues

Use the bug report or new-concept issue forms under **New issue**. Include the exact
command you ran (concept + flags), your OS and Python version, and expected vs actual.
