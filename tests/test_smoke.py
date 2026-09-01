"""Smoke tests: every concept generator runs and emits a correctly-sized PNG.

All concepts use shipped caches or are procedural, so the suite runs offline with no
GPT-2 download — and thereby also checks that the shipped caches are present and valid.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import presets

CONCEPTS = [
    "1-hidden-state-divergence",
    "2-attention-arcs",
    "3-loss-landscape",
    "4-spectral-interference",
    "5-chladni-eigenmodes",
    "6-strange-attractor",
    "7-embedding-constellation",
    "8-forking-paths-tree",
    "9-attention-head-atlas",
    "10-logit-lens-aurora",
]

ENV = {**os.environ, "MPLBACKEND": "Agg", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}


# --- presets unit tests ------------------------------------------------------
def test_resolve_preset():
    assert presets.resolve("square") == (2048, 2048)


def test_resolve_custom():
    assert presets.resolve("1280x720") == (1280, 720)


def test_resolve_invalid():
    with pytest.raises(ValueError):
        presets.resolve("not-a-size")


@pytest.mark.parametrize(
    "bad",
    ["1920x", "x720", "0x0", "-5x100", "10x0", "12.5x40", "1920x1080x3", "potato"],
)
def test_resolve_rejects_malformed(bad):
    with pytest.raises(ValueError):
        presets.resolve(bad)


def test_resolve_tuple_idempotent():
    assert presets.resolve((100, 50)) == (100, 50)
    assert presets.resolve(presets.resolve("1280x720")) == (1280, 720)
    with pytest.raises(ValueError):
        presets.resolve((0, 0))
    with pytest.raises(ValueError):
        presets.resolve((640,))


# --- CLI error handling ------------------------------------------------------
def test_cli_rejects_bad_size():
    """Bad --size must exit 2 via argparse with a clean error, never a traceback."""
    script = ROOT / "concepts" / "4-spectral-interference" / "generate.py"
    result = subprocess.run(
        [sys.executable, str(script), "--size", "0x0"],
        capture_output=True,
        text=True,
        timeout=120,
        env=ENV,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


# --- generator smoke tests ---------------------------------------------------
@pytest.mark.parametrize("concept", CONCEPTS)
def test_generator_runs(concept, tmp_path):
    script = ROOT / "concepts" / concept / "generate.py"
    out = tmp_path / f"{concept}.png"
    result = subprocess.run(
        [sys.executable, str(script), "--size", "240x160", "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=600,
        env=ENV,
        check=False,
    )
    assert result.returncode == 0, f"{concept} failed:\n{result.stderr[-1000:]}"
    assert out.exists(), f"{concept} produced no output"
    img = Image.open(out)
    assert img.size == (240, 160)
    # Guard against a degenerate flat render (e.g. a broken cache painting one color).
    colors = img.convert("RGB").getcolors(maxcolors=1_000_000)
    assert colors is None or len(colors) > 1, f"{concept} rendered a single flat color"
