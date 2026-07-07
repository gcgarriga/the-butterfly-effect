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
from common import presets  # noqa: E402

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
    )
    assert result.returncode == 0, f"{concept} failed:\n{result.stderr[-1000:]}"
    assert out.exists(), f"{concept} produced no output"
    assert Image.open(out).size == (240, 160)
