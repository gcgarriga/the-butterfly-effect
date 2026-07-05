"""Smoke tests: every concept generator runs and emits a correctly-sized PNG.

The FAST set uses shipped caches (or is procedural), so it needs no GPT-2 download —
runs in CI without torch. The one model-dependent concept (3) is marked `model` and
skipped unless RUN_MODEL_TESTS=1.
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

# Concepts runnable without downloading GPT-2 (cache-backed or purely procedural).
FAST_CONCEPTS = [
    "1-hidden-state-divergence",
    "2-attention-arcs",
    "4-loss-landscape",
    "5-light-trails",
    "6-attention-flow-field",
    "7-spectral-interference",
    "8-chladni-eigenmodes",
    "9-strange-attractor",
    "10-embedding-constellation",
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
@pytest.mark.parametrize("concept", FAST_CONCEPTS)
def test_generator_runs(concept, tmp_path):
    script = ROOT / "concepts" / concept / "generate.py"
    out = tmp_path / f"{concept}.png"
    result = subprocess.run(
        [sys.executable, str(script), "--size", "240x160", "--out", str(out)],
        capture_output=True, text=True, timeout=600, env=ENV,
    )
    assert result.returncode == 0, f"{concept} failed:\n{result.stderr[-1000:]}"
    assert out.exists(), f"{concept} produced no output"
    assert Image.open(out).size == (240, 160)


@pytest.mark.model
@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1",
                    reason="downloads GPT-2; set RUN_MODEL_TESTS=1 to run")
def test_exploration_sketches(tmp_path):
    script = ROOT / "concepts" / "3-exploration-sketches" / "generate.py"
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path,
        capture_output=True, text=True, timeout=1800, env=ENV,
    )
    assert result.returncode == 0, result.stderr[-1000:]
    assert len(list(tmp_path.glob("concept_*.png"))) == 4
