"""Tests for the game-agnostic cell-grid renderer (secondplayer.policy.grid)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from secondplayer.errors import ConfigurationError, ModelError
from secondplayer.policy import grid
from secondplayer.policy.grid import GridSpec

BENCH0 = (
    Path(__file__).parent.parent
    / "secondplayer"
    / "benchmarks"
    / "tetris"
    / "bench0-reference.png"
)
# Validated display geometry for the 256x224 benchmark renderer (probes).
ROI = "93,44,8,8.2,10,20"


def paint(frame: np.ndarray, cols, rows, color=(30, 144, 255)) -> np.ndarray:
    out = frame.copy()
    for c in cols:
        for r in rows:
            out[r * 8 : (r + 1) * 8, c * 8 : (c + 1) * 8] = color
    return out


def test_parse_spec_valid():
    spec = grid.parse_spec(ROI)
    assert spec == GridSpec(93.0, 44.0, 8.0, 8.2, 10, 20)


def test_parse_spec_blank_is_adaptive():
    assert grid.parse_spec("") is None
    assert grid.parse_spec("   ") is None


@pytest.mark.parametrize(
    "bad",
    ["93,44,8,8.2,10", "a,b,c,d,e,f", "93,44,0,8,10,20", "93,44,8,-1,10,20",
     "93,44,8,8.2,0,20", "93,44,8,8.2,10.5,20"],
)
def test_parse_spec_malformed(bad: str):
    with pytest.raises(ConfigurationError):
        grid.parse_spec(bad)


def test_adaptive_spec_covers_whole_frame():
    spec = grid.adaptive_spec(256, 224)
    assert (spec.ox, spec.oy) == (0.0, 0.0)
    assert (spec.pw, spec.ph) == (8.0, 8.0)
    assert (spec.cols, spec.rows) == (32, 28)
    tiny = grid.adaptive_spec(32, 32)
    assert (tiny.cols, tiny.rows) == (32, 32)


def test_classify_synthetic_cells():
    frame = np.full((40, 40, 3), (61, 55, 0), dtype=np.uint8)
    frame = paint(frame, [1, 3], [0, 4])
    spec = GridSpec(0.0, 0.0, 8.0, 8.0, 5, 5)
    mask, bg = grid.classify(frame, spec)
    assert bg == (61, 55, 0)
    filled = {(r, c) for r in range(5) for c in range(5) if mask[r][c]}
    assert filled == {(0, 1), (0, 3), (4, 1), (4, 3)}


def test_render_filled_and_changed_exact_text():
    spec = GridSpec(0.0, 0.0, 8.0, 8.0, 5, 5)
    bg = np.full((40, 40, 3), (61, 55, 0), dtype=np.uint8)
    prev = paint(bg, [2], [4])
    cur = paint(paint(bg, [2], [4]), [2], [3])
    assert grid.render(prev, cur, spec) == (
        "Current frame filled cells (row,col), row 0 top, col 0 is left: (3,2) (4,2)\n"
        "Cells that changed between the previous frame and this frame: (3,2)"
    )


def test_render_first_screen_and_static():
    spec = GridSpec(0.0, 0.0, 8.0, 8.0, 5, 5)
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    first = grid.render(None, frame, spec)
    assert first.splitlines()[0].endswith(": none")
    assert first.splitlines()[1].endswith("first screen, nothing to compare yet")
    static = grid.render(frame.copy(), frame, spec)
    assert static.splitlines()[1].endswith("none - everything is static")


def test_bench0_reference_golden_cells():
    frame = np.asarray(Image.open(BENCH0).convert("RGB"))
    assert frame.shape == (224, 256, 3)
    mask, bg = grid.classify(frame, grid.parse_spec(ROI))
    assert bg == (61, 55, 0)  # probe-identical sampler: exact match required
    filled = {(r, c) for r in range(20) for c in range(10) if mask[r][c]}
    # Probe-validated: falling blob only, well otherwise empty.
    assert filled == {(6, 4), (6, 5), (6, 6), (7, 4), (7, 5)}


def test_render_caps_long_lists():
    spec = GridSpec(0.0, 0.0, 2.0, 2.0, 20, 20)
    frame = np.full((40, 40, 3), (61, 55, 0), dtype=np.uint8)
    # Background stays the plurality (150) so 250 split-color cells read filled.
    for i in range(250):
        color = (200, 30, 30) if i < 125 else (30, 30, 200)
        r, c = divmod(150 + i, 20)
        frame[r * 2 : r * 2 + 2, c * 2 : c * 2 + 2] = color
    out = grid.render(None, frame, spec)
    assert "(+50 more)" in out.splitlines()[0]


def test_classify_out_of_bounds_fails_loud():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    with pytest.raises(ModelError):
        grid.classify(frame, GridSpec(0.0, 0.0, 8.0, 8.0, 6, 6))
    with pytest.raises(ModelError):
        grid.classify(frame[:, :, :2], GridSpec(0.0, 0.0, 8.0, 8.0, 5, 5))
