"""Tests for the Tetris & Dr. Mario HUD decoder over committed fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from secondplayer.benchmarks.tetris import decoder as D

FIXTURES = Path(__file__).with_name("fixtures") / "tetris"


def load(name: str) -> np.ndarray:
    with Image.open(FIXTURES / name) as image:
        return np.asarray(image.convert("RGB"))


def test_templates_cover_all_digits_exactly_once():
    templates = D.templates()
    assert sorted(templates) == list("0123456789")
    seen = set()
    for digit, mask in templates.items():
        assert mask.shape == (9, 8)
        assert mask.sum() > 0
        key = mask.tobytes()
        assert key not in seen, f"duplicate template for {digit}"
        seen.add(key)


@pytest.mark.parametrize(
    ("name", "score", "lines", "level", "stats", "game_over"),
    [
        ("gameplay-start.png", 0, 0, 0, [0, 0, 0, 0, 0, 1, 0], False),
        ("gameplay-68.png", 68, 0, 0, [1, 1, 1, 0, 1, 2, 0], False),
        ("gameplay-99.png", 99, 0, 0, [1, 1, 1, 0, 3, 2, 0], False),
        ("gameover-80.png", 80, 0, 0, [2, 2, 1, 0, 3, 3, 0], True),
    ],
)
def test_fixture_reads(name, score, lines, level, stats, game_over):
    mask = D.white_mask(load(name))
    assert D.read_score(mask) == score
    assert D.read_top(mask) == 10000
    assert D.read_lines(mask) == lines
    assert D.read_level(mask) == level
    assert D.read_stats(mask) == stats
    assert D.is_game_over(mask) == game_over


def test_unreadable_cell_yields_none_not_guess():
    mask = D.white_mask(load("gameplay-start.png"))
    mask[127:136, 192:200] = 0  # blank the first score cell
    assert D.read_score(mask) is None
