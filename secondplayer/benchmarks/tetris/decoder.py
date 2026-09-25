"""Tetris & Dr. Mario (USA) HUD decoder: score/lines/level/stats + GAME OVER.

Positions are locked to ROM sha256
62cef84f03ea6f3d0a05cc399731da193e01f0aee54e01b13ad08fb98b4a427c
(SMC, 1049088 bytes incl. 512-byte header).

All HUD digits share one 8px-cell white-on-maze font. Glyph pixels are pure
white (channel sum > 600); the maze background never is, so classification
uses the white mask only and requires an EXACT template match. Any cell that
does not exactly match a known digit (line-clear flash, animation, garbage)
decodes as None instead of misreading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_TPL_DIR = Path(__file__).with_name("templates")

# (x, y) of the 8x9 cell top-left; glyph rows are y+1..y+7.
SCORE_CELLS = [(192 + 8 * n, 127) for n in range(6)]
TOP_CELLS = [(192 + 8 * n, 103) for n in range(6)]
LINES_CELLS = [(152 + 8 * n, 23) for n in range(3)]
LEVEL_CELLS = [(208 + 8 * n, 167) for n in range(2)]
STATS_CELLS = [[(48 + 8 * j, 95 + 16 * k) for j in range(3)] for k in range(7)]

WHITE_THRESHOLD = 600
_CELL_W, _CELL_H = 8, 9


def _load_templates() -> dict[str, np.ndarray]:
    templates = {}
    for digit in "0123456789":
        path = _TPL_DIR / f"d{digit}W.npy"
        templates[digit] = np.load(path).astype(np.uint8)
    return templates


_TEMPLATES: dict[str, np.ndarray] | None = None


def templates() -> dict[str, np.ndarray]:
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = _load_templates()
    return _TEMPLATES


def white_mask(frame: np.ndarray) -> np.ndarray:
    """Binary white mask for a 256x224x3 RGB frame."""
    return (frame.astype(int).sum(axis=2) > WHITE_THRESHOLD).astype(np.uint8)


def classify_cell(mask: np.ndarray, x: int, y: int) -> str | None:
    """Classify one 8x9 digit cell; None when no template matches exactly."""
    cell = mask[y : y + _CELL_H, x : x + _CELL_W]
    if cell.shape != (_CELL_H, _CELL_W):
        return None
    for digit, template in templates().items():
        if np.array_equal(cell, template):
            return digit
    return None


def read_number(mask: np.ndarray, cells: list[tuple[int, int]]) -> int | None:
    """Read a multi-digit field; None if any cell is unreadable."""
    digits = [classify_cell(mask, x, y) for x, y in cells]
    if any(d is None for d in digits):
        return None
    return int("".join(digits))  # type: ignore[arg-type]


def read_score(mask: np.ndarray) -> int | None:
    return read_number(mask, SCORE_CELLS)


def read_top(mask: np.ndarray) -> int | None:
    return read_number(mask, TOP_CELLS)


def read_lines(mask: np.ndarray) -> int | None:
    return read_number(mask, LINES_CELLS)


def read_level(mask: np.ndarray) -> int | None:
    return read_number(mask, LEVEL_CELLS)


def read_stats(mask: np.ndarray) -> list[int | None]:
    return [read_number(mask, row) for row in STATS_CELLS]


# GAME OVER splash: white text over the well (text bbox rows 88-113,
# cols 120-149, 325 white pixels). Gameplay reads 0-4 here; line-clear
# flashes can spike it, so the harness only ends a run after two consecutive
# positive reads (~1s apart), which a sub-second flash cannot produce.
GAMEOVER_ROWS = (88, 120)
GAMEOVER_COLS = (96, 160)
GAMEOVER_MIN_WHITE = 200


def is_game_over(mask: np.ndarray) -> bool:
    region = mask[GAMEOVER_ROWS[0] : GAMEOVER_ROWS[1], GAMEOVER_COLS[0] : GAMEOVER_COLS[1]]
    return int(region.sum()) >= GAMEOVER_MIN_WHITE
