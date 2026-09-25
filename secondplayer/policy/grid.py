"""Game-agnostic framebuffer-to-cell-grid text renderer.

Deterministic pixels-only perception for text-only policies: each frame is
sampled on a fixed cell grid (display geometry, never game rules), cells are
split into background (the dominant color) vs filled, and the previous frame
yields the changed-cell set. Output is the coordinate text the letter readout
was probed on (docs/OJCORE_PROBE.md): no learned model, no game vocabulary.

A grid spec is ``ox,oy,cell_w,cell_h,cols,rows`` in pixels (origin and pitch
may be fractional; counts are integers). Empty spec means whole-frame
adaptive: the grid covers the full frame at roughly the validated cell
scale. The spec is display data (like screen resolution), supplied by config.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ..errors import ConfigurationError, ModelError

FILLED_PREFIX = "Current frame filled cells (row,col), row 0 top, col 0 is left: "
CHANGED_PREFIX = "Cells that changed between the previous frame and this frame: "

# Foreground cutoff: euclidean RGB distance from the dominant color. Same
# value the probes validated (docs/OJCORE_PROBE.md).
_FG_DIST = 60.0
# Coordinate lists past this length are truncated with a remainder note so a
# busy unconfigured frame cannot flood the state text.
_MAX_CELLS = 200
# Adaptive target: short side spans about this many cells (~8px cells at
# benchmark resolution, the validated scale).
_ADAPTIVE_CELLS = 28


@dataclass(frozen=True, slots=True)
class GridSpec:
    ox: float
    oy: float
    pw: float
    ph: float
    cols: int
    rows: int


def parse_spec(text: str) -> GridSpec | None:
    """Parse ``ox,oy,cell_w,cell_h,cols,rows``; blank means adaptive (None)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    parts = [part.strip() for part in cleaned.split(",")]
    if len(parts) != 6:
        raise ConfigurationError(
            f"screen_grid must be 'ox,oy,cell_w,cell_h,cols,rows', got {text!r}"
        )
    try:
        ox, oy, pw, ph = (float(parts[i]) for i in range(4))
        cols, rows = (int(parts[4]), int(parts[5]))
    except ValueError as exc:
        raise ConfigurationError(f"screen_grid has non-numeric parts: {text!r}") from exc
    if pw <= 0 or ph <= 0:
        raise ConfigurationError(f"screen_grid cell size must be positive: {text!r}")
    if cols <= 0 or rows <= 0:
        raise ConfigurationError(f"screen_grid needs positive cols/rows: {text!r}")
    return GridSpec(ox, oy, pw, ph, cols, rows)


def adaptive_spec(width: int, height: int) -> GridSpec:
    """Whole-frame grid at roughly the validated cell scale. No constants."""
    pitch = max(1, min(width, height) // _ADAPTIVE_CELLS)
    return GridSpec(0.0, 0.0, float(pitch), float(pitch), width // pitch, height // pitch)


def classify(frame: np.ndarray, spec: GridSpec) -> tuple[list[list[bool]], tuple[int, int, int]]:
    """Per-cell foreground mask plus the dominant background color.

    Each cell is center-sampled (middle half, bilinear single pixel --
    the exact sampler the probes validated); background is the most common
    sample since filled cells are always the minority.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ModelError(f"expected HWC RGB frame, got shape {frame.shape}")
    height, width = frame.shape[:2]
    if spec.ox < 0 or spec.oy < 0 or spec.ox + spec.cols * spec.pw > width + 1e-9:
        raise ModelError(f"grid {spec} exceeds frame width {width}")
    if spec.oy + spec.rows * spec.ph > height + 1e-9:
        raise ModelError(f"grid {spec} exceeds frame height {height}")
    image = Image.fromarray(frame)
    samples: list[tuple[int, int, int]] = []
    for row in range(spec.rows):
        for col in range(spec.cols):
            x0 = int(spec.ox + col * spec.pw)
            x1 = int(spec.ox + (col + 1) * spec.pw)
            y0 = int(spec.oy + row * spec.ph)
            y1 = int(spec.oy + (row + 1) * spec.ph)
            dx = (x1 - x0) // 4
            dy = (y1 - y0) // 4
            px = image.crop((x0 + dx, y0 + dy, x1 - dx, y1 - dy)).resize((1, 1), Image.BILINEAR)
            samples.append(px.getpixel((0, 0)))
    bg = Counter(samples).most_common(1)[0][0]
    mask: list[list[bool]] = []
    for row in range(spec.rows):
        line = []
        for col in range(spec.cols):
            px = samples[row * spec.cols + col]
            dist = sum((a - b) ** 2 for a, b in zip(px, bg)) ** 0.5
            line.append(dist >= _FG_DIST)
        mask.append(line)
    return mask, bg


def _coords(cells: list[tuple[int, int]]) -> str:
    if not cells:
        return "none"
    shown = " ".join(f"({r},{c})" for r, c in cells[:_MAX_CELLS])
    if len(cells) > _MAX_CELLS:
        shown += f" (+{len(cells) - _MAX_CELLS} more)"
    return shown


def render(
    prev: np.ndarray | None, frame: np.ndarray, spec: GridSpec | None = None
) -> str:
    """Two-line state text: filled cells, then changed cells. See module doc."""
    active = spec if spec is not None else adaptive_spec(frame.shape[1], frame.shape[0])
    mask, _ = classify(frame, active)
    filled = [(r, c) for r in range(active.rows) for c in range(active.cols) if mask[r][c]]
    lines = [FILLED_PREFIX + _coords(filled)]
    if prev is None or prev.shape != frame.shape:
        lines.append(CHANGED_PREFIX + "first screen, nothing to compare yet")
    else:
        before, _ = classify(prev, active)
        moved = [
            (r, c)
            for r in range(active.rows)
            for c in range(active.cols)
            if before[r][c] != mask[r][c]
        ]
        lines.append(CHANGED_PREFIX + ("none - everything is static" if not moved else _coords(moved)))
    return "\n".join(lines)


def dims_line(spec: GridSpec) -> str:
    """One-sentence legend so the reader knows the grid extent."""
    return (
        f"The screen grid is {spec.cols} columns (0-{spec.cols - 1}, left to right) "
        f"by {spec.rows} rows (0-{spec.rows - 1}, top to bottom). "
        "Changed cells are the moving object; other filled cells are static."
    )


__all__: list[Any] = ["GridSpec", "parse_spec", "adaptive_spec", "classify", "render", "dims_line"]
