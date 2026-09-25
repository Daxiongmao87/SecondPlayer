"""Laya-201m mirrored SNES discrimination via the production policy.

Feeds the H5-H8/A1 held-out pocket frames (same geometries as
probes/2026-09-24-drive/pjquiz2.py) through LayaVisionPolicy.decide()
with the normal movement choice interface and learn=False, so every
output is a pure model distribution (no harness-forced sweep actions).
One fresh policy per state (empty history) sharing a single loaded
agent. Reports whether mirrored LEFT-vs-RIGHT states yield materially
different per-option distributions.
"""
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/projects/secondplayer"))
import numpy as np
from PIL import Image, ImageDraw

from secondplayer.config import RuntimeConfig
from secondplayer.policy.laya import LayaVisionPolicy

SP = os.path.expanduser(
    "~/projects/secondplayer/secondplayer/benchmarks/tetris/bench0-reference.png")
base = Image.open(SP).convert("RGB")


def paint(im, cols, rows, color=(30, 144, 255)):
    im = im.copy()
    d = ImageDraw.Draw(im)
    for c in cols:
        for r in rows:
            x0, y0 = int(93 + c * 8.0), int(44 + r * 8.2)
            d.rectangle([x0, y0, x0 + 7, y0 + 7], fill=color)
    return im


def clean(im):
    return paint(im, [3, 4, 5, 6, 7], [5, 6, 7, 8], color=(61, 55, 0))


SQ = dict(color=(0, 200, 200))


def scenario(stack_cols, stack_rows, sq_cols, sq_rows_now):
    cur = paint(paint(clean(base), stack_cols, stack_rows), sq_cols, sq_rows_now, **SQ)
    prv = paint(paint(clean(base), stack_cols, stack_rows), sq_cols,
                [r - 1 for r in sq_rows_now], **SQ)
    return prv, cur


ALL = list(range(10))
scen = {
    "H5": (scenario([c for c in ALL if c not in (7, 8)], [16, 17, 18, 19], [6, 7], [14, 15]),
           "right"),
    "H6": (scenario([c for c in ALL if c not in (1, 2)], [16, 17, 18, 19], [2, 3], [14, 15]),
           "left"),
    "H7": (scenario([c for c in ALL if c not in (8, 9)], [15, 16, 17, 18, 19], [7, 8], [13, 14]),
           "right"),
    "H8": (scenario([c for c in ALL if c not in (0, 1)], [15, 16, 17, 18, 19], [1, 2], [13, 14]),
           "left"),
    "A1": (scenario([c for c in ALL if c not in (6, 7)], [16, 17, 18, 19], [6, 7], [14, 15]),
           "down"),
}

cfg = RuntimeConfig(
    model="thaitea/laya-vision-201m",
    model_revision="0b6228f7a0762566de1c4539e9aa4eb1c1aef5f4",
    device="auto",
    learn=False,
    permutations=1,
    window_frames=6,
)
t0 = time.time()
shared = LayaVisionPolicy(cfg)
shared.load()
print(f"LOAD {(time.time() - t0):.1f}s :: {shared.model_info()}", flush=True)

for name, ((prv, cur), exp) in scen.items():
    pol = LayaVisionPolicy(cfg)
    pol.agent = shared.agent  # fresh history, shared weights
    frame = np.asarray(cur.convert("RGB"))
    t0 = time.perf_counter()
    dec = pol.decide(frame, [1], game="tetris")[1]
    ms = (time.perf_counter() - t0) * 1000.0
    mv = dec.raw["movement"]
    dist = " ".join(f"{k}={v:.3f}" for k, v in mv["probabilities"].items())
    print(f"MIRROR {name} exp={exp} got={mv['choice']} "
          f"conf={dec.movement_confidence:.3f} {ms:.0f}ms :: {dist}", flush=True)
