"""Clean held-out round: DEPLOYED qint4e via production scorer, 4 NEW pockets."""
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/projects/secondplayer"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from PIL import Image, ImageDraw

from secondplayer.config import RuntimeConfig
from secondplayer.controller import NEUTRAL
from secondplayer.policy.memory import EpisodicMemory, FrameDiff
from secondplayer.policy.openjev import OpenJevPolicy
from secondplayer.policy.grid import GridSpec, classify
from secondplayer.policy.ojcore import load_model, load_tokenizer, score as ojscore
import secondplayer.policy.ojcore.messages as ojmsg

SP = os.path.expanduser("~/projects/secondplayer/secondplayer/benchmarks/tetris/bench0-reference.png")
QDIR = "/data/tmp/ojcore/qwen3.5-4b-qint4e"
SPEC = GridSpec(93.0, 44.0, 8.0, 8.2, 10, 20)
base = Image.open(SP).convert("RGB")

SYS_REV = ("You are the game's controller. Convert the visible state into one immediate "
    "controller direction. Use the stated objective, the Game field, game knowledge "
    "already in your weights, and the supplied pixel-derived state. Use the temporal "
    "occupancy fields to distinguish current motion from static structure instead of "
    "guessing from shape. NEUTRAL and KEEP are ordinary actions, not abstentions or "
    "safe defaults; choose them only when they are actually the best action. Output "
    "exactly one option letter and nothing else.")
# The system line is part of the prompt under test; everything else below is the
# untouched production path (template flags, slots, subset head + sidecar).
ojmsg.SYSTEM_INSTRUCTIONS = SYS_REV

Q_REV = ("Which one controller direction should you execute now to best advance the "
    "stated objective from the current visible state?")
NAMES = ["neutral", "up", "down", "left", "right", "up-left", "up-right",
         "down-left", "down-right", "keep"]
DESCS = {
    "neutral": "Make no directional change this decision.",
    "up": "Apply UP, toward smaller row numbers.",
    "down": "Apply DOWN, toward larger row numbers.",
    "left": "Apply LEFT, toward smaller column numbers.",
    "right": "Apply RIGHT, toward larger column numbers.",
    "up-left": "Apply UP and LEFT together, toward smaller row and column numbers.",
    "up-right": "Apply UP and RIGHT together, toward smaller row numbers and larger column numbers.",
    "down-left": "Apply DOWN and LEFT together, toward larger row numbers and smaller column numbers.",
    "down-right": "Apply DOWN and RIGHT together, toward larger row and column numbers.",
    "keep": "Continue the previous directional input unchanged.",
}
# Effective-action sets: exact lateral, or that lateral + DOWN (soft drop is
# compatible with any lateral placement; UP effects are unmapped -> excluded).
EFF = {"right": {"right", "down-right"}, "left": {"left", "down-left"}}

def paint(im, cols, rows, color=(30, 144, 255)):
    im = im.copy(); d = ImageDraw.Draw(im)
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
# NEW wells (prior rounds used 4-5, 6-7, 2-3, 5-6, 3-4):
scen = {
    "H5": (scenario([c for c in ALL if c not in (7, 8)], [16, 17, 18, 19],
                    [6, 7], [14, 15]), "right"),
    "H6": (scenario([c for c in ALL if c not in (1, 2)], [16, 17, 18, 19],
                    [2, 3], [14, 15]), "left"),
    "H7": (scenario([c for c in ALL if c not in (8, 9)], [15, 16, 17, 18, 19],
                    [7, 8], [13, 14]), "right"),
    "H8": (scenario([c for c in ALL if c not in (0, 1)], [15, 16, 17, 18, 19],
                    [1, 2], [13, 14]), "left"),
}

import numpy as np

def filled_set(frame):
    mask, _ = classify(frame, SPEC)
    return {(r, c) for r in range(20) for c in range(10) if mask[r][c]}

def fmt(cells):
    return "none" if not cells else " ".join(f"({r},{c})" for r, c in sorted(cells))

LEGEND = ("The screen grid is 10 columns (0-9, left to right) by "
          "20 rows (0-19, top to bottom).")

print("loading deployed artifact via production loader...", flush=True)
tok = load_tokenizer(QDIR, offline=True)
model, compute = load_model(QDIR, "auto", quant="quanto_qint4", offline=True)
print(f"device: {compute.label} head: {'dropped' if getattr(model, 'lm_head', None) is None else 'live'}",
      flush=True)
policy = OpenJevPolicy(RuntimeConfig())

exact = eff = 0
for name, ((prv_img, cur_img), exp) in scen.items():
    prv = np.asarray(prv_img); cur = np.asarray(cur_img)
    now = filled_set(cur); prev = filled_set(prv)
    new_f, new_e, unch = now - prev, prev - now, now & prev
    print(f"=== {name} exp={exp} === NEWF:{fmt(new_f)} NEWE:{fmt(new_e)}", flush=True)
    screen = (
        f"{LEGEND}\n"
        "Temporal occupancy from two consecutive framebuffer observations, older first:\n"
        f"FILLED NOW: {fmt(now)}\n"
        f"FILLED PREVIOUSLY: {fmt(prev)}\n"
        f"NEWLY FILLED: {fmt(new_f)}\n"
        f"NEWLY EMPTIED: {fmt(new_e)}\n"
        f"UNCHANGED FILLED: {fmt(unch)}\n"
        "NEWLY FILLED and NEWLY EMPTIED report occupancy changes only. Use them "
        "together with FILLED NOW, FILLED PREVIOUSLY, the game, and the objective "
        "to infer what is currently moving. Do not assume every changed cell is "
        "part of the object's current position."
    )
    mem = EpisodicMemory()
    obs = FrameDiff.summarize(prv, cur)
    mem.note_observed(obs)
    state = policy._state_text(1, NEUTRAL, "tetris", screen, mem, obs)
    row = {"id": f"heldout-{name}", "state": state, "question": Q_REV,
           "options": [{"id": n, "description": DESCS[n]} for n in NAMES]}
    ts = time.time()
    res = ojscore(model, tok, row, {"weights": QDIR})
    t = (time.time() - ts) * 1000
    probs = res["probabilities"]
    got = res["option_ids"][max(range(len(probs)), key=probs.__getitem__)]
    exact += got == exp; eff += got in EFF[exp]
    dist = " ".join(f"{n}={p:.2f}" for n, p in zip(res["option_ids"], probs))
    print(f"CLEAN {'PASS' if got==exp else 'fail'} {name} exp={exp} got={got} "
          f"effective={got in EFF[exp]} {t:.0f}ms", flush=True)
    print(f"   {dist}", flush=True)
print(f"== exact {exact}/4, effective {eff}/4 (gate: 3/4 exact) ==", flush=True)
