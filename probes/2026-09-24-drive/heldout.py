"""Held-out test of the reviewer's prompt: 4 FRESH pockets, two frames each."""
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/projects/secondplayer"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch, torch.nn.functional as F
from PIL import Image, ImageDraw
import transformers

from secondplayer.config import RuntimeConfig
from secondplayer.controller import NEUTRAL
from secondplayer.policy.memory import EpisodicMemory
from secondplayer.policy.openjev import OpenJevPolicy
from secondplayer.policy.grid import GridSpec, dims_line, classify

SP = os.path.expanduser("~/projects/secondplayer/secondplayer/benchmarks/tetris/bench0-reference.png")
QWEN = "Qwen/Qwen3.5-4B"
SPEC = GridSpec(93.0, 44.0, 8.0, 8.2, 10, 20)
base = Image.open(SP).convert("RGB")

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
    """Current + previous (square one row up = natural fall) frames."""
    cur = paint(paint(clean(base), stack_cols, stack_rows), sq_cols, sq_rows_now, **SQ)
    prv = paint(paint(clean(base), stack_cols, stack_rows), sq_cols,
                [r - 1 for r in sq_rows_now], **SQ)
    return prv, cur

ALL = list(range(10))
# Fresh geometries (S1-S3 used well 4-5, rows 14-15/16-19):
scen = {
    # well 6-7, square straddling left of mouth -> RIGHT into well
    "H1": (scenario([c for c in ALL if c not in (6, 7)], [16, 17, 18, 19],
                    [5, 6], [14, 15]), "right"),
    # well 2-3, square straddling right of mouth -> LEFT into well
    "H2": (scenario([c for c in ALL if c not in (2, 3)], [16, 17, 18, 19],
                    [3, 4], [14, 15]), "left"),
    # taller stack (mouth row 15), well 5-6, square left of mouth -> RIGHT
    "H3": (scenario([c for c in ALL if c not in (5, 6)], [15, 16, 17, 18, 19],
                    [4, 5], [13, 14]), "right"),
    # taller stack, well 3-4, square right of mouth -> LEFT
    "H4": (scenario([c for c in ALL if c not in (3, 4)], [15, 16, 17, 18, 19],
                    [4, 5], [13, 14]), "left"),
}

import numpy as np

def filled_set(img):
    frame = np.asarray(img)
    mask, _ = classify(frame, SPEC)
    return {(r, c) for r in range(20) for c in range(10) if mask[r][c]}

def fmt(cells):
    return "none" if not cells else " ".join(f"({r},{c})" for r, c in sorted(cells))

policy = OpenJevPolicy(RuntimeConfig())
mem = EpisodicMemory()
OBS = "motion middle (small)"
screens = {}
for name, ((prv, cur), exp) in scen.items():
    now = filled_set(cur)
    prev = filled_set(prv)
    new_f = now - prev
    new_e = prev - now
    unch = now & prev
    print(f"=== {name} exp={exp} ===", flush=True)
    print(f"NOW: {fmt(now)}", flush=True)
    print(f"PREV: {fmt(prev)}", flush=True)
    print(f"NEWF: {fmt(new_f)} NEWE: {fmt(new_e)}", flush=True)
    screen = (
        f"{dims_line(SPEC)}\n"
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
    screens[name] = (policy._state_text(1, NEUTRAL, "tetris", screen, mem, OBS), exp)

SYS_REV = ("You are the game's controller. Convert the visible state into one immediate "
    "controller direction. Use the stated objective, the Game field, game knowledge "
    "already in your weights, and the supplied pixel-derived state. Use the temporal "
    "occupancy fields to distinguish current motion from static structure instead of "
    "guessing from shape. NEUTRAL and KEEP are ordinary actions, not abstentions or "
    "safe defaults; choose them only when they are actually the best action. Output "
    "exactly one option letter and nothing else.")
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

tok = transformers.AutoTokenizer.from_pretrained(QWEN, trust_remote_code=False)
qm = transformers.AutoModelForImageTextToText.from_pretrained(
    QWEN, dtype=torch.bfloat16, device_map={"": "cpu"}, trust_remote_code=False)
from quanto import freeze, qint4, quantize
quantize(qm, weights=qint4, activations=None); freeze(qm)
qm.to("xpu"); qm.eval(); torch.xpu.synchronize()
L = [tok.convert_tokens_to_ids(c) for c in "ABCDEFGHIJ"]

ok = 0
for name, (state, exp) in screens.items():
    opts = "\n".join(f"{chr(65+i)}. {n}: {DESCS[n]}" for i, n in enumerate(NAMES))
    user = f"{state}\n\nQuestion: {Q_REV}\n\nOptions:\n{opts}\n\nAnswer with one letter: " + \
        ", ".join(chr(65+i) for i in range(len(NAMES))) + "."
    msgs = [{"role": "system", "content": SYS_REV}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)
    ids = tok(text, add_special_tokens=False).input_ids
    kw = {"input_ids": torch.tensor([ids], dtype=torch.long).to("xpu")}
    ts = time.time()
    with torch.no_grad():
        logits = qm(**kw).logits[0, -1, L[:len(NAMES)]].float()
    torch.xpu.synchronize()
    probs = F.softmax(logits, -1).tolist()
    top = max(range(len(NAMES)), key=probs.__getitem__)
    got = NAMES[top]
    ok += got == exp
    dist = " ".join(f"{n}={p:.2f}" for n, p in zip(NAMES, probs))
    print(f"HELDOUT {'PASS' if got==exp else 'FAIL'} {name} exp={exp} got={got} "
          f"{(time.time()-ts)*1000:.0f}ms", flush=True)
    print(f"   {dist}", flush=True)
print(f"== {ok}/{len(screens)} (gate: 3/4) ==", flush=True)
