"""Action quiz: can the 4B map board state to obvious moves? Real prompt path."""
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/projects/secondplayer"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch, torch.nn.functional as F
from PIL import Image, ImageDraw
import transformers

from secondplayer.config import RuntimeConfig
from secondplayer.controller import NEUTRAL, MOVEMENT_CHOICES
from secondplayer.policy.memory import EpisodicMemory
from secondplayer.policy.openjev import (
    OpenJevPolicy, MOVE_CRITERION, _MOVE_DESCRIPTIONS, MENU_HINT,
)
from secondplayer.policy.grid import GridSpec, adaptive_spec, dims_line, render

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

# cover the base frame's own falling blob with well bg
def clean(im):
    return paint(im, [3, 4, 5, 6, 7], [5, 6, 7, 8], color=(61, 55, 0))

SQ = dict(color=(0, 200, 200))
scenarios = {
    # tower left + square falling left-of-center -> move RIGHT to open space
    "R1": (paint(paint(clean(base), [0, 1, 2, 3], [14, 15, 16, 17, 18, 19]),
                 [1, 2], [5, 6], **SQ), "right"),
    # mirror -> LEFT
    "L1": (paint(paint(clean(base), [6, 7, 8, 9], [14, 15, 16, 17, 18, 19]),
                 [7, 8], [5, 6], **SQ), "left"),
    # tall-left staircase, piece center-left -> RIGHT
    "R2": (paint(paint(paint(clean(base), [0, 1, 2, 3, 4], [16, 17, 18, 19]),
                       [0, 1, 2], [12, 13, 14, 15]), [3, 4], [8, 9], **SQ), "right"),
    # mirror -> LEFT
    "L2": (paint(paint(paint(clean(base), [5, 6, 7, 8, 9], [16, 17, 18, 19]),
                       [7, 8, 9], [12, 13, 14, 15]), [5, 6], [8, 9], **SQ), "left"),
}

import numpy as np
policy = OpenJevPolicy(RuntimeConfig())
mem = EpisodicMemory()
OBS = "first screen, nothing to compare yet"
states = {}
for name, (img, exp) in scenarios.items():
    frame = np.asarray(img)
    screen = f"{dims_line(SPEC)}\n{render(None, frame, SPEC)}"
    states[name] = (policy._state_text(1, NEUTRAL, "tetris", screen, mem, OBS), exp)
print("=== R1 state head ===")
print(states["R1"][0][:600], flush=True)

tok = transformers.AutoTokenizer.from_pretrained(QWEN, trust_remote_code=False)
qm = transformers.AutoModelForImageTextToText.from_pretrained(
    QWEN, dtype=torch.bfloat16, device_map={"": "cpu"}, trust_remote_code=False)
from quanto import freeze, qint4, quantize
quantize(qm, weights=qint4, activations=None); freeze(qm)
qm.to("xpu"); qm.eval(); torch.xpu.synchronize()

SYS = "You are a decision readout. Read the state and the question, then answer with exactly one letter and nothing else."

def ask(state, names, rotate=False):
    if rotate:
        names = names[1:] + names[:1]
    opts = "\n".join(f"{chr(65+i)}. {n}: {_MOVE_DESCRIPTIONS[n]}" for i, n in enumerate(names))
    user = f"{state}\n\nQuestion: {MOVE_CRITERION}\n\nOptions:\n{opts}\n\nAnswer with one letter: " + \
        ", ".join(chr(65+i) for i in range(len(names))) + "."
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)
    ids = tok(text, add_special_tokens=False).input_ids
    L = [tok.convert_tokens_to_ids(c) for c in "ABCDEFGHIJ"]
    kw = {"input_ids": torch.tensor([ids], dtype=torch.long).to("xpu")}
    ts = time.time()
    with torch.no_grad():
        logits = qm(**kw).logits[0, -1, L[:len(names)]].float()
    torch.xpu.synchronize()
    probs = F.softmax(logits, -1).tolist()
    top = max(range(len(names)), key=probs.__getitem__)
    return names[top], probs, (time.time()-ts)*1000

for rotate in (False, True):
    ok = 0
    for name, (state, exp) in states.items():
        got, probs, t = ask(state, list(MOVEMENT_CHOICES), rotate=rotate)
        ok += got == exp
        tag = "ROT" if rotate else "STD"
        dist = " ".join(f"{n}={p:.2f}" for n, p in zip(
            list(MOVEMENT_CHOICES)[1:]+list(MOVEMENT_CHOICES)[:1] if rotate else list(MOVEMENT_CHOICES), probs))
        print(f"{tag} {'PASS' if got==exp else 'FAIL'} {name} exp={exp} got={got} {t:.0f}ms", flush=True)
        print(f"   {dist}", flush=True)
    print(f"== {'ROT' if rotate else 'STD'}: {ok}/4 ==", flush=True)
print(f"xpu-peak {torch.xpu.max_memory_allocated()/1e9:.2f}GB", flush=True)
