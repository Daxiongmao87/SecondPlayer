"""Pocket quiz: square one tap from a perfect well fill. Valid ground truth."""
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
from secondplayer.policy.openjev import OpenJevPolicy, _MOVE_DESCRIPTIONS
from secondplayer.policy.grid import GridSpec, dims_line, render

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
WELL = [0, 1, 2, 3, 6, 7, 8, 9]  # rows 16-19 filled except cols 4-5
scen = {
    # square straddling well mouth left -> one RIGHT tap drops it in
    "S1": (paint(paint(clean(base), WELL, [16, 17, 18, 19]), [3, 4], [14, 15], **SQ), {"right"}),
    # mirror -> LEFT
    "S2": (paint(paint(clean(base), WELL, [16, 17, 18, 19]), [5, 6], [14, 15], **SQ), {"left"}),
    # already aligned -> stay (neutral) or drop (down)
    "S3": (paint(paint(clean(base), WELL, [16, 17, 18, 19]), [4, 5], [14, 15], **SQ), {"neutral", "down"}),
}
DYN = "Control note: each decision shifts the moving object by one cell."

import numpy as np
policy = OpenJevPolicy(RuntimeConfig())
mem = EpisodicMemory()
OBS = "first screen, nothing to compare yet"

tok = transformers.AutoTokenizer.from_pretrained(QWEN, trust_remote_code=False)
qm = transformers.AutoModelForImageTextToText.from_pretrained(
    QWEN, dtype=torch.bfloat16, device_map={"": "cpu"}, trust_remote_code=False)
from quanto import freeze, qint4, quantize
quantize(qm, weights=qint4, activations=None); freeze(qm)
qm.to("xpu"); qm.eval(); torch.xpu.synchronize()

SYS = "You are a decision readout. Read the state and the question, then answer with exactly one letter and nothing else."
Q_DIRECT = "Which way should the falling square move to reach open space?"
names = list(MOVEMENT_CHOICES)
L = [tok.convert_tokens_to_ids(c) for c in "ABCDEFGHIJ"]

def ask(state):
    opts = "\n".join(f"{chr(65+i)}. {n}: {_MOVE_DESCRIPTIONS[n]}" for i, n in enumerate(names))
    user = f"{state}\n\nQuestion: {Q_DIRECT}\n\nOptions:\n{opts}\n\nAnswer with one letter: " + \
        ", ".join(chr(65+i) for i in range(len(names))) + "."
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)
    ids = tok(text, add_special_tokens=False).input_ids
    kw = {"input_ids": torch.tensor([ids], dtype=torch.long).to("xpu")}
    ts = time.time()
    with torch.no_grad():
        logits = qm(**kw).logits[0, -1, L[:len(names)]].float()
    torch.xpu.synchronize()
    probs = F.softmax(logits, -1).tolist()
    top = max(range(len(names)), key=probs.__getitem__)
    return names[top], probs, (time.time()-ts)*1000

for with_dyn in (False, True):
    ok = tot = 0
    for name, (img, exp) in scen.items():
        frame = np.asarray(img)
        screen = f"{dims_line(SPEC)}\n{render(None, frame, SPEC)}"
        if with_dyn:
            screen += "\n" + DYN
        state = policy._state_text(1, NEUTRAL, "tetris", screen, mem, OBS)
        got, probs, t = ask(state)
        good = got in exp
        ok += good; tot += 1
        tag = "DYN" if with_dyn else "NODYN"
        print(f"{tag} {'PASS' if good else 'FAIL'} {name} exp={sorted(exp)} got={got} "
              f"n={probs[0]:.2f} d={probs[2]:.2f} l={probs[3]:.2f} r={probs[4]:.2f} {t:.0f}ms", flush=True)
    print(f"== {'DYN' if with_dyn else 'NODYN'}: {ok}/{tot} ==", flush=True)
