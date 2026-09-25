"""Fact + urgency quiz: fact-reporting on drive scenarios, urgent low-piece actions."""
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
from secondplayer.policy.openjev import OpenJevPolicy, MOVE_CRITERION, _MOVE_DESCRIPTIONS
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
scen = {
    "R1": paint(paint(clean(base), [0, 1, 2, 3], [14, 15, 16, 17, 18, 19]), [1, 2], [5, 6], **SQ),
    "L1": paint(paint(clean(base), [6, 7, 8, 9], [14, 15, 16, 17, 18, 19]), [7, 8], [5, 6], **SQ),
    "R2": paint(paint(paint(clean(base), [0, 1, 2, 3, 4], [16, 17, 18, 19]), [0, 1, 2], [12, 13, 14, 15]), [3, 4], [8, 9], **SQ),
    "L2": paint(paint(paint(clean(base), [5, 6, 7, 8, 9], [16, 17, 18, 19]), [7, 8, 9], [12, 13, 14, 15]), [5, 6], [8, 9], **SQ),
    "RU": paint(paint(clean(base), [0, 1, 2, 3, 4, 5], [15, 16, 17, 18, 19]), [4, 5], [12, 13], **SQ),
    "LU": paint(paint(clean(base), [4, 5, 6, 7, 8, 9], [15, 16, 17, 18, 19]), [4, 5], [12, 13], **SQ),
}
# tall-side ground truth + falling-square half + urgent move
TRUTH = {"R1": ("LEFT", "LEFT", None), "L1": ("RIGHT", "RIGHT", None),
         "R2": ("LEFT", "LEFT", None), "L2": ("RIGHT", "RIGHT", None),
         "RU": ("LEFT", "RIGHT", "right"), "LU": ("RIGHT", "LEFT", "left")}

import numpy as np
policy = OpenJevPolicy(RuntimeConfig())
mem = EpisodicMemory()
OBS = "first screen, nothing to compare yet"
states = {}
for name, img in scen.items():
    frame = np.asarray(img)
    screen = f"{dims_line(SPEC)}\n{render(None, frame, SPEC)}"
    states[name] = policy._state_text(1, NEUTRAL, "tetris", screen, mem, OBS)

tok = transformers.AutoTokenizer.from_pretrained(QWEN, trust_remote_code=False)
qm = transformers.AutoModelForImageTextToText.from_pretrained(
    QWEN, dtype=torch.bfloat16, device_map={"": "cpu"}, trust_remote_code=False)
from quanto import freeze, qint4, quantize
quantize(qm, weights=qint4, activations=None); freeze(qm)
qm.to("xpu"); qm.eval(); torch.xpu.synchronize()

SYS = "You are a decision readout. Read the state and the question, then answer with exactly one letter and nothing else."

def ask(state, question, names, descs):
    opts = "\n".join(f"{chr(65+i)}. {n}: {descs[n]}" for i, n in enumerate(names))
    user = f"{state}\n\nQuestion: {question}\n\nOptions:\n{opts}\n\nAnswer with one letter: " + \
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

ok = tot = 0
Q_TALL = ("Which side holds the taller stack of static blocks?", ["LEFT", "RIGHT"],
          {"LEFT": "The left side.", "RIGHT": "The right side."})
Q_HALF = ("Which half contains the falling square?", ["LEFT", "RIGHT"],
          {"LEFT": "Columns 0-4.", "RIGHT": "Columns 5-9."})
for name in ("R1", "L1", "R2", "L2"):
    for qi, (q, names, descs) in enumerate((Q_TALL, Q_HALF)):
        exp = TRUTH[name][qi]
        got, probs, t = ask(states[name], q, names, descs)
        ok += got == exp; tot += 1
        print(f"FACT {'PASS' if got==exp else 'FAIL'} {name} q{qi} exp={exp} got={got} {[round(p,2) for p in probs]} {t:.0f}ms", flush=True)
for name in ("RU", "LU"):
    exp = TRUTH[name][2]
    got, probs, t = ask(states[name], MOVE_CRITERION, list(MOVEMENT_CHOICES), _MOVE_DESCRIPTIONS)
    ok += got == exp; tot += 1
    print(f"URGENT {'PASS' if got==exp else 'FAIL'} {name} exp={exp} got={got} n={probs[0]:.2f} l={probs[3]:.2f} r={probs[4]:.2f} {t:.0f}ms", flush=True)
print(f"== {ok}/{tot} ==", flush=True)
