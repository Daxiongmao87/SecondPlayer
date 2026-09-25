"""Direct-language + few-shot action quizzes. Real prompt path, real options."""
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
scen = {
    "R1": paint(paint(clean(base), [0, 1, 2, 3], [14, 15, 16, 17, 18, 19]), [1, 2], [5, 6], **SQ),
    "L1": paint(paint(clean(base), [6, 7, 8, 9], [14, 15, 16, 17, 18, 19]), [7, 8], [5, 6], **SQ),
    "R2": paint(paint(paint(clean(base), [0, 1, 2, 3, 4], [16, 17, 18, 19]), [0, 1, 2], [12, 13, 14, 15]), [3, 4], [8, 9], **SQ),
    "L2": paint(paint(paint(clean(base), [5, 6, 7, 8, 9], [16, 17, 18, 19]), [7, 8, 9], [12, 13, 14, 15]), [5, 6], [8, 9], **SQ),
}
EXP = {"R1": "right", "L1": "left", "R2": "right", "L2": "left"}

import numpy as np
policy = OpenJevPolicy(RuntimeConfig())
mem = EpisodicMemory()
OBS = "first screen, nothing to compare yet"
screens = {}
for name, img in scen.items():
    frame = np.asarray(img)
    screens[name] = f"{dims_line(SPEC)}\n{render(None, frame, SPEC)}"

# abstract few-shot examples (no game words; different geometry from queries)
EX1 = ("Filled: (4,6) (4,7) (5,6) (5,7) (15,7) (15,8) (15,9) (16,7) (16,8) (16,9) "
       "(17,7) (17,8) (17,9) (18,7) (18,8) (18,9) (19,7) (19,8) (19,9). "
       "Falling block at columns 6-7, stack on the right. Move: A")
EX2 = ("Filled: (4,3) (4,4) (5,3) (5,4) (15,0) (15,1) (15,2) (16,0) (16,1) (16,2) "
       "(17,0) (17,1) (17,2) (18,0) (18,1) (18,2) (19,0) (19,1) (19,2). "
       "Falling block at columns 3-4, stack on the left. Move: B")
FEWSHOT = f"Example 1. {EX1}\nExample 2. {EX2}"

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
    top = max(range(len(options := names)), key=probs.__getitem__)
    return names[top], probs, (time.time()-ts)*1000

Q_DIRECT = "Which way should the falling square move to reach open space?"
ok = tot = 0
for name in ("R1", "L1", "R2", "L2"):
    state = policy._state_text(1, NEUTRAL, "tetris", screens[name], mem, OBS)
    got, probs, t = ask(state, Q_DIRECT, list(MOVEMENT_CHOICES), _MOVE_DESCRIPTIONS)
    ok += got == EXP[name]; tot += 1
    print(f"DIRECT {'PASS' if got==EXP[name] else 'FAIL'} {name} exp={EXP[name]} got={got} "
          f"n={probs[0]:.2f} l={probs[3]:.2f} r={probs[4]:.2f} {t:.0f}ms", flush=True)

Q_FS = "Study the examples, then move the falling square in the new state."
for name, exp_letter in (("R1", "B"), ("L1", "A")):
    screen = screens[name] + "\n" + FEWSHOT
    state = policy._state_text(1, NEUTRAL, "tetris", screen, mem, OBS)
    got, probs, t = ask(state, Q_FS, ["LEFT", "RIGHT"], {"LEFT": "Steer left.", "RIGHT": "Steer right."})
    exp = "RIGHT" if exp_letter == "B" else "LEFT"
    ok += got == exp; tot += 1
    print(f"FEWSHOT {'PASS' if got==exp else 'FAIL'} {name} exp={exp} got={got} "
          f"A={probs[0]:.2f} B={probs[1]:.2f} {t:.0f}ms", flush=True)
print(f"== {ok}/{tot} ==", flush=True)
