"""Direct-options quiz: same pockets, options in piece language."""
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
WELL = [0, 1, 2, 3, 6, 7, 8, 9]
scen = {
    "S1": (paint(paint(clean(base), WELL, [16, 17, 18, 19]), [3, 4], [14, 15], **SQ), {"RIGHT"}),
    "S2": (paint(paint(clean(base), WELL, [16, 17, 18, 19]), [5, 6], [14, 15], **SQ), {"LEFT"}),
    "S3": (paint(paint(clean(base), WELL, [16, 17, 18, 19]), [4, 5], [14, 15], **SQ), {"NEUTRAL", "DOWN"}),
}

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
Q = "Which way should the falling square move to reach open space?"
NAMES = ["NEUTRAL", "DOWN", "LEFT", "RIGHT"]
DESCS = {"NEUTRAL": "Let the square keep falling straight.",
         "DOWN": "Drop the square straight down fast.",
         "LEFT": "Move the square one cell left.",
         "RIGHT": "Move the square one cell right."}
L = [tok.convert_tokens_to_ids(c) for c in "ABCD"]

ok = 0
for name, (img, exp) in scen.items():
    frame = np.asarray(img)
    screen = f"{dims_line(SPEC)}\n{render(None, frame, SPEC)}"
    state = policy._state_text(1, NEUTRAL, "tetris", screen, mem, OBS)
    opts = "\n".join(f"{chr(65+i)}. {n}: {DESCS[n]}" for i, n in enumerate(NAMES))
    user = f"{state}\n\nQuestion: {Q}\n\nOptions:\n{opts}\n\nAnswer with one letter: A, B, C, D."
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
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
    ok += got in exp
    print(f"DIRECTOPT {'PASS' if got in exp else 'FAIL'} {name} exp={sorted(exp)} got={got} "
          f"N={probs[0]:.2f} D={probs[1]:.2f} L={probs[2]:.2f} R={probs[3]:.2f} {(time.time()-ts)*1000:.0f}ms", flush=True)
print(f"== {ok}/{len(scen)} ==", flush=True)
