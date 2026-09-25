"""Rule quiz: one-conditional-hop move from a generic halves digest."""
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/projects/secondplayer"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch, torch.nn.functional as F
import transformers

from secondplayer.config import RuntimeConfig
from secondplayer.controller import NEUTRAL, MOVEMENT_CHOICES
from secondplayer.policy.memory import EpisodicMemory
from secondplayer.policy.openjev import OpenJevPolicy, MOVE_CRITERION, _MOVE_DESCRIPTIONS

QWEN = "Qwen/Qwen3.5-4B"
RULE = ("Rule: hold DOWN when the moving object is over the emptier half "
        "(ties count as over it), otherwise hold toward the emptier half.")

def digest(left, right, mover):
    return (f"Left-half filled cells: {left}. Right-half filled cells: {right}. "
            f"Moving object half: {mover}.\n{RULE}")

CASES = [
    ("D1", digest(24, 0, "RIGHT"), "down"),
    ("D2", digest(0, 24, "LEFT"), "down"),
    ("S1", digest(24, 0, "LEFT"), "right"),
    ("S2", digest(0, 24, "RIGHT"), "left"),
    ("T1", digest(12, 12, "LEFT"), "down"),
]

policy = OpenJevPolicy(RuntimeConfig())
mem = EpisodicMemory()
OBS = "first screen, nothing to compare yet"
states = [(n, policy._state_text(1, NEUTRAL, "tetris", d, mem, OBS), e) for n, d, e in CASES]
print("=== D1 state ===", flush=True)
print(states[0][1][:700], flush=True)

tok = transformers.AutoTokenizer.from_pretrained(QWEN, trust_remote_code=False)
qm = transformers.AutoModelForImageTextToText.from_pretrained(
    QWEN, dtype=torch.bfloat16, device_map={"": "cpu"}, trust_remote_code=False)
from quanto import freeze, qint4, quantize
quantize(qm, weights=qint4, activations=None); freeze(qm)
qm.to("xpu"); qm.eval(); torch.xpu.synchronize()

SYS = "You are a decision readout. Read the state and the question, then answer with exactly one letter and nothing else."

def ask(state):
    names = list(MOVEMENT_CHOICES)
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

ok = 0
for name, state, exp in states:
    got, probs, t = ask(state)
    ok += got == exp
    dist = " ".join(f"{n}={p:.2f}" for n, p in zip(MOVEMENT_CHOICES, probs))
    print(f"RULE {'PASS' if got==exp else 'FAIL'} {name} exp={exp} got={got} {t:.0f}ms", flush=True)
    print(f"   {dist}", flush=True)
print(f"== {ok}/{len(states)} ==", flush=True)
