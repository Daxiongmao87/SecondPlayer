"""Ladder: can the brain output LEFT/RIGHT at all? Binary A/B steering."""
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/projects/secondplayer"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch, torch.nn.functional as F
import transformers

from secondplayer.config import RuntimeConfig
from secondplayer.controller import NEUTRAL
from secondplayer.policy.memory import EpisodicMemory
from secondplayer.policy.openjev import OpenJevPolicy

QWEN = "Qwen/Qwen3.5-4B"
CASES = [
    # (name, screen, question, [optA, optB], expected)
    ("L0b", "Hold toward the RIGHT side.", "Which direction should the player hold?",
     ["RIGHT", "LEFT"], "RIGHT"),
    ("L0a", "Hold toward the RIGHT side.", "Which direction should the player hold?",
     ["LEFT", "RIGHT"], "RIGHT"),
    ("L1", "Mover half: LEFT. Emptier half: RIGHT. Hold toward the emptier half.",
     "Which direction should the player hold?", ["LEFT", "RIGHT"], "RIGHT"),
    ("L1flip", "Mover half: RIGHT. Emptier half: LEFT. Hold toward the emptier half.",
     "Which direction should the player hold?", ["LEFT", "RIGHT"], "LEFT"),
]

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
LA = tok.convert_tokens_to_ids("A")
LB = tok.convert_tokens_to_ids("B")
print(f"slot ids: A={LA} B={LB}", flush=True)

ok = 0
for name, screen, q, names, exp in CASES:
    state = policy._state_text(1, NEUTRAL, "tetris", screen, mem, OBS)
    opts = "\n".join(f"{chr(65+i)}. {n}." for i, n in enumerate(names))
    user = f"{state}\n\nQuestion: {q}\n\nOptions:\n{opts}\n\nAnswer with one letter: A, B."
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)
    ids = tok(text, add_special_tokens=False).input_ids
    kw = {"input_ids": torch.tensor([ids], dtype=torch.long).to("xpu")}
    ts = time.time()
    with torch.no_grad():
        logits = qm(**kw).logits[0, -1, [LA, LB]].float()
    torch.xpu.synchronize()
    probs = F.softmax(logits, -1).tolist()
    got = names[0] if probs[0] > probs[1] else names[1]
    ok += got == exp
    print(f"LADDER {'PASS' if got==exp else 'FAIL'} {name} exp={exp} got={got} "
          f"A={probs[0]:.2f} B={probs[1]:.2f} {1000*(time.time()-ts):.0f}ms", flush=True)
print(f"== {ok}/{len(CASES)} ==", flush=True)
