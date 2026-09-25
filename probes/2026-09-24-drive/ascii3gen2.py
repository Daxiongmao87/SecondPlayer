"""ASCII probe v3b: generation read test, thinking disabled, answers only."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from PIL import Image
import transformers

SP = os.path.expanduser("~/projects/secondplayer/secondplayer/benchmarks/tetris/bench0-reference.png")
QWEN = "Qwen/Qwen3.5-4B"
frames = {"orig": Image.open(SP).convert("RGB"),
          "synthA": Image.open("/tmp/ascii_synthA.png").convert("RGB"),
          "synthB": Image.open("/tmp/ascii_synthB.png").convert("RGB")}
OX, OY, CW, CH = 93, 44, 8.0, 8.2
PAL = {"B": (30, 144, 255), "C": (0, 255, 255), "M": (255, 0, 255), "Y": (255, 255, 0),
       "R": (255, 0, 0), "G": (0, 255, 0), "W": (255, 255, 255), "O": (255, 165, 0)}

def cell_rgb(im, c, r):
    x0, y0 = int(OX + c * CW), int(OY + r * CH)
    return im.crop((x0 + 2, y0 + 2, x0 + 6, y0 + 6)).resize((1, 1), Image.BILINEAR).getpixel((0, 0))

def well_ascii(im):
    cells = [[cell_rgb(im, c, r) for c in range(10)] for r in range(20)]
    flat = [p for row in cells for p in row]
    bg = max(set(flat), key=flat.count)
    def glyph(p):
        if sum((a - b) ** 2 for a, b in zip(p, bg)) ** 0.5 < 60:
            return "."
        return min(PAL, key=lambda k: sum((a - c) ** 2 for a, c in zip(p, PAL[k])))
    return "".join(str(c % 10) for c in range(10)) + "\n" + "\n".join(
        "".join(glyph(p) for p in row) for row in cells)

ARTS = {k: well_ascii(v) for k, v in frames.items()}

tok = transformers.AutoTokenizer.from_pretrained(QWEN, trust_remote_code=False)
qm = transformers.AutoModelForImageTextToText.from_pretrained(
    QWEN, dtype=torch.bfloat16, device_map={"": "cpu"}, trust_remote_code=False)
from quanto import freeze, qint4, quantize
quantize(qm, weights=qint4, activations=None); freeze(qm)
qm.to("xpu"); qm.eval(); torch.xpu.synchronize()

SYS = ("You are a precise grid reader. The user shows a Tetris board as text: 10 columns "
       "numbered 0-9 in the top ruler, 20 rows below. . means empty, B means a block. "
       "Answer directly with no preamble and no thinking process.")

def gen(fname, task):
    user = f"Board:\n{ARTS[fname]}\n\nTask: {task}"
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, add_special_tokens=False).input_ids
    inp = torch.tensor([ids], dtype=torch.long).to("xpu")
    with torch.no_grad():
        out = qm.generate(inp, max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, len(ids):], skip_special_tokens=True).strip()

CASES = [
    ("synthA", "List the columns that contain B blocks in the bottom three rows. Reply with just the column numbers, e.g. 2, 3, 4, 5."),
    ("synthB", "List the columns that contain B blocks in the bottom two rows. Reply with just the column numbers."),
    ("orig", "Are there any B blocks in the bottom five rows? Reply with YES or NO only."),
    ("orig", "The cluster of B blocks near the top sits in: LEFT third (cols 0-3), MIDDLE third (cols 3-6), or RIGHT third (cols 7-9)? Reply with one word: LEFT, MIDDLE, or RIGHT."),
    ("synthA", "In one sentence: which columns are stacked at the bottom and where is the falling cluster at the top?"),
]
for fname, task in CASES:
    print(f"===== {fname} :: {task[:70]}", flush=True)
    print(gen(fname, task), flush=True)
print("DONE", flush=True)
