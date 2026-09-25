"""Coordinate probe: filled-cell (row,col) lists instead of grids. Same quiz, letter readout."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch, torch.nn.functional as F
from PIL import Image
import transformers

SP = os.path.expanduser("~/projects/secondplayer/secondplayer/benchmarks/tetris/bench0-reference.png")
QWEN = "Qwen/Qwen3.5-4B"
frames = {"orig": Image.open(SP).convert("RGB"),
          "synthA": Image.open("/tmp/ascii_synthA.png").convert("RGB"),
          "synthB": Image.open("/tmp/ascii_synthB.png").convert("RGB")}
OX, OY, CW, CH = 93, 44, 8.0, 8.2

def cell_rgb(im, c, r):
    x0, y0 = int(OX + c * CW), int(OY + r * CH)
    return im.crop((x0 + 2, y0 + 2, x0 + 6, y0 + 6)).resize((1, 1), Image.BILINEAR).getpixel((0, 0))

def coord_text(im):
    cells = [[cell_rgb(im, c, r) for c in range(10)] for r in range(20)]
    flat = [p for row in cells for p in row]
    bg = max(set(flat), key=flat.count)
    filled = [(r, c) for r in range(20) for c in range(10)
              if sum((a - b) ** 2 for a, b in zip(cells[r][c], bg)) ** 0.5 >= 60]
    s = "Filled cells as (row,col), row 0 is top, col 0 is left: "
    s += "none" if not filled else " ".join(f"({r},{c})" for r, c in filled)
    return s, bg, filled

COORDS = {}
for k, v in frames.items():
    s, bg, filled = coord_text(v)
    COORDS[k] = s
    print(f"=== {k} bg={bg} nfilled={len(filled)} ===", flush=True)
    print(s, flush=True)

tok = transformers.AutoTokenizer.from_pretrained(QWEN, trust_remote_code=False)
qm = transformers.AutoModelForImageTextToText.from_pretrained(
    QWEN, dtype=torch.bfloat16, device_map={"": "cpu"}, trust_remote_code=False)
from quanto import freeze, qint4, quantize
quantize(qm, weights=qint4, activations=None); freeze(qm)
qm.to("xpu"); qm.eval(); torch.xpu.synchronize()

SYS = "You are a decision readout. Read the state and the question, then answer with exactly one letter and nothing else."
L = [tok.convert_tokens_to_ids(c) for c in "ABC"]

def ask(coords, question, options):
    user = ("Game: tetris. The board is 10 columns (0-9, left to right) and 20 rows (0-19, top to bottom).\n\n"
            + coords + "\n\nQuestion: " + question + "\n\nOptions:\n" +
            "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options)) +
            "\n\nAnswer with one letter.")
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, add_special_tokens=False).input_ids
    kw = {"input_ids": torch.tensor([ids], dtype=torch.long).to("xpu")}
    ts = __import__("time").time()
    with torch.no_grad():
        logits = qm(**kw).logits[0, -1, L[:len(options)]].float()
    torch.xpu.synchronize()
    probs = F.softmax(logits, -1).tolist()
    top = max(range(len(options)), key=probs.__getitem__)
    return options[top], probs, (__import__("time").time() - ts) * 1000

TESTS = [
    ("orig", "Where is the falling piece?", ["left edge", "middle", "right edge"], "middle"),
    ("orig", "Which columns hold stacked blocks?", ["well is empty", "columns 2-5", "columns 0-1 and 8-9"], "well is empty"),
    ("synthA", "Where is the falling piece?", ["left edge", "middle", "right edge"], "middle"),
    ("synthA", "Which columns hold stacked blocks?", ["well is empty", "columns 2-5", "columns 0-1 and 8-9"], "columns 2-5"),
    ("synthB", "Where is the falling piece?", ["left edge", "middle", "right edge"], "middle"),
    ("synthB", "Which columns hold stacked blocks?", ["well is empty", "columns 2-5", "columns 0-1 and 8-9"], "columns 0-1 and 8-9"),
]
ok = 0; ms = []
for fname, q, opts, exp in TESTS:
    got, probs, t = ask(COORDS[fname], q, opts)
    ok += got == exp; ms.append(t)
    res = "PASS" if got == exp else "FAIL"
    print(f"COORD {res} {fname} {q[:20]!r} got={got!r} {[round(p, 2) for p in probs]} {t:.0f}ms", flush=True)
print(f"== COORD: {ok}/6 mean {sum(ms) / len(ms):.0f}ms ==", flush=True)
print(f"xpu-peak {torch.xpu.max_memory_allocated() / 1e9:.2f}GB", flush=True)
