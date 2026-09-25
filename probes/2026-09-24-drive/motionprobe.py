"""Motion probe: current-frame coords + changed-cells list. Thinking disabled."""
import os, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch, torch.nn.functional as F
from PIL import Image, ImageDraw
import transformers

SP = os.path.expanduser("~/projects/secondplayer/secondplayer/benchmarks/tetris/bench0-reference.png")
QWEN = "Qwen/Qwen3.5-4B"
base = Image.open(SP).convert("RGB")
OX, OY, CW, CH = 93, 44, 8.0, 8.2

def paint(im, cols, rows, color=(30, 144, 255)):
    im = im.copy(); d = ImageDraw.Draw(im)
    for c in cols:
        for r in rows:
            x0, y0 = int(OX + c * CW), int(OY + r * CH)
            x1, y1 = int(OX + (c + 1) * CW) - 1, int(OY + (r + 1) * CH) - 1
            d.rectangle([x0, y0, x1, y1], fill=color)
    return im

# Pair M: static stack cols 2-5 rows 17-19 + object at rows 10-11 -> rows 11-12
m1 = paint(paint(base, [2, 3, 4, 5], [17, 18, 19]), [4, 5], [10, 11])
m2 = paint(paint(base, [2, 3, 4, 5], [17, 18, 19]), [4, 5], [11, 12])
# Pair S: static stack cols 0,1,8,9 rows 18-19, no motion
s1 = paint(base, [0, 1, 8, 9], [18, 19])
s2 = paint(base, [0, 1, 8, 9], [18, 19])
# Pair E: true empty well (falling blob covered with bg), no motion
e1 = paint(base, [4, 5, 6], [6, 7], color=(61, 55, 0))
e2 = e1.copy()

def cell_rgb(im, c, r):
    x0, y0 = int(OX + c * CW), int(OY + r * CH)
    return im.crop((x0 + 2, y0 + 2, x0 + 6, y0 + 6)).resize((1, 1), Image.BILINEAR).getpixel((0, 0))

def classify(im):
    cells = [[cell_rgb(im, c, r) for c in range(10)] for r in range(20)]
    flat = [p for row in cells for p in row]
    bg = max(set(flat), key=flat.count)
    fg = [[sum((a - b) ** 2 for a, b in zip(p, bg)) ** 0.5 >= 60 for p in row] for row in cells]
    return fg

def state_text(prev, cur):
    fg0, fg1 = classify(prev), classify(cur)
    filled = [(r, c) for r in range(20) for c in range(10) if fg1[r][c]]
    moved = [(r, c) for r in range(20) for c in range(10) if fg0[r][c] != fg1[r][c]]
    s = ("Current frame filled cells (row,col), row 0 top, col 0 left: " +
         ("none" if not filled else " ".join(f"({r},{c})" for r, c in filled)))
    s += ("\nCells that changed between the previous frame and this frame: " +
          ("none - everything is static" if not moved else " ".join(f"({r},{c})" for r, c in moved)))
    return s

STATES = {"M": state_text(m1, m2), "S": state_text(s1, s2), "E": state_text(e1, e2)}
for k, s in STATES.items():
    print(f"=== pair {k} ===", flush=True)
    print(s, flush=True)

tok = transformers.AutoTokenizer.from_pretrained(QWEN, trust_remote_code=False)
qm = transformers.AutoModelForImageTextToText.from_pretrained(
    QWEN, dtype=torch.bfloat16, device_map={"": "cpu"}, trust_remote_code=False)
from quanto import freeze, qint4, quantize
quantize(qm, weights=qint4, activations=None); freeze(qm)
qm.to("xpu"); qm.eval(); torch.xpu.synchronize()

SYS = "You are a decision readout. Read the state and the question, then answer with exactly one letter and nothing else."
L = [tok.convert_tokens_to_ids(c) for c in "ABC"]

def template(msgs):
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

def ask(state, question, options):
    user = ("Game: tetris. The board is 10 columns (0-9, left to right) and 20 rows (0-19, top to bottom). "
            "Changed cells are the moving object; other filled cells are static.\n\n" + state +
            "\n\nQuestion: " + question + "\n\nOptions:\n" +
            "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options)) +
            "\n\nAnswer with one letter.")
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    ids = tok(template(msgs), add_special_tokens=False).input_ids
    kw = {"input_ids": torch.tensor([ids], dtype=torch.long).to("xpu")}
    ts = time.time()
    with torch.no_grad():
        logits = qm(**kw).logits[0, -1, L[:len(options)]].float()
    torch.xpu.synchronize()
    probs = F.softmax(logits, -1).tolist()
    top = max(range(len(options)), key=probs.__getitem__)
    return options[top], probs, (time.time() - ts) * 1000

TESTS = [
    ("M", "Which columns hold static stacked blocks? Ignore the moving object.",
     ["well is empty", "columns 2-5", "columns 0-1 and 8-9"], "columns 2-5"),
    ("M", "Where is the moving object?", ["left edge", "middle", "right edge"], "middle"),
    ("S", "Which columns hold static stacked blocks? Ignore the moving object.",
     ["well is empty", "columns 2-5", "columns 0-1 and 8-9"], "columns 0-1 and 8-9"),
    ("E", "Which columns hold static stacked blocks? Ignore the moving object.",
     ["well is empty", "columns 2-5", "columns 0-1 and 8-9"], "well is empty"),
]
ok = 0; ms = []
for pname, q, opts, exp in TESTS:
    got, probs, t = ask(STATES[pname], q, opts)
    ok += got == exp; ms.append(t)
    res = "PASS" if got == exp else "FAIL"
    print(f"MOTION {res} {pname} {q[:34]!r} got={got!r} {[round(p, 2) for p in probs]} {t:.0f}ms", flush=True)
print(f"== MOTION: {ok}/4 mean {sum(ms) / len(ms):.0f}ms ==", flush=True)
print(f"xpu-peak {torch.xpu.max_memory_allocated() / 1e9:.2f}GB", flush=True)
