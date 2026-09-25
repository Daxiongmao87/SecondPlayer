"""Decisive probe: TEXT vs COORD vs ASCII vs VISION, all with thinking DISABLED."""
import os, time
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
PAL = {"B": (30, 144, 255), "C": (0, 255, 255), "M": (255, 0, 255), "Y": (255, 255, 0),
       "R": (255, 0, 0), "G": (0, 255, 0), "W": (255, 255, 255), "O": (255, 165, 0)}

def cell_rgb(im, c, r):
    x0, y0 = int(OX + c * CW), int(OY + r * CH)
    return im.crop((x0 + 2, y0 + 2, x0 + 6, y0 + 6)).resize((1, 1), Image.BILINEAR).getpixel((0, 0))

def grid_cells(im):
    cells = [[cell_rgb(im, c, r) for c in range(10)] for r in range(20)]
    flat = [p for row in cells for p in row]
    bg = max(set(flat), key=flat.count)
    return cells, bg

def is_fg(p, bg):
    return sum((a - b) ** 2 for a, b in zip(p, bg)) ** 0.5 >= 60

def ascii_text(im):
    cells, bg = grid_cells(im)
    def glyph(p):
        if not is_fg(p, bg):
            return "."
        return min(PAL, key=lambda k: sum((a - c) ** 2 for a, c in zip(p, PAL[k])))
    return ("".join(str(c % 10) for c in range(10)) + "\n" +
            "\n".join("".join(glyph(p) for p in row) for row in cells))

def coord_text(im):
    cells, bg = grid_cells(im)
    filled = [(r, c) for r in range(20) for c in range(10) if is_fg(cells[r][c], bg)]
    return ("Filled cells as (row,col), row 0 is top, col 0 is left: " +
            ("none" if not filled else " ".join(f"({r},{c})" for r, c in filled)))

ASCII = {k: ascii_text(v) for k, v in frames.items()}
COORD = {k: coord_text(v) for k, v in frames.items()}

tok = transformers.AutoTokenizer.from_pretrained(QWEN, trust_remote_code=False)
cfg = transformers.AutoConfig.from_pretrained(QWEN, trust_remote_code=False)
qm = transformers.AutoModelForImageTextToText.from_pretrained(
    QWEN, dtype=torch.bfloat16, device_map={"": "cpu"}, trust_remote_code=False)
from quanto import freeze, qint4, quantize
quantize(qm, weights=qint4, activations=None); freeze(qm)
qm.to("xpu"); qm.eval(); torch.xpu.synchronize()

SYS = "You are a decision readout. Read the state and the question, then answer with exactly one letter and nothing else."
L = [tok.convert_tokens_to_ids(c) for c in "ABC"]
NPIX = 256 * 224
IP = transformers.Qwen2VLImageProcessor(patch_size=16, min_pixels=NPIX, max_pixels=NPIX * 4)

def template(msgs):
    try:
        t = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                    enable_thinking=False)
    except TypeError:
        t = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return t

# verify thinking is really off
_probe = template([{"role": "system", "content": SYS}, {"role": "user", "content": "hi"}])
print("think-tag-in-template:", ("<think>" in _probe.lower()) or ("thinking" in _probe.lower()), flush=True)

def ask(question, options, image=None, state_text=None):
    if state_text is not None:
        user = ("Game: tetris.\n\nState:\n" + state_text + "\n\nQuestion: " + question +
                "\n\nOptions:\n" + "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options)) +
                "\n\nAnswer with one letter.")
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
        ids = tok(template(msgs), add_special_tokens=False).input_ids
        kw = {"input_ids": torch.tensor([ids], dtype=torch.long).to("xpu")}
    elif image is not None:
        user = (f"Game: tetris.\n\nQuestion: {question}\n\nOptions:\n" +
                "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options)) +
                "\n\nAnswer with one letter.")
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": [{"type": "text", "text": user}, {"type": "image"}]}]
        ids = tok(template(msgs), add_special_tokens=False).input_ids
        pix = IP(images=image, return_tensors="pt")
        thw = pix["image_grid_thw"][0].tolist()
        nimg = (thw[0] * thw[1] * thw[2]) // (cfg.vision_config.spatial_merge_size ** 2)
        full, mm = [], []
        for t in ids:
            if t == cfg.image_token_id:
                full += [t] * nimg; mm += [1] * nimg
            else:
                full += [t]; mm += [0]
        kw = {"input_ids": torch.tensor([full], dtype=torch.long).to("xpu"),
              "mm_token_type_ids": torch.tensor([mm], dtype=torch.int).to("xpu"),
              "pixel_values": pix["pixel_values"].to(torch.bfloat16).to("xpu"),
              "image_grid_thw": pix["image_grid_thw"].to("xpu")}
    else:
        user = (f"Game: tetris.\n\nQuestion: {question}\n\nOptions:\n" +
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
    ("orig", "Where is the falling piece?", ["left edge", "middle", "right edge"], "middle"),
    ("orig", "Which columns hold stacked blocks?", ["well is empty", "columns 2-5", "columns 0-1 and 8-9"], "well is empty"),
    ("synthA", "Where is the falling piece?", ["left edge", "middle", "right edge"], "middle"),
    ("synthA", "Which columns hold stacked blocks?", ["well is empty", "columns 2-5", "columns 0-1 and 8-9"], "columns 2-5"),
    ("synthB", "Where is the falling piece?", ["left edge", "middle", "right edge"], "middle"),
    ("synthB", "Which columns hold stacked blocks?", ["well is empty", "columns 2-5", "columns 0-1 and 8-9"], "columns 0-1 and 8-9"),
]
STATES = {"COORD": COORD, "ASCII": ASCII}
for label in ("TEXT", "COORD", "ASCII", "IMG"):
    ok = 0; ms = []
    for fname, q, opts, exp in TESTS:
        if label in STATES:
            got, probs, t = ask(q, opts, state_text=STATES[label][fname])
        elif label == "IMG":
            got, probs, t = ask(q, opts, image=frames[fname])
        else:
            got, probs, t = ask(q, opts)
        ok += got == exp; ms.append(t)
        res = "PASS" if got == exp else "FAIL"
        print(f"{label} {res} {fname} {q[:20]!r} got={got!r} {[round(p, 2) for p in probs]} {t:.0f}ms", flush=True)
    print(f"== {label}: {ok}/6 mean {sum(ms) / len(ms):.0f}ms ==", flush=True)
print(f"xpu-peak {torch.xpu.max_memory_allocated() / 1e9:.2f}GB", flush=True)
