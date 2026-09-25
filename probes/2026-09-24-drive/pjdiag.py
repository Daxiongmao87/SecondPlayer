"""PlayJev diagnostic: their frame+their moves vs SNES pixels, options, size."""
import os, sys, time
sys.path.insert(0, "/tmp/playjev-pub")
from PIL import Image, ImageDraw
from playjev.model import PlayJevModel

SP = os.path.expanduser("~/projects/secondplayer/secondplayer/benchmarks/tetris/bench0-reference.png")
base = Image.open(SP).convert("RGB")
THUMB = "/tmp/playjev-pub/docs/assets/thumbs/tetris.png"

MOVES5 = [
    {"name": "left", "description": "shift the falling piece one column to the left"},
    {"name": "right", "description": "shift the falling piece one column to the right"},
    {"name": "rotate", "description": "turn the falling piece a quarter turn clockwise"},
    {"name": "drop", "description": "send the falling piece straight down until it lands"},
    {"name": "none", "description": "leave the piece alone and let it sink one row"},
]
H3 = [
    {"name": "left", "description": "shift the falling piece one column to the left"},
    {"name": "right", "description": "shift the falling piece one column to the right"},
    {"name": "none", "description": "leave the piece alone and let it sink one row"},
]

def paint(im, cols, rows, color=(30, 144, 255)):
    im = im.copy(); d = ImageDraw.Draw(im)
    for c in cols:
        for r in rows:
            x0, y0 = int(93 + c * 8.0), int(44 + r * 8.2)
            d.rectangle([x0, y0, x0 + 7, y0 + 7], fill=color)
    return im

clean = paint(base, [3, 4, 5, 6, 7], [5, 6, 7, 8], color=(61, 55, 0))
h5 = paint(paint(clean, [0, 1, 2, 3, 4, 5, 9] if False else [c for c in range(10) if c not in (7, 8)],
                 [16, 17, 18, 19]), [6, 7], [14, 15], color=(0, 200, 200))
thumb = Image.open(THUMB).convert("RGB")
print("thumb size:", thumb.size, "h5 size:", h5.size, flush=True)

model = PlayJevModel("OmniJev/PlayJev-0.8B", device="xpu", dtype="bfloat16").load()
print("prompt audit:", repr(model.build_prompt(MOVES5, "Which move?", 1, "temporal", None)[:200]), flush=True)

def run(tag, img, opts, long_side=None):
    ts = time.time()
    d = model.decide([img], opts, frames_per_state=1, batch_size=1, long_side=long_side)[0]
    t = (time.time() - ts) * 1000
    dist = " ".join(f"{o['name']}={p:.2f}" for o, p in zip(opts, d.probs))
    print(f"{tag} choice={opts[d.choice]['name']} conf={d.confidence:.2f} mass={d.allowed_mass:.3f} "
          f"top={d.top_token} vtoks={model.last_timing.visual_tokens} {t:.0f}ms :: {dist}", flush=True)

run("A-thumb-theirs5", thumb, MOVES5)
run("B-h5-theirs5", h5, MOVES5)
run("C-h5-factor3", h5, H3)
run("D-h5-factor3-ls512", h5, H3, long_side=512)
run("E-thumb-factor3", thumb, H3)
