"""PlayJev factorized quiz v2: long_side=448, verbatim hook descriptions."""
import os, sys, time
sys.path.insert(0, "/tmp/playjev-pub")
from PIL import Image, ImageDraw
from playjev.model import PlayJevModel

SP = os.path.expanduser("~/projects/secondplayer/secondplayer/benchmarks/tetris/bench0-reference.png")
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
def scenario(stack_cols, stack_rows, sq_cols, sq_rows_now):
    cur = paint(paint(clean(base), stack_cols, stack_rows), sq_cols, sq_rows_now, **SQ)
    prv = paint(paint(clean(base), stack_cols, stack_rows), sq_cols,
                [r - 1 for r in sq_rows_now], **SQ)
    return prv, cur

ALL = list(range(10))
scen = {
    "H5": (scenario([c for c in ALL if c not in (7, 8)], [16, 17, 18, 19], [6, 7], [14, 15]),
           "right", "none"),
    "H6": (scenario([c for c in ALL if c not in (1, 2)], [16, 17, 18, 19], [2, 3], [14, 15]),
           "left", "none"),
    "H7": (scenario([c for c in ALL if c not in (8, 9)], [15, 16, 17, 18, 19], [7, 8], [13, 14]),
           "right", "none"),
    "H8": (scenario([c for c in ALL if c not in (0, 1)], [15, 16, 17, 18, 19], [1, 2], [13, 14]),
           "left", "none"),
    "A1": (scenario([c for c in ALL if c not in (6, 7)], [16, 17, 18, 19], [6, 7], [14, 15]),
           "none", "drop"),
}
H = lambda n, d: {"name": n, "description": d}
OPTS_H = [H("left", "shift the falling piece one column to the left"),
          H("right", "shift the falling piece one column to the right"),
          H("none", "leave the piece alone and let it sink one row")]
OPTS_D = [H("drop", "send the falling piece straight down until it lands"),
          H("none", "leave the piece alone and let it sink one row")]

model = PlayJevModel("OmniJev/PlayJev-0.8B", device="xpu", dtype="bfloat16").load()
for fps in (1, 2):
    ok = tot = 0
    for name, ((prv, cur), exp_h, exp_d) in scen.items():
        frames = [cur] if fps == 1 else [(prv, cur)]
        for tag, opts, exp in (("H", OPTS_H, exp_h), ("D", OPTS_D, exp_d)):
            ts = time.time()
            dec = model.decide(frames, opts, frames_per_state=fps, batch_size=1,
                               long_side=448)[0]
            t = (time.time() - ts) * 1000
            got = opts[dec.choice]["name"]
            ok += got == exp; tot += 1
            dist = " ".join(f"{o['name']}={p:.2f}" for o, p in zip(opts, dec.probs))
            print(f"PJ{fps} {'PASS' if got==exp else 'FAIL'} {name}{tag} exp={exp} got={got} "
                  f"conf={dec.confidence:.2f} mass={dec.allowed_mass:.3f} top={dec.top_token} "
                  f"{t:.0f}ms :: {dist}", flush=True)
    print(f"== fps={fps}: {ok}/{tot} (chance ~4.2, gate 8) ==", flush=True)
