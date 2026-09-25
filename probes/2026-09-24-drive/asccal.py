"""Calibrate: blue-profile per x-column over the well band, all 3 frames."""
import os
from PIL import Image

SP = os.path.expanduser("~/projects/secondplayer/secondplayer/benchmarks/tetris/bench0-reference.png")
frames = {"orig": Image.open(SP).convert("RGB"),
          "synthA": Image.open("/tmp/synthA2.png").convert("RGB"),
          "synthB": Image.open("/tmp/synthB2.png").convert("RGB")}

def is_blue(px):
    r, g, b = px
    return b > 120 and b > r + 30 and b > g + 10

for name, im in frames.items():
    px = im.load()
    prof = []
    for x in range(80, 184):
        n = sum(1 for y in range(44, 208) if is_blue(px[x, y]))
        prof.append("#" if n > 100 else ("+" if n > 20 else ("." if n > 0 else " ")))
    print(name + " x=80..183:")
    print("".join(prof))
    print("0123456789" * 10 + "0123")
    # vertical extent of blue at a painted column
    colx = {"orig": 132, "synthA": 124, "synthB": 92}[name]
    ys = [y for y in range(30, 220) if is_blue(px[colx, y])]
    print(f"blue-y at x={colx}: {min(ys) if ys else None}..{max(ys) if ys else None} n={len(ys)}")
    print("interior bg sample (100,120): %02x%02x%02x" % px[100, 120])
    print("wall sample (90,120): %02x%02x%02x" % px[90, 120])
