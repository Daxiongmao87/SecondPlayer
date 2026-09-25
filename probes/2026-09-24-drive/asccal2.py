"""Calibrate 2: exact blue bboxes, sizes, and suspect-cell means."""
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
    print(name, "size", im.size)
    px = im.load()
    xs = [x for x in range(im.size[0]) for y in range(im.size[1]) if is_blue(px[x, y])]
    ys = [y for y in range(im.size[1]) for x in range(im.size[0]) if is_blue(px[x, y])]
    print(f"  bluecount={len(xs)} xbbox={min(xs)}..{max(xs)} ybbox={min(ys)}..{max(ys)}")

# suspect cells: synthA ascii cols 2 and 6 rows 17-19; synthB ascii col 8 rows 18-19
imA = frames["synthA"]
for c in (2, 3, 6):
    x0 = int(88 + c * 8.0)
    m = imA.crop((x0, 183, x0 + 8, 208)).resize((1, 1), Image.BILINEAR).getpixel((0, 0))
    print(f"synthA col{c} x[{x0},{x0+8}] y[183,208] mean=%02x%02x%02x" % m)
imB = frames["synthB"]
for c in (8, 9):
    x0 = int(88 + c * 8.0)
    m = imB.crop((x0, 191, x0 + 8, 208)).resize((1, 1), Image.BILINEAR).getpixel((0, 0))
    print(f"synthB col{c} x[{x0},{x0+8}] y[191,208] mean=%02x%02x%02x" % m)
# column histogram of blue over paint band, synthA
px = imA.load()
hist = {}
for x in range(96, 145):
    hist[x] = sum(1 for y in range(183, 208) if is_blue(px[x, y]))
print("synthA bluecounts x=96..144:")
print(" ".join(f"{x}:{hist[x]}" for x in range(96, 145)))
