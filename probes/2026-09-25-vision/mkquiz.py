"""Vision-quiz ground-truth analyzer (runs locally on /tmp/vcap mirror).

Analyzes motion pairs (Tetris well, Dr. Mario bottle), virus color counts,
and well geometry; dumps candidate montages for human verification and
emits verified candidates for the manifest.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.expanduser("~/projects/secondplayer"))
import numpy as np
from PIL import Image, ImageDraw

VC = "/tmp/vcap"
WELL = (93, 44, 173, 208)      # tetris playfield, native coords
BOTTLE = (146, 50, 202, 214)   # dr.mario bottle, native coords


def load(sub, name):
    return np.asarray(Image.open(f"{VC}/{sub}/{name}.png").convert("RGB"))


def roi(img, box):
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


def clusters(mask):
    """Greedy connected components (4-neigh) over a bool mask; returns list of pixel arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    out = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            pts = []
            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append(np.array(pts))
    return out


def pair_motion(a, b, box, min_px=12):
    """Direction + halves between two frames restricted to box.

    Returns dict or None when ambiguous (lock+spawn distance split, no motion).
    """
    ra, rb = roi(a, box).astype(int), roi(b, box).astype(int)
    d = np.abs(ra - rb).sum(axis=2)
    changed = d > 60
    if changed.sum() < min_px:
        return None
    # Old-position pixels: bright in A, dark in B; new: dark in A, bright in B.
    la = ra.sum(axis=2)
    lb = rb.sum(axis=2)
    old = changed & (la > lb + 30)
    new = changed & (lb > la + 30)
    if old.sum() < 8 or new.sum() < 8:
        return None
    oy, ox = old.nonzero()
    ny, nx = new.nonzero()
    oc = np.array([ox.mean(), oy.mean()])
    nc = np.array([nx.mean(), ny.mean()])
    delta = nc - oc
    if abs(delta).max() < 2:
        return None
    # Lock+spawn guard: far-apart old/new means different objects.
    if abs(delta).max() > 46:
        return None
    adx, ady = abs(delta[0]), abs(delta[1])
    if ady >= adx * 1.5:
        direction = "DOWN" if delta[1] > 0 else "UP"
    elif adx >= ady * 1.5:
        direction = "RIGHT" if delta[0] > 0 else "LEFT"
    else:
        direction = "DIAG"
    w = box[2] - box[0]
    third = "LEFT" if nc[0] < w / 3 else ("RIGHT" if nc[0] > 2 * w / 3 else "CENTER")
    half = "LEFT" if nc[0] < w / 2 else "RIGHT"
    return {"dir": direction, "third": third, "half": half,
            "dx": round(float(delta[0]), 1), "dy": round(float(delta[1]), 1),
            "nc": (round(float(nc[0]), 1), round(float(nc[1]), 1)), "n": int(changed.sum())}


def tetris_candidates():
    fs = sorted(glob.glob(f"{VC}/play/f*.png"))
    out = []
    for i in range(len(fs) - 1):
        a = np.asarray(Image.open(fs[i]).convert("RGB"))
        b = np.asarray(Image.open(fs[i + 1]).convert("RGB"))
        m = pair_motion(a, b, WELL)
        if m and m["dir"] in ("LEFT", "RIGHT", "DOWN"):
            m["a"] = fs[i].split("/")[-1][:-4]
            m["b"] = fs[i + 1].split("/")[-1][:-4]
            out.append(m)
    return out


def dr_candidates():
    fs = sorted(glob.glob(f"{VC}/drplay/g*.png"))
    out = []
    for i in range(len(fs) - 1):
        a = np.asarray(Image.open(fs[i]).convert("RGB"))
        b = np.asarray(Image.open(fs[i + 1]).convert("RGB"))
        m = pair_motion(a, b, BOTTLE)
        if m and m["dir"] in ("LEFT", "RIGHT", "DOWN"):
            m["a"] = fs[i].split("/")[-1][:-4]
            m["b"] = fs[i + 1].split("/")[-1][:-4]
            out.append(m)
    return out


def montage(pairs, sub, out_path, cols=5):
    ims = []
    for p in pairs:
        a = Image.open(f"{VC}/{sub}/{p['a']}.png").convert("RGB")
        b = Image.open(f"{VC}/{sub}/{p['b']}.png").convert("RGB")
        w, h = a.size
        combo = Image.new("RGB", (w * 2 + 4, h + 18), "white")
        combo.paste(a, (0, 18))
        combo.paste(b, (w + 4, 18))
        d = ImageDraw.Draw(combo)
        d.text((4, 3), f"{p['a']}->{p['b']} {p['dir']} {p['third']}/{p['half']} d=({p['dx']},{p['dy']})", fill="black")
        ims.append(combo)
    if not ims:
        print("no pairs for", out_path)
        return
    cw, ch = ims[0].size
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cw * cols, ch * rows), "white")
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * cw, (i // cols) * ch))
    sheet.save(out_path)
    print(f"{out_path}: {len(ims)} pairs")


if __name__ == "__main__":
    t = tetris_candidates()
    print(f"tetris pairs: {len(t)}")
    for p in t:
        print("  T", p["a"], p["b"], p["dir"], p["third"], p["half"], p["dx"], p["dy"], p["n"])
    d = dr_candidates()
    print(f"drm pairs: {len(d)}")
    for p in d:
        print("  D", p["a"], p["b"], p["dir"], p["third"], p["half"], p["dx"], p["dy"], p["n"])
    montage(t, "play", f"{VC}/tpairs.png", cols=4)
    montage(d, "drplay", f"{VC}/dpairs.png", cols=4)
