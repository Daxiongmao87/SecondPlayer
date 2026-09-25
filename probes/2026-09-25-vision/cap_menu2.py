"""Capture v2: every GAME SELECT quadrant + every submenu, shot per input."""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/projects/secondplayer"))
from PIL import Image

from secondplayer.adapters.mesence.adapter import MesenCEAdapter, MesenCEOptions
from secondplayer.config import SessionConfig
from secondplayer.controller import NEUTRAL, ControllerState

OUT = "/data/tmp/vision-cap/menu2"
os.makedirs(OUT, exist_ok=True)
opts = MesenCEOptions(
    binary="/home/agent/.local/opt/mesence/Mesen",
    headless=False,
    xvfb=True,
    startup_timeout_s=60,
    capture_timeout_s=10,
)
sess = SessionConfig()
sess.players = {1: "ai", 2: "disabled"}
ad = MesenCEAdapter(opts, sess, "/data/tmp/roms/tetris_dr_mario.smc")
N = [0]


def tap(btns, hold=8, settle=50):
    ad.apply(1, ControllerState.from_buttons(btns))
    ad.wait_frames(hold)
    ad.apply(1, NEUTRAL)
    ad.wait_frames(settle)


def shot(tag):
    N[0] += 1
    name = f"{N[0]:02d}-{tag}"
    Image.fromarray(ad.capture()).save(f"{OUT}/{name}.png")
    print("saved", name, flush=True)


ad.start()
try:
    ad.wait_frames(400)  # past copyright to title
    shot("title")
    tap(["START"])
    shot("gsel-init")
    # Walk the 2x2: find all four highlights.
    for tag, btn in [("d1", "DOWN"), ("d2", "DOWN"), ("u1", "UP"), ("u2", "UP"),
                     ("r1", "RIGHT"), ("d3", "DOWN"), ("l1", "LEFT"), ("u3", "UP"),
                     ("l2", "LEFT"), ("d4", "DOWN"), ("r2", "RIGHT")]:
        tap([btn])
        shot(f"gsel-{tag}-{btn.lower()}")
    # Enter Dr.MARIO: from wherever, go top-right then A. Reset via re-walk:
    # go UP+LEFT corner first (TETRIS), RIGHT (Dr.MARIO), A.
    tap(["UP"])
    tap(["UP"])
    tap(["LEFT"])
    tap(["LEFT"])
    shot("gsel-tetris")
    tap(["RIGHT"])
    shot("gsel-drmario")
    tap(["A"])
    shot("drm-sub0")
    for tag, btn in [("d1", "DOWN"), ("d2", "DOWN"), ("d3", "DOWN"), ("u1", "UP"),
                     ("l1", "LEFT"), ("l2", "LEFT"), ("r1", "RIGHT"), ("r2", "RIGHT")]:
        tap([btn])
        shot(f"drm-{tag}-{btn.lower()}")
    tap(["A"])
    shot("drm-after-A")
    tap(["B"])
    shot("drm-after-B")
    tap(["B"])
    shot("gsel-back1")
    # MIXED MATCH submenu: DOWN from Dr.MARIO row? try DOWN then A.
    tap(["DOWN"])
    shot("gsel-mixed?")
    tap(["A"])
    shot("mix-sub0")
    for tag, btn in [("d1", "DOWN"), ("d2", "DOWN"), ("u1", "UP"),
                     ("l1", "LEFT"), ("r1", "RIGHT")]:
        tap([btn])
        shot(f"mix-{tag}-{btn.lower()}")
    tap(["B"])
    shot("gsel-back2")
finally:
    ad.close()
print("done", flush=True)
