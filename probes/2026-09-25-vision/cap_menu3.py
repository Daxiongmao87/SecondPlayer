"""Capture v3: Tetris options/startup, Dr.Mario 1P options + gameplay boot."""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/projects/secondplayer"))
from PIL import Image

from secondplayer.adapters.mesence.adapter import MesenCEAdapter, MesenCEOptions
from secondplayer.config import SessionConfig
from secondplayer.controller import NEUTRAL, ControllerState

OUT = "/data/tmp/vision-cap/menu3"
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


def tap(btns, hold=8, settle=60):
    ad.apply(1, ControllerState.from_buttons(btns))
    ad.wait_frames(hold)
    ad.apply(1, NEUTRAL)
    ad.wait_frames(settle)


def screen(btns):
    tap(btns, settle=150)


def shot(tag):
    N[0] += 1
    name = f"{N[0]:02d}-{tag}"
    Image.fromarray(ad.capture()).save(f"{OUT}/{name}.png")
    print("saved", name, flush=True)


ad.start()
try:
    ad.wait_frames(400)
    screen(["START"])  # title -> game select (cursor: TETRIS)
    shot("gsel-tetris")
    screen(["A"])  # -> tetris submenu
    shot("tsub-row1")
    tap(["DOWN"])
    shot("tsub-row2")
    tap(["DOWN"])
    shot("tsub-row3")
    tap(["UP"])
    tap(["UP"])
    shot("tsub-row1b")
    screen(["A"])  # 1PLAYER -> options?
    shot("topt-0")
    for i, b in enumerate(["DOWN", "DOWN", "DOWN", "DOWN", "UP", "RIGHT", "LEFT", "LEFT"]):
        tap([b])
        shot(f"topt-{i}-{b.lower()}")
    screen(["A"])  # maybe start game
    shot("tgame-0")
    for i in range(4):
        ad.wait_frames(90)
        shot(f"tgame-{i + 1}")
    # Back out to game select (B x3 with long settles).
    screen(["B"])
    shot("back-0")
    screen(["B"])
    shot("back-1")
    screen(["B"])
    shot("gsel-again")
    # Dr.MARIO quadrant: UP/LEFT to corner, RIGHT.
    tap(["UP"])
    tap(["UP"])
    tap(["LEFT"])
    tap(["LEFT"])
    shot("gsel-tetris2")
    tap(["RIGHT"])
    shot("gsel-drmario")
    screen(["A"])  # -> dr submenu (cursor 1PLAYER)
    shot("dsub-row1")
    screen(["A"])  # 1PLAYER -> 1P options
    shot("dopt-0")
    for i, b in enumerate(["RIGHT", "RIGHT", "DOWN", "RIGHT", "DOWN", "RIGHT",
                           "RIGHT", "UP", "UP", "LEFT", "DOWN", "DOWN", "DOWN"]):
        tap([b])
        shot(f"dopt-{i}-{b.lower()}")
    screen(["START"])  # begin Dr. Mario game?
    shot("dgame-0")
    for i in range(6):
        ad.wait_frames(60)
        shot(f"dgame-{i + 1}")
    screen(["B"])
    shot("dback-0")
finally:
    ad.close()
print("done", flush=True)
