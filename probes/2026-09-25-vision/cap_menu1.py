"""Capture v1: cold boot, title, and blind menu exploration frames."""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/projects/secondplayer"))
from PIL import Image

from secondplayer.adapters.mesence.adapter import MesenCEAdapter, MesenCEOptions
from secondplayer.config import SessionConfig
from secondplayer.controller import NEUTRAL, ControllerState

OUT = "/data/tmp/vision-cap/menu1"
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


def tap(btns, hold=8, settle=50):
    ad.apply(1, ControllerState.from_buttons(btns))
    ad.wait_frames(hold)
    ad.apply(1, NEUTRAL)
    ad.wait_frames(settle)


def shot(name):
    Image.fromarray(ad.capture()).save(f"{OUT}/{name}.png")
    print("saved", name, flush=True)


ad.start()
try:
    ad.wait_frames(150)
    shot("00-boot")
    for i in range(4):
        ad.wait_frames(60)
        shot(f"01-title{i}")
    tap(["START"])
    shot("02-after-start")
    for i, btn in enumerate(["DOWN", "DOWN", "DOWN", "UP", "UP", "RIGHT", "LEFT"]):
        tap([btn])
        shot(f"03-nav{i}-{btn.lower()}")
    tap(["A"])
    shot("04-after-A")
    for i, btn in enumerate(["DOWN", "DOWN", "UP", "RIGHT", "RIGHT"]):
        tap([btn])
        shot(f"05-nav{i}-{btn.lower()}")
    tap(["B"])
    shot("06-after-B")
    tap(["START"])
    shot("07-after-start2")
finally:
    ad.close()
print("done", flush=True)
