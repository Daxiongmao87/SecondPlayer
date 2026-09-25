"""Capture v4: Dr. Mario gameplay at virus level 12, wiggle for L/R pairs."""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/projects/secondplayer"))
from PIL import Image

from secondplayer.adapters.mesence.adapter import MesenCEAdapter, MesenCEOptions
from secondplayer.config import SessionConfig
from secondplayer.controller import NEUTRAL, ControllerState

OUT = "/data/tmp/vision-cap/drplay"
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


def tap(btns, hold=8, settle=60):
    ad.apply(1, ControllerState.from_buttons(btns))
    ad.wait_frames(hold)
    ad.apply(1, NEUTRAL)
    ad.wait_frames(settle)


ad.start()
try:
    ad.wait_frames(400)
    tap(["START"], settle=150)
    tap(["UP"])
    tap(["UP"])
    tap(["LEFT"])
    tap(["LEFT"])
    tap(["RIGHT"])  # Dr.MARIO quadrant
    tap(["A"], settle=150)  # submenu
    tap(["A"], settle=150)  # 1P options (virus 00)
    for _ in range(6):
        tap(["RIGHT"])  # virus 12
    Image.fromarray(ad.capture()).save(f"{OUT}/options.png")
    tap(["START"], settle=200)  # begin game
    n = 0
    for i in range(30):
        Image.fromarray(ad.capture()).save(f"{OUT}/g{n:03d}.png")
        n += 1
        ad.wait_frames(20)
        # Gentle alternating wiggle for lateral pill motion.
        if i % 3 == 2:
            tap(["LEFT"] if (i // 3) % 2 == 0 else ["RIGHT"], hold=6, settle=14)
    print("saved", n, flush=True)
finally:
    ad.close()
print("done", flush=True)
