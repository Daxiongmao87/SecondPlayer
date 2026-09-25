"""Capture v3: scripted Tetris scatter-play from bench0.state for motion pairs."""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/projects/secondplayer"))
from PIL import Image

from secondplayer.adapters.mesence.adapter import MesenCEAdapter, MesenCEOptions
from secondplayer.config import SessionConfig
from secondplayer.controller import NEUTRAL, ControllerState

SP = os.path.expanduser("~/projects/secondplayer")
OUT = "/data/tmp/vision-cap/play"
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
STATE = open(f"{SP}/secondplayer/benchmarks/tetris/bench0.state", "rb").read()


def hold(btns, frames):
    ad.apply(1, ControllerState.from_buttons(btns))
    ad.wait_frames(frames)
    ad.apply(1, NEUTRAL)


ad.start()
try:
    ad.load_state(STATE)
    ad.wait_frames(30)
    n = 0
    # Scatter script: lateral sweeps + spins + soft drops, shot every 10f.
    script = ([("LEFT", 25), (None, 35), ("RIGHT", 25), (None, 35)] * 3
              + [("A", 6), (None, 30), ("B", 6), (None, 30)]
              + [(None, 60), ("DOWN", 12), (None, 40)]
              + [("RIGHT", 40), (None, 30), ("LEFT", 40), (None, 30)] * 2
              + [(None, 200)])
    for btns, frames in script:
        if btns is None:
            ad.wait_frames(frames)
        else:
            hold([btns] if isinstance(btns, str) else list(btns), frames)
        for _ in range(max(1, frames // 10)):
            Image.fromarray(ad.capture()).save(f"{OUT}/f{n:04d}.png")
            n += 1
            ad.wait_frames(10)
    print("saved", n, flush=True)
finally:
    ad.close()
print("done", flush=True)
