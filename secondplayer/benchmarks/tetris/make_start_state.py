"""Regenerate bench0.state: boot -> GAME SELECT detect -> 5x START -> verify -> savestate.

Dev tool (needs the pinned ROM + MesenCE); not used by the benchmark itself,
which loads the committed bench0.state. Usage:
    python -m secondplayer.benchmarks.tetris.make_start_state <rom> <out.state>
"""
import hashlib
import sys
import time

import numpy as np
from PIL import Image

from secondplayer.adapters.mesence.adapter import MesenCEAdapter, MesenCEOptions
from secondplayer.config import SessionConfig
from secondplayer.controller import NEUTRAL, ControllerState

ROM, SAVEPATH = sys.argv[1], sys.argv[2]
START = ControllerState.from_buttons(["start"])

options = MesenCEOptions(
    binary="/home/agent/.local/opt/mesence/Mesen",
    headless=True,
    xvfb=True,
    deterministic=True,
    pulse_frames=9,
    startup_timeout_s=60,
)
ad = MesenCEAdapter(options, SessionConfig(players={1: "ai", 2: "disabled"}), ROM)
tick = 0


def wait(n):
    global tick
    ad.wait_frames(n)
    tick += n


def tap(btn=START, frames=12):
    ad.apply(1, btn)
    wait(frames)
    ad.apply(1, NEUTRAL)
    wait(frames)


def is_gsel(a):
    r = a[100:175, 10:245].astype(int)
    R, G, B = r[:, :, 0], r[:, :, 1], r[:, :, 2]
    black = int(((R + G + B) < 30).sum())
    brown = int(((R >= 60) & (G >= 40) & (B <= 30)).sum())
    gridblue = int(((R < 40) & (G < 100) & (B > 120)).sum())
    return black > 8000 and brown > 2800 and gridblue < 1000


def poll_gsel(rounds, step=30):
    for i in range(rounds):
        a = ad.capture()
        if a.sum() == 0:
            raise RuntimeError("blank capture during poll")
        if is_gsel(a):
            return True
        wait(step)
    return False


t0 = time.time()
ad.start()
try:
    wait(600)  # blind boot
    det = None
    # Boot parks on the animated copyright/press-start screen; one START
    # reaches GAME SELECT. Retry loop covers a swallowed press mid-fade.
    for attempt in range(3):
        if poll_gsel(10):
            det = tick
            break
        tap()
        wait(120)
    if det is None:
        raise RuntimeError("GAME SELECT not detected")
    print(f"gsel detected at tick {det} (+{time.time()-t0:.1f}s wall)", flush=True)
    wait(120)  # let fade finish
    for i in range(5):
        tap()
        wait(300)
        print(f"tap {i+1} done tick={tick}", flush=True)
    # motion verify: falling piece must change pixels over 120 frames
    a1 = ad.capture()
    Image.fromarray(a1).save("/tmp/sp-test/tetris/verify-a.png")
    wait(120)
    a2 = ad.capture()
    Image.fromarray(a2).save("/tmp/sp-test/tetris/verify-b.png")
    diff = int((np.abs(a1.astype(int) - a2.astype(int)).sum(axis=2) > 60).sum())
    print(f"motion diff pixels: {diff}", flush=True)
    if diff < 100:
        raise RuntimeError("no gameplay motion detected (stuck in menus?)")
    s1 = ad.save_state(SAVEPATH)
    h1 = hashlib.sha256(s1).hexdigest()
    print(f"save tick={tick} size={len(s1)} sha={h1[:16]}", flush=True)
    time.sleep(0.5)
    s2 = ad.save_state(SAVEPATH + ".probe")
    h2 = hashlib.sha256(s2).hexdigest()
    print(f"resave size={len(s2)} sha={h2[:16]} rest-stable={h1==h2}", flush=True)
    ad.load_state(s1)
    wait(30)
    s3 = ad.save_state(SAVEPATH + ".probe2")
    print(f"load ok tick={tick} size={len(s3)}", flush=True)
    print(f"PREFIX OK wall={time.time()-t0:.1f}s", flush=True)
finally:
    ad.close()
