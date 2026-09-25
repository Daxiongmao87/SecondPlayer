from __future__ import annotations

import argparse
import time

from ...config import SessionConfig
from ...controller import ControllerState, NEUTRAL
from .adapter import MesenCEAdapter, MesenCEOptions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Headless MesenCE bridge smoke test (no Laya required)")
    parser.add_argument("rom")
    parser.add_argument("--binary", default="Mesen")
    parser.add_argument("--no-xvfb", action="store_true")
    parser.add_argument("--keep-session", action="store_true")
    args = parser.parse_args(argv)

    options = MesenCEOptions(
        binary=args.binary,
        headless=True,
        xvfb=not args.no_xvfb,
        deterministic=True,
        keep_session=args.keep_session,
    )
    session = SessionConfig(players={1: "human", 2: "ai"})
    adapter = MesenCEAdapter(options, session, args.rom)
    try:
        adapter.start()
        frame = adapter.capture()
        print(f"capture: {frame.shape[1]}x{frame.shape[0]} RGB")

        probe = ControllerState.from_buttons(["A", "RIGHT"])
        expected = adapter._mask(probe)
        adapter.apply(2, probe)
        # INPUT is stored on receipt but applied at the next inputPolled, so a
        # single capture round-trip can observe the previous port state. Poll
        # until the injected state is reported or the wait is exhausted.
        deadline = time.monotonic() + 10.0
        observed = None
        while time.monotonic() < deadline:
            adapter.capture()
            observed = adapter._get_input_mask(2)
            if observed == expected:
                break
        print(f"player2 input: expected=0x{expected:03x} observed=0x{observed:03x}")
        print(f"setinput mode: {adapter._input_mode}")
        if observed != expected:
            print("FAIL: MesenCE did not report the injected Player 2 state")
            return 3

        adapter.apply(2, NEUTRAL)
        print("PASS: bridge connected, framebuffer captured, Player 2 input injected")
        if adapter.session_root:
            print(f"session: {adapter.session_root}")
        return 0
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
