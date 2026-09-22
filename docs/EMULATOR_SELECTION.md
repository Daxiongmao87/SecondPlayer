# v1 Emulator Selection

## Decision

**MesenCE is the v1 reference emulator.**

The decision optimizes for the cleanest proof of SecondPlayer's central idea rather than maximum emulator coverage.

## Why the original Snes9x choice changed

Snes9x was selected before the integration constraints were fully defined. Once the requirements became:

- stock emulator binary;
- no emulator source modification;
- all SecondPlayer-specific behavior isolated in an adapter;
- user configuration owned by SecondPlayer rather than persistent emulator edits;
- local Laya Vision;
- clean pixels-to-controller proof before pursuing universality;

stock Snes9x required more infrastructure than desirable for a first POC.

The prototype needed:

- X11/XWayland window discovery and capture;
- OpenGL child-window assumptions;
- `/dev/uinput` virtual controllers;
- SDL joystick-slot probing;
- generated Snes9x controller bindings;
- Linux-specific permissions/install setup.

None of those mechanisms were conceptually wrong, but together they made the adapter prove too many unrelated systems before it could prove SecondPlayer itself.

The entire prototype adapter and its supporting platform code have therefore been removed.

## Why MesenCE is preferred

MesenCE exposes the two seams SecondPlayer fundamentally needs through its stock scripting environment:

1. the emulated video framebuffer;
2. controller input injection synchronized to emulator input polling.

That permits a very small bundled Lua bridge with local IPC to the standalone daemon:

```text
MesenCE framebuffer
      |
      v
bridge.lua ---- local IPC ---- secondplayerd ---- Laya Vision
      ^                                      |
      |                                      |
      +---------- controller state ----------+
```

The emulator-specific component can remain almost entirely a transport adapter.

## Why not RetroArch first

RetroArch remains strategically attractive because RetroPad and libretro could make one adapter useful across many cores.

That is a universality optimization. It is not needed to prove the v1 hypothesis, and it adds frontend/core configuration and capture questions that MesenCE can avoid through its scripting API.

RetroArch should be reconsidered after the clean MesenCE POC works.

## Initial topology

The v1 proof targets ordinary SNES Player 1 and Player 2 controller ports only.

Expected configurations:

- P1 human / P2 AI;
- P1 AI / P2 human;
- P1 AI / P2 AI for experimental autonomous play.

Multitap/Players 3–5 are deferred until stock MesenCE Lua `subPort` behavior is validated. The core does not encode this limit; the MesenCE adapter will.

## Success criterion

The reference POC is successful when:

> An unmodified stock MesenCE build runs a normal SNES game, a human can retain one controller seat, SecondPlayer controls the other seat entirely from the rendered game pixels through local Laya Vision, and neither the emulator nor SecondPlayer core contains special modifications for the other component.

## Implemented v1 integration

The decision has now been implemented. The adapter uses stock MesenCE command-line script loading for graphical sessions and stock `--testrunner` for headless sessions. A bundled Lua bridge communicates only over localhost, captures the rendered console image with `emu.takeScreenshot()`, and applies AI-owned controller state from `inputPolled`.

MesenCE 2.2.1/current source contains a `LuaApi::SetInput` argument-stack defect that affects explicit port/subport selection. For the two-port v1 case, SecondPlayer detects the stock calling convention at runtime and adapts its Lua invocation; no MesenCE patch is required. Multitap remains deferred.

Real-emulator validation steps are in `HEADLESS_TESTING.md`.
