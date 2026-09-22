# SecondPlayer

SecondPlayer is a standalone local AI-player runtime that turns controller seats in existing games into visually controlled AI players.

The v1 proof-of-concept targets **stock MesenCE running an SNES game** with **Laya Vision** as the local visual decision model. The emphasis for v1 is implementation cleanliness, not emulator breadth.

## Core architecture

```text
stock MesenCE
    ^   |
    |   | stock Lua API
    |   v
SecondPlayer MesenCE adapter
    ^   |
    |   | local IPC
    |   v
secondplayerd
    |
    +-- Laya Vision
    +-- asynchronous decision scheduler
    +-- player policy/state
    +-- logging/configuration
```

The MesenCE adapter now uses only capabilities exposed by a stock MesenCE build: command-line Lua loading, `emu.takeScreenshot()`, `inputPolled`, and `emu.setInput()`. It does not require a patched emulator executable. The Lua bridge contains only transport/input compatibility logic; all AI remains in the standalone daemon.

## Adapter isolation is a hard requirement

Core SecondPlayer must not contain emulator-specific implementation logic.

The core may know:

- which adapter is selected;
- opaque configuration passed to that adapter;
- which logical player IDs are human, AI or disabled;
- the generic observation/control lifecycle exposed by `EmulatorAdapter`.

Only an adapter may know:

- emulator executable names or command-line behavior;
- emulator config files or profile layout;
- scripting APIs;
- framebuffer/capture implementation;
- controller-port topology and seat limits;
- emulator-specific IPC;
- launch/attach behavior;
- adapter-specific dependencies and health checks.

If supporting an emulator requires adding an emulator name, config key, capture path, controller quirk or platform workaround to core modules, the abstraction has failed and that code belongs in the adapter.

## Configuration boundary

Core configuration selects an adapter but treats its options as opaque:

```toml
[runtime]
model = "thaitea/laya-vision"
device = "cpu"
reaction_ms = 200
frames = 1

[adapter]
name = "mesence"

[adapter.options]
binary = "Mesen"

[session.players]
1 = "human"
2 = "ai"
```

`[adapter.options]` is parsed and validated by the selected adapter, not by `secondplayer.config`.

Likewise, core does not define how many controller seats an emulator supports. MesenCE's SNES adapter can reject unsupported topology itself.

## v1 scope

The clean first proof is intentionally narrow:

- SNES games;
- stock MesenCE;
- Player 1 / Player 2 standard controller ports;
- human + AI, AI + human, and experimental AI + AI;
- pixels only for AI observation;
- local Laya Vision inference;
- asynchronous decisions so emulator timing never waits for inference;
- no external model endpoint.

Players 3–5/multitap are deferred until MesenCE's Lua sub-port behavior is verified with a stock release.

## Snes9x prototype status

The earlier Snes9x implementation was an architectural experiment used to validate the external-adapter concept. It depended on X11/XWayland capture, `/dev/uinput`, SDL slot discovery and an isolated Snes9x configuration profile.

That adapter has been **removed**, not retained as compatibility baggage. Its platform modules, dependencies, tests and configuration schema were removed with it. No Snes9x-specific behavior remains in core SecondPlayer.

The strategic reason is documented in [`docs/EMULATOR_SELECTION.md`](docs/EMULATOR_SELECTION.md).

## MesenCE adapter status

The v1 MesenCE adapter and bundled Lua bridge are implemented. It:

- launches a stock MesenCE binary with the ROM and bridge script;
- copies/patches MesenCE settings only inside a temporary session home;
- preserves the user's normal `settings.json`;
- enables only the Lua I/O/network permissions required for the local bridge;
- forces ordinary SNES controllers on ports 1 and 2;
- captures the rendered console image using MesenCE's in-memory PNG screenshot API;
- injects complete AI controller state from the `inputPolled` callback;
- works in graphical mode or MesenCE `--testrunner` mode under Xvfb;
- contains a runtime compatibility shim for the current MesenCE 2.2.1 `setInput` port-argument bug, without patching MesenCE.

Run unit tests with `pytest -q`. For real headless integration testing, follow [`docs/HEADLESS_TESTING.md`](docs/HEADLESS_TESTING.md).
