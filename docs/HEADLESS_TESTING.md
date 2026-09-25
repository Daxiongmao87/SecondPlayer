# Headless Linux Testing — MesenCE v1 Adapter

This document is written for an implementation/testing agent. It tests the real stock-emulator boundary before involving Laya, then tests the full SecondPlayer loop.

## What is being tested

The v1 MesenCE adapter uses only stock MesenCE facilities:

- command-line ROM + Lua script loading;
- `--testrunner` for headless execution;
- `emu.takeScreenshot()` for a PNG of the rendered console image;
- `emu.setInput()` from `inputPolled` for SNES controller injection;
- LuaSocket on localhost for IPC with `secondplayerd`.

No MesenCE source modification, binary injection, virtual controller or desktop capture is part of the adapter.

The bridge contains a runtime compatibility probe for the MesenCE 2.2.1/current `LuaApi::SetInput` stack bug. Do **not** patch MesenCE for the two-controller v1 test. The bridge detects the affected calling convention and addresses Player 2 through the compatible stock call shape.

## Host requirements

Use Linux x86-64 or ARM64 with:

```bash
python3 --version        # 3.11+
which xvfb-run
which Mesen              # or record the absolute MesenCE path
```

MesenCE's Linux release requires SDL2. Install the normal distribution packages required by the MesenCE release plus `xvfb`.

Example Debian/Ubuntu packages:

```bash
sudo apt-get update
sudo apt-get install -y xvfb libsdl2-2.0-0
```

Use a legally obtained SNES ROM (`.sfc` or `.smc`) that can reach active gameplay without unusual peripherals. Do not commit ROMs to this repository.

## 1. Install SecondPlayer development checkout

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
pytest -q
```

The unit suite must pass before integration testing.

## 2. Run adapter doctor

Create a temporary configuration (or edit a copy of `config/example.toml`):

```toml
[adapter]
name = "mesence"

[adapter.options]
binary = "/absolute/path/to/Mesen"
headless = true
xvfb = true
keep_session = true

[session.players]
1 = "human"
2 = "ai"
```

Then:

```bash
secondplayer --config /tmp/secondplayer-test.toml doctor
```

Expected adapter checks:

- MesenCE executable found;
- v1 seat topology accepted (players 1-2 only);
- `xvfb-run` found;
- source config either detected or explicitly reported as absent/minimal.

## 3. Bridge-only smoke test — no Laya/model download

This is the first real integration gate. It launches stock MesenCE under Xvfb in `--testrunner` mode, captures one real rendered frame, injects `RIGHT+A` on SNES Player 2, and asks MesenCE to report the resulting port state.

```bash
python -m secondplayer.adapters.mesence.smoke \
  --binary /absolute/path/to/Mesen \
  --keep-session \
  /absolute/path/to/game.sfc
```

Expected terminal output resembles:

```text
capture: 256x224 RGB
player2 input: expected=0x0.. observed=0x0..
PASS: bridge connected, framebuffer captured, Player 2 input injected
session: /tmp/secondplayer-mesence-...
```

The exact resolution may vary by SNES mode/scene. A matching input mask is mandatory.

If it fails, inspect the retained session:

```bash
cat /tmp/secondplayer-mesence-*/mesen.stdout.log
cat /tmp/secondplayer-mesence-*/mesen.stderr.log
cat /tmp/secondplayer-mesence-*/home/.config/MesenCE/settings.json
```

Do not work around a failure by editing MesenCE source. Fix the adapter unless the stock emulator truly lacks the required capability.

## 4. Verify configuration isolation

Before the test:

```bash
sha256sum ~/.config/MesenCE/settings.json 2>/dev/null || true
```

Run the smoke test, then repeat the checksum. It must be unchanged.

The adapter may **read** the existing settings to preserve user mappings in an isolated copy. It writes only beneath its temporary session root.

## 5. Full local-Laya headless test

Once bridge-only smoke passes, use a config like:

```toml
[runtime]
model = "thaitea/laya-vision"
device = "cpu"
reaction_ms = 200
permutations = 1
poll_ms = 100
window_frames = 6
sample = false
offline = false
objective = "Play the game and make useful progress."

[adapter]
name = "mesence"

[adapter.options]
binary = "/absolute/path/to/Mesen"
headless = true
xvfb = true
keep_session = true
deterministic = true

[session.players]
1 = "human"
2 = "ai"
```

Run:

```bash
secondplayer --config /tmp/secondplayer-test.toml -v play /absolute/path/to/game.sfc
```

On a headless server there is no human Player 1 interaction; P1 being `human` means SecondPlayer leaves that port untouched. For autonomous headless exercise, set both seats to AI:

```toml
[session.players]
1 = "ai"
2 = "ai"
```

Success criteria for the POC:

1. stock MesenCE launches without a configuration wizard;
2. the Lua bridge connects without manual interaction;
3. repeated captures decode as `uint8 HxWx3` RGB frames;
4. Laya inference runs asynchronously while MesenCE continues emulating;
5. AI input appears on the intended SNES port;
6. closing SecondPlayer terminates the test MesenCE process and neutralizes AI state;
7. the user's normal MesenCE settings remain byte-for-byte unchanged.

## 6. Graphical parity test

After headless passes, run the same config with:

```toml
[adapter.options]
headless = false
```

Then run `secondplayer ... play ...` from a desktop session. Current stock MesenCE opens command-line Lua scripts in its Script Window and then returns focus to the main emulator window. That extra script window is a known v1 UX wart; it is not justification for modifying MesenCE.

Confirm that a real human can control the human-owned seat while SecondPlayer exclusively controls the AI seat.

## Agent reporting requirements

When reporting integration results, include:

- MesenCE version/commit and binary path;
- Linux distribution and architecture;
- headless vs graphical;
- ROM console/region (do not upload the ROM);
- reported `MODE fixed` or `MODE legacy-2.2.1` if visible in verbose logs;
- first capture dimensions;
- bridge-only Player 2 expected/observed input masks;
- Laya device and measured inference latency;
- any Mesen stderr output;
- whether the pre/post MesenCE settings checksum matched.
