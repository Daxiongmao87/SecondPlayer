# SecondPlayer v1 Specification

Status: MesenCE adapter implemented; awaiting real-emulator integration validation
Reference emulator: **stock MesenCE**  
Reference system: **SNES**  
Policy: **local Laya Vision**  
Primary goal: **prove AI-controlled multiplayer seats with the cleanest possible standalone integration**

## 1. Product definition

SecondPlayer is a standalone local AI-player runtime. It observes a game's rendered pixels and produces ordinary controller decisions for one or more logical player seats.

The emulator remains an independent stock application. SecondPlayer owns the AI runtime, scheduling, policy, configuration and logging. Emulator-specific integration is delegated to a replaceable adapter.

The v1 name is conceptual: SecondPlayer represents the missing human player role, not literally controller index 2.

## 2. v1 proof

The first proof targets a two-player SNES game in stock MesenCE.

Required v1 seat modes:

- Player 1 human / Player 2 AI;
- Player 1 AI / Player 2 human;
- Player 1 AI / Player 2 AI as an experimental mode.

The first release does not need to solve universal emulator support, multitap, remote inference, hidden game-state parsing or per-game training.

## 3. Non-negotiable architectural constraints

### 3.1 Standalone runtime

Laya/model execution must not run inside MesenCE.

`secondplayerd` owns:

- model loading;
- inference;
- decision scheduling;
- per-player policy state;
- reaction timing;
- logs/metrics;
- user configuration.

### 3.2 Stock emulator

SecondPlayer must not require MesenCE source changes, a custom fork or a custom emulator build.

A bundled Lua bridge loaded through MesenCE's normal scripting facilities is allowed and preferred.

### 3.3 Adapter isolation

All MesenCE-specific behavior belongs under the MesenCE adapter.

Core must not know:

- MesenCE executable names;
- MesenCE config paths;
- Lua APIs;
- MesenCE controller port/subport rules;
- MesenCE launch flags;
- bridge protocol details;
- MesenCE version quirks.

The same rule applies to every future emulator.

See `ADAPTER_CONTRACT.md`.

### 3.4 Visual-only AI observation

The AI receives the rendered framebuffer and optional textual session context. It must not receive emulator RAM, entity coordinates, internal physics state, memory-watch values or other information hidden from a human player.

Using MesenCE's framebuffer API is acceptable because it exposes the rendered game image directly.

### 3.5 Local-only model execution

v1 uses Laya Vision locally. There is no external inference endpoint and no cloud fallback.

If inference is slow, controller state remains asynchronous; the emulator is not blocked waiting for the model.

## 4. System architecture

```text
+-------------------------------------------------------+
|                    stock MesenCE                      |
|                                                       |
|   SNES core/game                                      |
|       |                                               |
|       +--> rendered framebuffer                       |
|       |        |                                      |
|       |        v                                      |
|       |   bridge.lua                            |
|       |        |                                      |
|       |        | local IPC                            |
|       |        v                                      |
|       |   secondplayerd                               |
|       |        |                                      |
|       |        +--> Laya Vision                       |
|       |        +--> player policy/state               |
|       |        +--> async scheduler                   |
|       |        |                                      |
|       |        +--> latest controller state ----------+
|       |                                               |
|       +<-- inputPolled / setInput --------------------+
+-------------------------------------------------------+
```

The Lua bridge is part of the adapter. It is intentionally thin and contains no AI behavior.

## 5. Adapter responsibilities

The MesenCE adapter owns:

- locating/launching a stock MesenCE executable;
- creating any temporary isolated session/profile needed for SecondPlayer;
- arranging automatic loading of the bundled Lua bridge using stock facilities;
- establishing local bridge IPC;
- receiving framebuffer observations from the bridge;
- mapping logical player IDs to MesenCE SNES controller ports;
- applying current controller state at the correct input polling point;
- validating that requested seat topology is supported;
- reporting emulator/game lifecycle to the core;
- cleanup.

No MesenCE responsibility belongs in `config.py`, `engine.py`, the Laya policy, or other generic modules.

## 6. Core adapter interface

The current minimal interface is:

```python
class EmulatorAdapter:
    @property
    def game_name(self) -> str: ...

    def start(self) -> None: ...
    def is_running(self) -> bool: ...
    def capture(self) -> np.ndarray: ...
    def apply(self, player: int, state: ControllerState) -> None: ...
    def close(self) -> None: ...
```

`capture()` returns a `uint8` HWC RGB frame.

`apply()` updates the latest controller state for one logical player. The adapter decides how and when that state reaches the emulator.

The interface should only expand when a demonstrated cross-adapter requirement appears.

## 7. Configuration architecture

Core configuration contains one selected adapter and an opaque adapter option table:

```toml
[runtime]
model = "thaitea/laya-vision"
device = "cpu"
reaction_ms = 200
permutations = 1
poll_ms = 100
window_frames = 6
sample = false

[adapter]
name = "mesence"

[adapter.options]
binary = "Mesen"

[session.players]
1 = "human"
2 = "ai"
```

Core parses `adapter.name` but does not interpret `adapter.options`.

The MesenCE adapter defines and validates its own options in `secondplayer.adapters.mesence`.

Likewise, core only requires player IDs to be positive integers. It does not encode a maximum controller count. Adapter topology validation is adapter-owned.

## 8. MesenCE Lua bridge

The implemented bridge is intentionally small and contains no AI behavior. Its responsibilities are:

```text
on/frame event:
    obtain current rendered framebuffer
    publish newest observation to secondplayerd

on inputPolled:
    read latest non-blocking controller state from bridge transport
    apply it to AI-owned controller port(s)
```

The bridge must never call Laya, load ML libraries or implement gameplay policy.

IPC must remain local to the machine. Loopback TCP is acceptable for the POC; Unix-domain sockets or another local transport may replace it later if useful.

The bridge should tolerate the daemon temporarily being unavailable without crashing MesenCE.

## 9. Frame semantics

Preferred observation is the actual emulator framebuffer before desktop scaling/shaders/window chrome.

The loop polls the framebuffer at `poll_ms` and runs inference only when
the scene changed (see `PollGate`; 1s heartbeat on frozen screens). The
policy state carries a sliding window of up to `window_frames` recent
native-resolution frames plus their control choices, trimmed per decision
to fit the checkpoint's context window (see `fit_window`).

For v1 the adapter uses `emu.takeScreenshot()`, which stock MesenCE explicitly returns as an in-memory PNG binary string. This avoids desktop capture and avoids iterating the entire Lua `getScreenBuffer()` table solely to repack pixels into a socket payload. Python decodes the PNG into the core's RGB `numpy.ndarray` contract.

A future benchmark may replace PNG transport with packed raw pixels if it materially improves end-to-end latency; that remains an adapter-only optimization.

## 10. Controller semantics

The current v1 controller domain is SNES:

```text
UP DOWN LEFT RIGHT
A B X Y
L R
START SELECT
```

Controller legality is console-level policy, not MesenCE-specific behavior. Opposing physical D-pad directions are sanitized.

Laya currently chooses movement and non-directional action separately to avoid illegal/random twelve-button combinations.

This SNES policy may later move behind a console/controller schema layer when SecondPlayer expands beyond SNES. It must not be confused with an emulator adapter responsibility.

## 11. Laya Vision policy

For every inference cycle, all due AI players share the same framebuffer observation.

Questions are generated independently per logical player, while Laya can reuse a single encoded image across those questions.

Current policy state includes:

- player number;
- game name;
- user/global objective;
- previous controller state;
- optional previous decision frame.

No emulator-specific MesenCE data may be included.

## 12. Timing model

MesenCE continues emulating independently of Laya inference.

SecondPlayer uses an asynchronous worker:

```text
capture latest frame
      |
      +--> inference in worker

emulator keeps running
latest controller state remains active

worker completes
      |
      v
replace latest controller state
```

`reaction_ms` is a minimum submission cadence, not a guarantee that inference completes within that period.

Current published Laya CPU behavior is not assumed to meet human reaction latency. CPU remains a desired deployment target, but actual latency must be measured during integration testing rather than asserted from model size alone.

## 13. Error behavior

On policy/inference failure:

- count/log the failure;
- release AI controller state to neutral;
- keep the emulator independent where safe.

On bridge/adapter failure:

- adapter reports failure to the core;
- AI inputs must fail safe to neutral;
- temporary resources must be cleaned when the session ends.

No adapter failure should cause core to silently fall back to an external inference service.

## 14. No persistent emulator mutation

SecondPlayer should not require users to manually edit their normal MesenCE configuration.

If adapter-specific settings are needed, prefer:

1. command-line/session-only configuration;
2. temporary isolated profile/config;
3. bundled script arguments/environment/session metadata.

Persistent mutation of the user's normal emulator setup is a last resort and is out of scope for the clean v1 proof.

## 15. Project layout target

```text
secondplayer/
    adapters/
        base.py
        loader.py
        mesence/
            __init__.py
            adapter.py          # v1 stock-MesenCE adapter
            bridge.lua          # bundled bridge
            smoke.py            # bridge-only headless integration smoke test
    policy/
        laya.py
    config.py
    controller.py
    engine.py
    doctor.py
    cli.py
```

Only `secondplayer/adapters/mesence*` should contain MesenCE implementation details.

## 16. Doctor/diagnostics contract

Core `doctor` checks common runtime prerequisites such as Laya importability.

The adapter module may expose adapter-specific doctor checks. Those checks are dynamically loaded with the adapter and must not be hard-coded in `secondplayer.doctor`.

For MesenCE, adapter doctor checks include:

- stock binary exists;
- compatible version;
- Lua scripting available;
- bridge can be loaded;
- requested controller topology supported.

These checks belong to the MesenCE adapter.

## 17. Acceptance criteria for the MesenCE POC

The first integration is successful when all of the following are demonstrated:

1. A stock MesenCE build is used with no source/binary modification.
2. SecondPlayer launches or attaches through only the MesenCE adapter.
3. A normal SNES game runs normally.
4. The adapter obtains the actual game framebuffer through MesenCE's stock scripting API.
5. The framebuffer reaches local Laya Vision without hidden game state.
6. Player 1 can remain human while Player 2 is AI controlled.
7. Player 2 input is applied through MesenCE's stock input API at the correct polling stage.
8. A human input path remains unaffected by AI seat ownership.
9. AI inference does not block emulator timing.
10. A policy failure releases AI input to neutral.
11. AI + AI can control both standard SNES ports experimentally.
12. No persistent changes are required to the user's ordinary MesenCE profile.
13. Deleting the MesenCE adapter directory leaves the core importable/testable.

## 18. Explicit v1 non-goals

Not required for the first proof:

- Snes9x support;
- RetroArch support;
- arbitrary emulator universality;
- SNES multitap/Players 3–5;
- NES/Genesis/PS1 controller schemas;
- RAM-derived observations;
- game-specific memory integrations;
- remote/cloud model endpoints;
- perfect CPU latency;
- sophisticated personalities or long-term game memory;
- reinforcement learning/training pipeline.

## 19. Retired Snes9x prototype

The first prototype adapter for Snes9x used external X11 capture, `/dev/uinput`, SDL device probing and generated isolated Snes9x config.

It has been deleted from the active codebase after the v1 strategy changed to optimize for the cleanest possible first implementation.

Its removal establishes an important design test: emulator support must be disposable at the adapter boundary. Core code was changed so deleting the Snes9x adapter no longer requires retaining Snes9x config classes, seat limits, platform capture/input dependencies, CLI imports or doctor checks.

See `EMULATOR_SELECTION.md` for the selection rationale.

## 20. Current integration boundary

The MesenCE adapter, Lua bridge, isolated-session launcher, graphical/headless command construction, screenshot transport and two-port input injection are implemented and covered by unit tests.

The next boundary requires a real stock MesenCE binary and SNES ROM:

1. run the bridge-only smoke test under Xvfb;
2. verify actual Player 2 injection and runtime `setInput` compatibility mode;
3. run full local Laya inference against the headless emulator;
4. verify graphical human+AI parity;
5. measure capture and inference latency before further optimization.

See `HEADLESS_TESTING.md`.
