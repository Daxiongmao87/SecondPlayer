from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .errors import ConfigurationError

PlayerMode = Literal["human", "ai", "disabled"]


@dataclass(slots=True)
class RuntimeConfig:
    model: str = "thaitea/laya-vision"
    # Pinned Hub revision (commit SHA) for `model`; blank follows the default branch.
    model_revision: str = ""
    openjev_weights: str = ""
    quant_backend: str = "quanto_qint4"
    # Display geometry for the text backend's cell grid: "ox,oy,cell_w,cell_h,cols,rows"
    # in pixels (blank = whole-frame adaptive). Renderer data, not game rules.
    screen_grid: str = ""
    device: str = "auto"
    reaction_ms: int = 200
    permutations: int = 1
    poll_ms: int = 100
    window_frames: int = 6
    sample: bool = False
    learn: bool = False
    memory_turns: int = 4
    objective: str = "Play the game effectively. If other players are present, treat them as teammates unless the game is clearly competitive."
    offline: bool = False


@dataclass(slots=True)
class AdapterSelection:
    """Core-visible adapter selection.

    The core owns only the adapter name and an opaque options mapping. Parsing,
    defaults, capability checks, launch behavior, capture and input injection are
    responsibilities of the adapter module itself.
    """

    name: str = "mesence"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionConfig:
    players: dict[int, PlayerMode] = field(default_factory=lambda: {1: "human", 2: "ai"})

    @property
    def ai_players(self) -> list[int]:
        return sorted(p for p, mode in self.players.items() if mode == "ai")

    @property
    def human_players(self) -> list[int]:
        return sorted(p for p, mode in self.players.items() if mode == "human")

    @property
    def active_players(self) -> list[int]:
        return sorted(p for p, mode in self.players.items() if mode != "disabled")


@dataclass(slots=True)
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    adapter: AdapterSelection = field(default_factory=AdapterSelection)
    session: SessionConfig = field(default_factory=SessionConfig)


def default_config_path() -> Path:
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "secondplayer" / "config.toml"


def _players(raw: object) -> dict[int, PlayerMode]:
    if raw is None:
        return {1: "human", 2: "ai"}
    if not isinstance(raw, dict):
        raise ConfigurationError("[session].players must be a table")
    out: dict[int, PlayerMode] = {}
    for k, v in raw.items():
        try:
            player = int(k)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid player key {k!r}") from exc
        if player < 1:
            raise ConfigurationError("player IDs must be positive integers")
        mode = str(v).lower()
        if mode not in {"human", "ai", "disabled"}:
            raise ConfigurationError(f"invalid mode for player {player}: {v!r}")
        out[player] = mode  # type: ignore[assignment]
    return out


def load_config(path: str | Path | None = None) -> AppConfig:
    cfg = AppConfig()
    path = Path(path or default_config_path())
    if not path.exists():
        return cfg

    with path.open("rb") as f:
        raw = tomllib.load(f)

    rr = raw.get("runtime", {})
    ar = raw.get("adapter", {})
    ss = raw.get("session", {})

    if not isinstance(rr, dict):
        raise ConfigurationError("[runtime] must be a table")
    if not isinstance(ar, dict):
        raise ConfigurationError("[adapter] must be a table")
    if not isinstance(ss, dict):
        raise ConfigurationError("[session] must be a table")

    for name in RuntimeConfig.__dataclass_fields__:
        if name in rr:
            setattr(cfg.runtime, name, rr[name])

    if "name" in ar:
        cfg.adapter.name = str(ar["name"]).strip().lower()
    options = ar.get("options", {})
    if not isinstance(options, dict):
        raise ConfigurationError("[adapter.options] must be a table")
    cfg.adapter.options = dict(options)
    cfg.session.players = _players(ss.get("players"))

    if not cfg.adapter.name:
        raise ConfigurationError("adapter.name must not be empty")
    if cfg.runtime.reaction_ms < 0:
        raise ConfigurationError("reaction_ms must be >= 0")
    if cfg.runtime.permutations < 1:
        raise ConfigurationError("permutations must be >= 1")
    if cfg.runtime.poll_ms < 1:
        raise ConfigurationError("poll_ms must be >= 1")
    if cfg.runtime.window_frames < 1:
        raise ConfigurationError("window_frames must be >= 1")
    if cfg.runtime.memory_turns < 1:
        raise ConfigurationError("memory_turns must be >= 1")
    if not cfg.session.ai_players:
        raise ConfigurationError("at least one player must be assigned to ai")
    return cfg


DEFAULT_TOML = """# SecondPlayer configuration

[runtime]
# Release builds are expected to bundle this checkpoint locally.
model = "thaitea/laya-vision"
# Pinned Hub revision (commit SHA) for the model above; blank follows default.
model_revision = ""
# OpenJev 4-bit decider: baked-in weights dir (empty = use `model` as-is).
openjev_weights = ""
# Backend for the decider: none | quanto_qint4 | torchao_int4 | awq.
quant_backend = "quanto_qint4"
# Cell-grid region for the text backend's screen reader, as
# "ox,oy,cell_w,cell_h,cols,rows" in pixels; blank covers the whole frame.
# Display geometry for the renderer (like screen resolution), never game rules.
screen_grid = ""
device = "auto"
# Minimum wall-clock gap between submitted decisions. If inference is slower,
# the current controller state simply remains held until the result arrives.
reaction_ms = 200
permutations = 1
# Visual poll rate: capture up to this often, but only run inference when the
# scene changed (or the 1s heartbeat fires on a frozen screen).
poll_ms = 100
# Sliding window: up to this many recent frames + their control choices are
# kept in the model state, trimmed to fit the checkpoint's context window.
window_frames = 6
sample = false
offline = false
learn = false
memory_turns = 4
objective = "Play the game effectively. If other players are present, treat them as teammates unless the game is clearly competitive."

# Adapter-specific behavior is opaque to core SecondPlayer. The selected adapter
# parses everything under [adapter.options].
[adapter]
name = "mesence"

[adapter.options]
binary = "Mesen"
headless = false
xvfb = true

[session.players]
1 = "human"
2 = "ai"
"""


def write_default_config(path: str | Path | None = None, overwrite: bool = False) -> Path:
    path = Path(path or default_config_path())
    if path.exists() and not overwrite:
        raise ConfigurationError(f"config already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_TOML, encoding="utf-8")
    return path
