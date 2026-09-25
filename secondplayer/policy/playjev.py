"""PlayJev backend: OmniJev/PlayJev single-forward-pass decisions.

Same harness scaffold as the Laya backend (episodic memory, sweep/stuck,
sliding window, change gating lives in the loops) and same decide()
interface; only inference differs. Each decision is two PlayJevModel calls
(movement options, button options) over the current + previous frame with
the frozen prompt, per-player state text, and lettered controller options.

Game-agnostic: option names/descriptions cover SNES controller inputs only,
never game entities. The game name rides in the state text so the model can
use any world knowledge it has.

Measured limitation (2026-09-23, XPU): the single-letter readout goes
uniform past ~7 options (26 options: conf 0.035 ~= 1/26 on menus and
gameplay; 5 options: up to 0.47). Their training never exceeded 7 moves.
SNES needs 20+ inputs, so this backend fits <=7-move games, not consoles.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Any

import numpy as np
from PIL import Image

from ..config import RuntimeConfig
from ..controller import ControllerState, NEUTRAL
from ..errors import ModelError
from ..platform.devices import resolve_device
from .laya import PlayerDecision, _parse_action_text
from .memory import EpisodicMemory, FrameDiff

log = logging.getLogger(__name__)

PLAYJEV_MODEL_IDS = ("omnijev/playjev-0.8b",)

# Their frozen wording, verbatim (playjev.model.DEFAULT_INSTRUCTIONS).
INSTRUCTIONS = "Which move should the player make next?"

# One atomic move per option, exactly like their training moves
# ("right+jump", "left+fire"). 26 slots max: neutral, 8 directions, 6 face
# buttons, START/SELECT, keep, and the 8 most-used direction+button combos.
# Console-domain (fixed SNES list), never game-specific.
_FULL_MOVES = (
    "neutral",
    "up", "down", "left", "right",
    "up-left", "up-right", "down-left", "down-right",
    "A", "B", "X", "Y", "L", "R", "START", "SELECT",
    "keep",
    "up+A", "down+A", "left+A", "right+A",
    "up+B", "left+B", "right+B", "A+B",
)
assert len(_FULL_MOVES) == 26


def _move_description(move: str) -> str:
    lowered = move.lower()
    if lowered == "neutral":
        return "press nothing"
    if lowered == "keep":
        return "repeat the previous input"
    dirs = {"up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT"}
    holds, presses = [], []
    for part in lowered.split("+"):
        part = part.strip()
        (holds if part in dirs or part in ("up-left", "up-right", "down-left", "down-right") else presses).append(part)
    bits = []
    if holds:
        names = [dirs.get(h, h.upper().replace("-", " and ")) for h in holds]
        bits.append("hold " + " and ".join(names))
    if presses:
        names = ["the {} button".format(p.upper()) for p in presses]
        bits.append("press " + " and ".join(names))
    return " and ".join(bits)


def _full_options() -> list[dict[str, str]]:
    return [{"name": move, "description": _move_description(move)} for move in _FULL_MOVES]


_DIAGONALS = {
    "up-left": ("UP", "LEFT"),
    "up-right": ("UP", "RIGHT"),
    "down-left": ("DOWN", "LEFT"),
    "down-right": ("DOWN", "RIGHT"),
}


def _move_to_state(move: str, prev: ControllerState) -> ControllerState:
    """One atomic move name -> controller state. keep repeats the whole previous state."""
    lowered = move.strip().lower()
    if lowered == "keep":
        return prev
    if lowered == "neutral":
        return NEUTRAL
    buttons: list[str] = []
    for part in lowered.split("+"):
        part = part.strip()
        if part in _DIAGONALS:
            buttons.extend(_DIAGONALS[part])
        elif part:
            buttons.append(part.upper())
    return ControllerState.from_buttons(buttons)


def _role_sentence(player: int) -> str:
    if player == 1:
        return "As the main player, you lead shared menu navigation and start/continue decisions."
    return (
        "You are not the main player: let Player 1 drive shared menus and "
        "only press buttons in a menu that is strictly for you."
    )


def _move_options() -> list[dict[str, str]]:
    return [{"name": name, "description": _MOVE_DESCRIPTIONS[name]} for name in MOVEMENT_CHOICES]


def _button_options() -> list[dict[str, str]]:
    return [{"name": name, "description": _button_description(name)} for name in BUTTON_CHOICES]


class PlayJevPolicy:
    """Local policy wrapper around OmniJev/PlayJev.

    All due players are asked in one batched decide() call per aspect, so the
    shared frames are encoded once and every player/row reuses the work.
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.model = None
        self.previous_frame: np.ndarray | None = None
        self.previous: dict[int, ControllerState] = {}
        self.memories: dict[int, EpisodicMemory] = {}
        self.turn_log: list[dict[str, Any]] = []
        self.history: deque[tuple[np.ndarray, dict[int, str]]] = deque()
        self.rng = np.random.default_rng()

    def load(self) -> None:
        if self.model is not None:
            return
        if self.config.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            from playjev.model import PlayJevModel
        except ImportError as exc:
            raise ModelError(
                "PlayJev backend needs the OmniJev/PlayJev checkout importable "
                "(clone https://github.com/OmniJev/PlayJev and add it to PYTHONPATH)"
            ) from exc
        compute = resolve_device(self.config.device)
        log.info("compute device: %s (%s)", compute.label, compute.reason)
        dtype = compute.dtype or ("bfloat16" if compute.device != "cpu" else "float32")
        try:
            self.model = PlayJevModel(self.config.model, device=compute.device, dtype=dtype).load()
        except Exception as exc:  # pragma: no cover - requires model/runtime
            raise ModelError(f"failed to load PlayJev model {self.config.model!r}: {exc}") from exc

    def _state_text(
        self,
        player: int,
        prev: ControllerState,
        game: str,
        mem: EpisodicMemory | None,
        observed: str | None,
    ) -> str:
        lines = [f"Game: {game}", _role_sentence(player), f"Previous controller state: {prev.as_text()}."]
        if mem is not None and observed is not None:
            rendered = mem.render(observed)
            lines.append(
                "Turn {turn} ({phase}). Last action: {last_action}. Observed: {observed}. "
                "Notes: {memory}. Recent: {recent}.".format(**rendered)
            )
            lines.append(
                "Your notes persist: repeat moves that scored, survived, or made progress; "
                "avoid moves that harmed you; prefer testing controls you have not tried."
            )
        return "\n".join(lines)

    def _ask(
        self,
        frames: list,
        frames_per_state: int,
        options: list[dict[str, str]],
        instructions: str,
        texts: list[str],
    ) -> list[dict[str, Any]]:
        assert self.model is not None
        decisions = self.model.decide(
            frames,
            options,
            instructions=instructions,
            frames_per_state=frames_per_state,
            batch_size=len(frames),
            state_text=texts,
        )
        answers = []
        for decision, options_row in zip(decisions, [options] * len(decisions)):
            answers.append({
                "choice": options_row[decision.choice]["name"],
                "probabilities": {
                    opt["name"]: prob for opt, prob in zip(options_row, decision.probs)
                },
                "confidence": decision.confidence,
            })
            log.debug(
                "playjev: choice=%s conf=%.2f mass=%.3f top=%r",
                options_row[decision.choice]["name"],
                decision.confidence,
                decision.allowed_mass,
                decision.top_token,
            )
        return answers

    def decide(
        self,
        frame: np.ndarray,
        players: list[int],
        *,
        game: str,
    ) -> dict[int, PlayerDecision]:
        self.load()
        assert self.model is not None
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ModelError(f"expected uint8 HWC RGB frame, got {frame.dtype} {frame.shape}")
        learn = self.config.learn
        observed = FrameDiff.summarize(self.previous_frame, frame) if learn else None
        mems: dict[int, EpisodicMemory] = {}
        if learn:
            assert observed is not None
            for player in players:
                mem = self.memories.setdefault(
                    player, EpisodicMemory(memory_turns=self.config.memory_turns)
                )
                mem.note_observed(observed)
                mems[player] = mem

        texts = [
            self._state_text(
                player,
                self.previous.get(player, NEUTRAL),
                game,
                mems.get(player),
                observed,
            )
            for player in players
        ]
        cur = Image.fromarray(frame)
        if self.previous_frame is not None:
            states: list = [(Image.fromarray(self.previous_frame), cur)] * len(players)
            frames_per_state = 2
        else:
            states = [cur] * len(players)
            frames_per_state = 1

        answers = self._ask(states, frames_per_state, _full_options(), INSTRUCTIONS, texts)

        decisions: dict[int, PlayerDecision] = {}
        for player, answer in zip(players, answers):
            prev = self.previous.get(player, NEUTRAL)
            if learn:
                mem = mems[player]
                forced = mem.sweep_action() or mem.stuck_action()
                if forced is not None:
                    state_out = _parse_action_text(forced)
                else:
                    state_out = _move_to_state(self._chosen(answer), prev)
                assert observed is not None
                effect = mem.attribute(mem.last_action, observed)
                mem.record(state_out.as_text(), effect)
                self.turn_log.append({
                    "turn": mem.turn - 1,
                    "player": player,
                    "applied": state_out.as_text(),
                    "forced": forced is not None,
                    "observed": observed,
                    "effect": effect or "none",
                })
            else:
                state_out = _move_to_state(self._chosen(answer), prev)
            # One atomic move per decision: both confidences are the move's.
            conf = float(answer.get("confidence", 0.0))
            decisions[player] = PlayerDecision(
                state=state_out,
                movement_confidence=conf,
                action_confidence=conf,
                raw={"move": answer},
            )
            self.previous[player] = state_out

        self.history.append((frame.copy(), {p: d.state.as_text() for p, d in decisions.items()}))
        while len(self.history) > self.config.window_frames:
            self.history.popleft()
        self.previous_frame = frame.copy()
        return decisions

    def _chosen(self, answer: dict[str, Any]) -> str:
        if not self.config.sample:
            return str(answer["choice"])
        names = list(answer["probabilities"])
        probs = np.asarray([answer["probabilities"][name] for name in names], dtype=float)
        probs = probs / probs.sum()
        return str(self.rng.choice(names, p=probs))
