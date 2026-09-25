"""OpenJev backend: render the screen as a cell grid, decide from text.

Single-model pipeline for the frozen OpenJev 4B direct readout (text-only):
a deterministic pixel reader describes each decision frame as filled-cell
coordinates plus the changed cells since the previous frame (see grid.py),
then the baked-in 4-bit quant (see ojcore) scores movement/button options
over that screen text plus motion and memory text. Same decide() interface
and harness scaffold (episodic memory, sweep/stuck, history) as the other
backends.

Scoring is in-process: one forward pass, first-token logprobs restricted
to the verified single-token letter slots. Probe evidence and agreement
numbers live in docs/OJCORE_PROBE.md.

Game-agnostic: the reader reports cell coordinates without naming any game;
options cover controller inputs only.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Any

import numpy as np

from ..config import RuntimeConfig
from ..controller import BUTTON_CHOICES, MOVEMENT_CHOICES, NEUTRAL, ControllerState, state_from_choices
from ..errors import ModelError
from .grid import adaptive_spec, dims_line, parse_spec, render as render_grid
from .laya import PlayerDecision, _parse_action_text
from .memory import EpisodicMemory, FrameDiff
from .ojcore import load_model, load_tokenizer
from .ojcore import score as ojcore_score

log = logging.getLogger(__name__)

OPENJEV_MODEL_IDS = ("qwen/qwen3.5-4b", "quanttrio/qwen3.5-4b-awq")

MOVE_CRITERION = "Which direction should the player hold on the controller?"
BUTTON_CRITERION = "Which buttons should the player press on the controller?"
WARM_LATENCY_WARN_S = 0.2
MENU_HINT = "If a menu or title screen is shown, press the confirm button to advance it."

_MOVE_DESCRIPTIONS = {
    "neutral": "Hold no direction.",
    "up": "Hold the UP direction.",
    "down": "Hold the DOWN direction.",
    "left": "Hold the LEFT direction.",
    "right": "Hold the RIGHT direction.",
    "up-left": "Hold UP and LEFT together.",
    "up-right": "Hold UP and RIGHT together.",
    "down-left": "Hold DOWN and LEFT together.",
    "down-right": "Hold DOWN and RIGHT together.",
    "keep": "Keep holding the current direction.",
}

_BUTTON_DESCRIPTIONS = {
    "none": "Press no buttons.",
    "A": "Press the A button.",
    "B": "Press the B button.",
    "X": "Press the X button.",
    "Y": "Press the Y button.",
    "L": "Press the L button.",
    "R": "Press the R button.",
    "A+B": "Press the A and B buttons together.",
    "A+Y": "Press the A and Y buttons together.",
    "B+Y": "Press the B and Y buttons together.",
    "X+Y": "Press the X and Y buttons together.",
    "START": "Press the START button.",
    "SELECT": "Press the SELECT button.",
    "keep": "Keep holding the current buttons.",
}


def _role_sentence(player: int) -> str:
    if player == 1:
        return "As the main player, you lead shared menu navigation and start/continue decisions."
    return (
        "You are not the main player: let Player 1 drive shared menus and "
        "only press buttons in a menu that is strictly for you."
    )


class OpenJevPolicy:
    """Caption-then-decide policy over the frozen OpenJev readout."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._loaded = False
        self._oj_model = None
        self._oj_tokenizer = None
        self._oj_score = None
        self._oj_metadata: dict[str, Any] = {}
        self._score_calls = 0
        # Fail fast on malformed display geometry; None means whole-frame adaptive.
        self._grid_spec = parse_spec(config.screen_grid)
        self.previous_frame: np.ndarray | None = None
        self.previous: dict[int, ControllerState] = {}
        self.memories: dict[int, EpisodicMemory] = {}
        self.turn_log: list[dict[str, Any]] = []
        self.history: deque[tuple[np.ndarray, dict[int, str]]] = deque()
        self.rng = np.random.default_rng()

    def load(self) -> None:
        if self._loaded:
            return
        if self.config.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        weights = self.config.openjev_weights.strip() or self.config.model
        tokenizer = load_tokenizer(weights, offline=self.config.offline)
        model, compute = load_model(
            weights,
            self.config.device,
            quant=self.config.quant_backend,
            offline=self.config.offline,
        )
        log.info("openjev weights: %s (%s, %s)", weights, compute.label, self.config.quant_backend)
        self._oj_model = model
        self._oj_tokenizer = tokenizer
        self._oj_score = ojcore_score
        self._oj_metadata = {
            "weights": weights,
            "backend": self.config.quant_backend,
            "device": compute.device,
        }
        self._loaded = True

    def _screen_text(self, frame: np.ndarray) -> str:
        spec = self._grid_spec or adaptive_spec(frame.shape[1], frame.shape[0])
        return f"{dims_line(spec)}\n{render_grid(self.previous_frame, frame, spec)}"

    def _state_text(
        self,
        player: int,
        prev: ControllerState,
        game: str,
        screen: str,
        mem: EpisodicMemory | None,
        observed: str | None,
    ) -> str:
        lines = [
            f"Game: {game}",
            f"Objective: {self.config.objective.strip()}",
            _role_sentence(player),
            f"Screen:\n{screen}",
            f"Previous controller state: {prev.as_text()}.",
            MENU_HINT,
        ]
        if mem is not None and observed is not None:
            rendered = mem.render(observed)
            lines.append(
                "Turn {turn} ({phase}). Last action: {last_action}. Changed since last screen: "
                "{observed}. Notes: {memory}. Recent: {recent}.".format(**rendered)
            )
            lines.append(
                "Your notes persist: repeat moves that scored, survived, or made progress; "
                "avoid moves that harmed you; prefer testing controls you have not tried."
            )
        return "\n".join(lines)

    def _ask(self, kind: str, state: str) -> dict[str, Any]:
        assert self._oj_score is not None
        if kind == "movement":
            criterion, names, descriptions = MOVE_CRITERION, MOVEMENT_CHOICES, _MOVE_DESCRIPTIONS
        else:
            criterion, names, descriptions = BUTTON_CRITERION, BUTTON_CHOICES, _BUTTON_DESCRIPTIONS
        row = {
            "id": kind,
            "state": state,
            "question": criterion,
            "options": [{"id": name, "description": descriptions[name]} for name in names],
        }
        start = time.monotonic()
        result = self._oj_score(self._oj_model, self._oj_tokenizer, row, self._oj_metadata)
        elapsed = time.monotonic() - start
        self._score_calls += 1
        if self._score_calls > 1 and elapsed > WARM_LATENCY_WARN_S:
            log.warning(
                "suboptimal AI performance: warm jev request %.0fms exceeds 200ms budget (%s)",
                elapsed * 1000.0,
                kind,
            )
        probs = result["probabilities"]
        top = max(range(len(names)), key=probs.__getitem__)
        return {
            "choice": names[top],
            "probabilities": {name: prob for name, prob in zip(names, probs)},
            "confidence": float(probs[top]),
            "latency_s": elapsed,
        }

    def decide(
        self,
        frame: np.ndarray,
        players: list[int],
        *,
        game: str,
    ) -> dict[int, PlayerDecision]:
        self.load()
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ModelError(f"expected uint8 HWC RGB frame, got {frame.dtype} {frame.shape}")
        learn = self.config.learn
        screen = self._screen_text(frame)
        observed = FrameDiff.summarize(self.previous_frame, frame) if learn else None
        mems: dict[int, EpisodicMemory] = {}
        if learn:
            assert observed is not None
            for player in players:
                mem = self.memories.setdefault(player, EpisodicMemory(memory_turns=self.config.memory_turns))
                mem.note_observed(observed)
                mems[player] = mem

        decisions: dict[int, PlayerDecision] = {}
        for player in players:
            prev = self.previous.get(player, NEUTRAL)
            move_state = self._state_text(
                player, prev, game, screen, mems.get(player), observed
            )
            move = self._ask("movement", move_state)
            move_choice = self._chosen(move)
            act_state = self._state_text(
                player, prev, game, screen, mems.get(player), observed
            )
            act = self._ask("buttons", act_state)
            act_choice = self._chosen(act)
            if learn:
                mem = mems[player]
                # No sweep: blind button-mashing pauses real games. Stuck-only
                # recovery fires solely on frozen screens, which need it.
                stuck = mem.stuck_action()
                if stuck is not None:
                    state_out = _parse_action_text(stuck)
                    forced = stuck
                else:
                    state_out = state_from_choices(move_choice, act_choice, prev)
                    forced = None
                assert observed is not None
                effect = mem.attribute(mem.last_action, observed)
                mem.record(state_out.as_text(), effect)
                self.turn_log.append(
                    {
                        "turn": mem.turn - 1,
                        "player": player,
                        "applied": state_out.as_text(),
                        "forced": forced is not None,
                        "move_choice": move_choice,
                        "move_conf": round(float(move.get("confidence", 0.0)), 3),
                        "act_choice": act_choice,
                        "act_conf": round(float(act.get("confidence", 0.0)), 3),
                        "observed": observed,
                        "effect": effect or "none",
                        "screen": screen,
                    }
                )
            else:
                state_out = state_from_choices(move_choice, act_choice, prev)
            decisions[player] = PlayerDecision(
                state=state_out,
                movement_confidence=float(move.get("confidence", 0.0)),
                action_confidence=float(act.get("confidence", 0.0)),
                raw={"movement": move, "buttons": act, "screen": screen},
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
