from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RuntimeConfig
from ..controller import BUTTON_CHOICES, MOVEMENT_CHOICES, ControllerState, NEUTRAL, state_from_choices
from ..errors import ModelError
from ..platform.devices import resolve_device
from .memory import EpisodicMemory, FrameDiff, fit_window

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PlayerDecision:
    state: ControllerState
    movement_confidence: float
    action_confidence: float
    raw: dict[str, Any]


BASE_SYSTEM = "SecondPlayer controls only the requested SNES controller ports from visible pixels."

# Universal learning prompt: teaches a stateless System-1 model to use the
# harness-kept history in its state for frame-to-frame cause and effect.
LEARN_SYSTEM = (
    "You are playing a game you have never seen before. Each turn you receive the current screen, "
    "what changed since the previous screen, the button you just pressed, and your notes from earlier turns. "
    "Your notes persist: they are your only memory, so use them. Every turn, do three things: "
    "1. CAUSE AND EFFECT. Your notes record what each button did on past turns. Trust effects you have "
    "seen repeat; distrust single coincidences. Things that move every turn regardless of your button "
    "move on their own: effects marked ambient happened without you, so never credit your button for them. "
    "2. LEARN THE CONTROLS. Early on, test buttons you have not tried and watch what happens. "
    "One new test per turn beats repeating what does nothing. "
    "3. PLAY TO WIN. Repeat moves your notes say scored, survived, or made progress; "
    "avoid moves your notes say harmed you. Score points, stay alive, finish what the game asks."
)

LEARN_SUFFIX = (
    " Consult your notes in the state: repeat moves that worked, avoid moves that failed, "
    "and prefer testing controls you have not tried."
)


def _parse_action_text(text: str) -> ControllerState:
    if text.strip().lower() == "neutral":
        return NEUTRAL
    return ControllerState.from_buttons(part.strip() for part in text.split("+") if part.strip())


def _laya_code_sha() -> str | None:
    """Installed laya-vision code commit, from pip's direct-URL metadata."""
    try:
        from importlib import metadata

        raw = metadata.direct_url_json("laya")
    except Exception:
        return None
    if not raw:
        return None
    try:
        commit = (raw.get("vcs_info") or {}).get("commit_id")
    except Exception:
        return None
    return str(commit) if commit else None


def _agent_params(agent: Any) -> int | None:
    try:
        return int(sum(p.numel() for p in agent.model.parameters()))
    except Exception:
        return None


class LayaVisionPolicy:
    """Thin local policy wrapper around Laya Vision.

    All due players are asked in one predict() call so Laya encodes the shared
    screenshot once and reuses those image features across every player/question.
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.agent = None
        self.previous_frame: np.ndarray | None = None
        self.previous: dict[int, ControllerState] = {}
        self.memories: dict[int, EpisodicMemory] = {}
        self.turn_log: list[dict[str, Any]] = []
        self.rng = np.random.default_rng()
        # Sliding window: (frame, {player: action text}) per decision, oldest first.
        self.history: deque[tuple[np.ndarray, dict[int, str]]] = deque()
        self._per_image: int | None = None
        self._zero_prefix: int = 3
        self._max_len: int = 1024
        self._head_len: int = 256

    def load(self) -> None:
        if self.agent is not None:
            return
        if self.config.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            import laya
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise ModelError(f"failed to import bundled Laya Vision runtime: {exc}") from exc
        compute = resolve_device(self.config.device)
        log.info("compute device: %s (%s)", compute.label, compute.reason)
        kwargs: dict[str, Any] = {"device": compute.device}
        if compute.dtype is not None:
            kwargs["dtype"] = compute.dtype
        if self.config.model_revision:
            kwargs["revision"] = self.config.model_revision
        try:
            self.agent = laya.load_vlm(self.config.model, **kwargs)
        except Exception as exc:  # pragma: no cover - requires model/runtime
            raise ModelError(f"failed to load Laya Vision model {self.config.model!r}: {exc}") from exc
        self._calibrate_window()

    def model_info(self) -> dict[str, Any]:
        """Provenance for benchmark reports: repo, revisions, code SHA, params."""
        self.load()
        assert self.agent is not None
        source = getattr(self.agent, "source", None) or {}
        info: dict[str, Any] = {
            "repo": self.config.model,
            "revision_requested": self.config.model_revision or None,
            "revision_resolved": source.get("revision"),
            "laya_code_sha": _laya_code_sha(),
            "params": _agent_params(self.agent),
        }
        return info

    @staticmethod
    def _role_instruction(player: int) -> str:
        if player == 1:
            return (
                "As the main player, you lead shared menu navigation and "
                "start/continue decisions."
            )
        return (
            "You are not the main player: let Player 1 drive shared menus and "
            "only press buttons in a menu that is strictly for you."
        )

    @staticmethod
    def _question(
        player: int, kind: str, previous: ControllerState, learn: bool = False
    ) -> dict[str, Any]:
        role = LayaVisionPolicy._role_instruction(player)
        suffix = LEARN_SUFFIX if learn else ""
        if kind == "movement":
            return {
                "type": "choice",
                "instructions": (
                    f"You control SNES Player {player}. {role} "
                    "Choose the directional input to use now. "
                    f"The previous controller state was {previous.as_text()}. "
                    "Choose keep only when continuing the current movement is useful." + suffix
                ),
                "criteria": list(MOVEMENT_CHOICES),
            }
        return {
            "type": "choice",
            "instructions": (
                f"You control SNES Player {player}. {role} "
                "Choose the non-directional button input to use now. "
                f"The previous controller state was {previous.as_text()}. "
                "Choose keep only when continuing the currently held action buttons is useful." + suffix
            ),
            "criteria": list(BUTTON_CHOICES),
        }

    def _calibrate_window(self) -> None:
        """Measure this checkpoint's image-token cost once (tokenizer only)."""
        assert self.agent is not None
        cfg = getattr(self.agent, "cfg", {}) or {}
        self._max_len = int(cfg.get("max_len", 1024))
        self._head_len = int(cfg.get("head_max_len", 256))
        try:
            from laya.vlm import vlm_prefix

            proc = self.agent.processor
            prep = getattr(proc, "laya_prep", None)
            dummy = np.zeros((16, 16, 3), dtype=np.uint8)
            zero = len(vlm_prefix(proc, [], prep)["ids"])
            one = len(vlm_prefix(proc, [dummy], prep)["ids"])
            self._zero_prefix = zero
            self._per_image = max(1, one - zero)
        except Exception:  # pragma: no cover - requires model/runtime
            log.warning("window calibration failed; using untrimmed history")
            self._per_image = None

    def _fit_count(self, text_state: dict[str, Any], n_available: int) -> int:
        """Frames of the sliding window that fit next to this decision's text."""
        capped = max(1, min(n_available, self.config.window_frames))
        if self._per_image is None:
            return capped
        tokenizer = getattr(getattr(self.agent, "processor", None), "tokenizer", None)
        if tokenizer is None:
            return capped
        text = json.dumps(text_state, ensure_ascii=False)
        text_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        return fit_window(
            capped,
            text_tokens,
            max_len=self._max_len,
            head_len=self._head_len,
            zero_prefix=self._zero_prefix,
            per_image=self._per_image,
        )

    def _window_images(self, text_state: dict[str, Any], frame: np.ndarray) -> list[np.ndarray]:
        frames = [entry[0] for entry in self.history] + [frame]
        return frames[-self._fit_count(text_state, len(frames)) :]

    def _remember(self, frame: np.ndarray, decisions: dict[int, PlayerDecision]) -> None:
        self.history.append((frame.copy(), {p: d.state.as_text() for p, d in decisions.items()}))
        while len(self.history) > self.config.window_frames:
            self.history.popleft()
        self.previous_frame = frame.copy()

    def decide(
        self,
        frame: np.ndarray,
        players: list[int],
        *,
        game: str,
    ) -> dict[int, PlayerDecision]:
        self.load()
        assert self.agent is not None
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ModelError(f"expected uint8 HWC RGB frame, got {frame.dtype} {frame.shape}")
        if self.config.learn:
            return self._decide_learn(frame, players, game=game)

        questions: dict[str, dict[str, Any]] = {}
        for player in players:
            prev = self.previous.get(player, NEUTRAL)
            questions[f"p{player}_movement"] = self._question(player, "movement", prev)
            questions[f"p{player}_buttons"] = self._question(player, "buttons", prev)

        state: dict[str, Any] = {
            "game": game,
            "objective": self.config.objective,
            "system": "SecondPlayer controls only the requested SNES controller ports from visible pixels.",
        }
        if len(players) == 1:
            state["last_action"] = self.previous.get(players[0], NEUTRAL).as_text()
        else:
            for player in players:
                state[f"p{player}_last_action"] = self.previous.get(player, NEUTRAL).as_text()
        state["images"] = self._window_images(state, frame)

        try:
            result = self.agent.predict(
                state,
                questions,
                n_permutations=self.config.permutations,
                batch_size=max(2, min(8, len(questions))),
            )
        except Exception as exc:  # pragma: no cover - requires model/runtime
            raise ModelError(f"Laya Vision inference failed: {exc}") from exc

        answers = result["answers"]
        decisions: dict[int, PlayerDecision] = {}

        def chosen(answer: dict[str, Any]) -> str:
            if not self.config.sample:
                return str(answer["choice"])
            names = list(answer["probabilities"])
            probs = np.asarray([answer["probabilities"][name] for name in names], dtype=float)
            probs = probs / probs.sum()
            return str(self.rng.choice(names, p=probs))

        for player in players:
            move = answers[f"p{player}_movement"]
            act = answers[f"p{player}_buttons"]
            prev = self.previous.get(player, NEUTRAL)
            move_choice, act_choice = chosen(move), chosen(act)
            state_out = state_from_choices(move_choice, act_choice, prev)
            decisions[player] = PlayerDecision(
                state=state_out,
                movement_confidence=float(move.get("confidence", 0.0)),
                action_confidence=float(act.get("confidence", 0.0)),
                raw={"movement": move, "buttons": act},
            )
            self.previous[player] = state_out

        self._remember(frame, decisions)
        return decisions

    def _decide_learn(
        self, frame: np.ndarray, players: list[int], *, game: str
    ) -> dict[int, PlayerDecision]:
        assert self.agent is not None
        observed = FrameDiff.summarize(self.previous_frame, frame)
        mems: dict[int, EpisodicMemory] = {}
        for player in players:
            mem = self.memories.setdefault(
                player, EpisodicMemory(memory_turns=self.config.memory_turns)
            )
            mem.note_observed(observed)
            mems[player] = mem

        questions: dict[str, dict[str, Any]] = {}
        for player in players:
            prev = self.previous.get(player, NEUTRAL)
            questions[f"p{player}_movement"] = self._question(player, "movement", prev, learn=True)
            questions[f"p{player}_buttons"] = self._question(player, "buttons", prev, learn=True)

        state: dict[str, Any] = {
            "game": game,
            "objective": self.config.objective,
            "system": LEARN_SYSTEM,
        }
        if len(players) == 1:
            state.update(mems[players[0]].render(observed))
        else:
            for player in players:
                for key, value in mems[player].render(observed).items():
                    state[f"p{player}_{key}"] = value
        state["images"] = self._window_images(state, frame)

        try:
            result = self.agent.predict(
                state,
                questions,
                n_permutations=self.config.permutations,
                batch_size=max(2, min(8, len(questions))),
            )
        except Exception as exc:  # pragma: no cover - requires model/runtime
            raise ModelError(f"Laya Vision inference failed: {exc}") from exc

        answers = result["answers"]
        decisions: dict[int, PlayerDecision] = {}
        for player in players:
            move = answers[f"p{player}_movement"]
            act = answers[f"p{player}_buttons"]
            prev = self.previous.get(player, NEUTRAL)
            mem = mems[player]
            forced = mem.sweep_action() or mem.stuck_action()
            if forced is not None:
                state_out = _parse_action_text(forced)
            else:
                state_out = state_from_choices(
                    self._chosen(move), self._chosen(act), prev
                )
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
            decisions[player] = PlayerDecision(
                state=state_out,
                movement_confidence=float(move.get("confidence", 0.0)),
                action_confidence=float(act.get("confidence", 0.0)),
                raw={"movement": move, "buttons": act, "effect": effect or "none"},
            )
            self.previous[player] = state_out

        self._remember(frame, decisions)
        return decisions

    def _chosen(self, answer: dict[str, Any]) -> str:
        if not self.config.sample:
            return str(answer["choice"])
        names = list(answer["probabilities"])
        probs = np.asarray([answer["probabilities"][name] for name in names], dtype=float)
        probs = probs / probs.sum()
        return str(self.rng.choice(names, p=probs))
