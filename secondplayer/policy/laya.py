from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RuntimeConfig
from ..controller import BUTTON_CHOICES, MOVEMENT_CHOICES, ControllerState, NEUTRAL, state_from_choices
from ..errors import ModelError


@dataclass(slots=True)
class PlayerDecision:
    state: ControllerState
    movement_confidence: float
    action_confidence: float
    raw: dict[str, Any]


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
        self.rng = np.random.default_rng()

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
        try:
            self.agent = laya.load_vlm(self.config.model, device=self.config.device)
        except Exception as exc:  # pragma: no cover - requires model/runtime
            raise ModelError(f"failed to load Laya Vision model {self.config.model!r}: {exc}") from exc

    @staticmethod
    def _question(player: int, kind: str, previous: ControllerState) -> dict[str, Any]:
        if kind == "movement":
            return {
                "type": "choice",
                "instructions": (
                    f"You control SNES Player {player}. Choose the directional input to use now. "
                    f"The previous controller state was {previous.as_text()}. "
                    "Choose keep only when continuing the current movement is useful."
                ),
                "criteria": list(MOVEMENT_CHOICES),
            }
        return {
            "type": "choice",
            "instructions": (
                f"You control SNES Player {player}. Choose the non-directional button input to use now. "
                f"The previous controller state was {previous.as_text()}. "
                "Choose keep only when continuing the currently held action buttons is useful."
            ),
            "criteria": list(BUTTON_CHOICES),
        }

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
        if self.config.frames == 2 and self.previous_frame is not None:
            state["images"] = [self.previous_frame, frame]
        else:
            state["image"] = frame

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

        self.previous_frame = frame.copy()
        return decisions
