"""Tests for the PlayJev backend: factory routing, single-call options, learn wiring."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from secondplayer.config import RuntimeConfig
from secondplayer.controller import ControllerState
from secondplayer.policy import create_policy
from secondplayer.policy.laya import LayaVisionPolicy
from secondplayer.policy.playjev import INSTRUCTIONS, PlayJevPolicy, _move_to_state


def frame(value: int = 0, size: int = 32) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


class StubPlayJevModel:
    """Mimics PlayJevModel.decide with a peaked answer."""

    def __init__(self, peaked: str = "left+A"):
        self.peaked = peaked
        self.calls: list[dict[str, Any]] = []

    def decide(self, frames, options, instructions="", frames_per_state=1, batch_size=32,
               long_side=None, state_text=None):
        texts = state_text if isinstance(state_text, list) else [state_text] * len(frames)
        self.calls.append({
            "n_states": len(frames),
            "frames_per_state": frames_per_state,
            "options": [dict(o) for o in options],
            "instructions": instructions,
            "texts": list(texts),
        })
        names = [o["name"] for o in options]
        out = []
        for _ in texts:
            probs = [0.01] * len(names)
            probs[names.index(self.peaked)] = 0.9
            out.append(SimpleNamespace(
                probs=probs, choice=names.index(self.peaked), confidence=0.8,
                allowed_mass=0.9, top_token="A",
            ))
        return out


def test_factory_routes_by_model_id():
    assert isinstance(create_policy(RuntimeConfig(model="OmniJev/PlayJev-0.8B")), PlayJevPolicy)
    assert isinstance(create_policy(RuntimeConfig(model="omnijev/playjev-0.8b")), PlayJevPolicy)
    assert isinstance(create_policy(RuntimeConfig(model="thaitea/laya-vision")), LayaVisionPolicy)
    assert isinstance(create_policy(RuntimeConfig(model="anything-else")), LayaVisionPolicy)


def test_move_to_state_parsing():
    prev = ControllerState.from_buttons(["LEFT"])
    assert _move_to_state("keep", prev) is prev
    assert _move_to_state("neutral", prev).as_text() == "neutral"
    assert _move_to_state("up-left", prev).as_text() == "UP+LEFT"
    assert _move_to_state("right+A", prev).as_text() == "RIGHT+A"
    assert _move_to_state("START", prev).as_text() == "START"


def test_playjev_learn_off_single_atomic_call():
    policy = PlayJevPolicy(RuntimeConfig(learn=False))
    policy.model = StubPlayJevModel(peaked="left+A")
    made = policy.decide(frame(), [1], game="tetris dr mario")
    assert len(policy.model.calls) == 1
    call = policy.model.calls[0]
    assert len(call["options"]) == 26
    assert all(o["description"].strip() for o in call["options"])
    assert call["instructions"] == INSTRUCTIONS
    text = call["texts"][0]
    assert "Game: tetris dr mario" in text
    assert "Notes:" not in text
    assert made[1].state.as_text() == "LEFT+A"
    assert made[1].movement_confidence == 0.8
    assert made[1].action_confidence == 0.8


def test_playjev_first_decide_uses_one_frame_then_two():
    policy = PlayJevPolicy(RuntimeConfig(learn=False))
    policy.model = StubPlayJevModel()
    policy.decide(frame(0), [1], game="game")
    moved = frame(0)
    moved[0:8, 0:8] = 200
    policy.decide(moved, [1], game="game")
    assert [c["frames_per_state"] for c in policy.model.calls] == [1, 2]


def test_playjev_learn_on_sweeps_and_renders_memory():
    policy = PlayJevPolicy(RuntimeConfig(learn=True, memory_turns=4))
    policy.model = StubPlayJevModel()
    first = policy.decide(frame(), [1], game="game")
    assert first[1].state.as_text() == "LEFT"  # sweep turn 0
    assert "nothing learned yet" in policy.model.calls[0]["texts"][0]
    assert "Previous controller state: neutral" in policy.model.calls[0]["texts"][0]

    moved = frame()
    moved[12:20, 0:8] = 200  # fully inside the middle-left sector
    policy.decide(moved, [1], game="game")
    assert "Last action: LEFT" in policy.model.calls[1]["texts"][0]
    mem = policy.memories[1]
    assert mem.facts[("LEFT", "moved-left")].count == 1
    assert [e["applied"] for e in policy.turn_log] == ["LEFT", "RIGHT"]
    assert all(e["forced"] for e in policy.turn_log)
    policy.decide(moved, [1], game="game")
    assert "LEFT->moved-left" in policy.model.calls[2]["texts"][0]


def test_playjev_batches_players_with_per_player_text():
    policy = PlayJevPolicy(RuntimeConfig(learn=False))
    policy.model = StubPlayJevModel()
    policy.decide(frame(), [1, 2], game="game")
    assert len(policy.model.calls) == 1
    call = policy.model.calls[0]
    assert call["n_states"] == 2
    assert "main player" in call["texts"][0]
    assert "not the main player" in call["texts"][1]
