"""Tests for the OpenJev backend: factory routing, grid+score wiring, learn flow."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pytest

from secondplayer.config import RuntimeConfig
from secondplayer.controller import NEUTRAL, state_from_choices
from secondplayer.errors import ConfigurationError
from secondplayer.policy import create_policy
from secondplayer.policy.laya import LayaVisionPolicy
from secondplayer.policy.openjev import OpenJevPolicy


def frame(value: int = 0, size: int = 32) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


class StubScorer:
    """Mimics openjev direct.score: movement answers left, buttons answer A."""

    def __init__(self):
        self.rows: list[dict[str, Any]] = []

    def __call__(self, model, tokenizer, row, metadata):
        self.rows.append(dict(row))
        peaked = "left" if row["id"] == "movement" else "A"
        names = [o["id"] for o in row["options"]]
        probs = [0.01] * len(names)
        probs[names.index(peaked)] = 0.9
        return {"option_ids": names, "probabilities": probs}


def _stubbed(learn: bool = False, **kwargs) -> OpenJevPolicy:
    policy = OpenJevPolicy(RuntimeConfig(learn=learn, **kwargs))
    policy._loaded = True
    policy._oj_model = object()
    policy._oj_tokenizer = object()
    policy._oj_score = StubScorer()
    return policy


def test_factory_routes_openjev_ids():
    assert isinstance(create_policy(RuntimeConfig(model="Qwen/Qwen3.5-4B")), OpenJevPolicy)
    assert isinstance(create_policy(RuntimeConfig(model="quanttrio/qwen3.5-4b-awq")), OpenJevPolicy)
    assert isinstance(create_policy(RuntimeConfig(model="thaitea/laya-vision")), LayaVisionPolicy)


# Slot readout coverage moved with the code to tests/test_ojcore.py.


def test_openjev_learn_off_scores_two_aspects():
    policy = _stubbed()
    made = policy.decide(frame(), [1], game="dr mario")
    rows = policy._oj_score.rows
    assert [r["id"] for r in rows] == ["movement", "buttons"]
    assert [o["id"] for o in rows[0]["options"]] == [
        "neutral",
        "up",
        "down",
        "left",
        "right",
        "up-left",
        "up-right",
        "down-left",
        "down-right",
        "keep",
    ]
    assert len(rows[1]["options"]) == 14
    assert all(len(r["options"]) <= 16 for r in rows)
    assert "Current frame filled cells" in rows[0]["state"]
    assert "first screen, nothing to compare yet" in rows[0]["state"]
    assert "Game: dr mario\n" in rows[0]["state"]
    assert "Objective: Play the game effectively." in rows[0]["state"]
    assert "Notes:" not in rows[0]["state"]
    assert made[1].state == state_from_choices("left", "A", NEUTRAL)
    assert made[1].movement_confidence == 0.9
    assert "Current frame filled cells" in made[1].raw["screen"]


def test_screen_grid_roi_reaches_state():
    policy = _stubbed(screen_grid="0,0,8,8,4,4")
    policy.decide(frame(), [1], game="game")
    state = policy._oj_score.rows[0]["state"]
    assert "4 columns (0-3" in state
    assert "4 rows (0-3" in state


def test_malformed_screen_grid_fails_fast():
    with pytest.raises(ConfigurationError):
        OpenJevPolicy(RuntimeConfig(screen_grid="93,44"))


def test_openjev_learn_on_records_memory_without_forcing():
    policy = _stubbed(learn=True, memory_turns=4)
    first = policy.decide(frame(), [1], game="game")
    applied0 = state_from_choices("left", "A", NEUTRAL).as_text()
    assert first[1].state.as_text() == applied0  # model drives, never swept
    assert "nothing learned yet" in policy._oj_score.rows[0]["state"]

    moved = frame()
    moved[12:20, 0:8] = 200  # fully inside the middle-left sector
    policy.decide(moved, [1], game="game")
    assert f"Last action: {applied0}" in policy._oj_score.rows[2]["state"]
    mem = policy.memories[1]
    assert mem.facts[(applied0, "moved-left")].count == 1
    assert [e["applied"] for e in policy.turn_log] == [applied0, applied0]
    assert not any(e["forced"] for e in policy.turn_log)
    assert policy.turn_log[0]["move_choice"] == "left"
    assert policy.turn_log[0]["act_choice"] == "A"
    assert policy.turn_log[0]["move_conf"] == 0.9
    assert "Current frame filled cells" in policy.turn_log[0]["screen"]
    policy.decide(moved, [1], game="game")
    assert f"{applied0}->moved-left" in policy._oj_score.rows[4]["state"]


def test_load_marks_loaded_once(monkeypatch):
    import secondplayer.policy.openjev as ojmod

    calls: list[str] = []

    class Compute:
        device = "cpu"
        label = "CPU"

    def fake_tokenizer(weights, offline=False):
        calls.append("tok")
        return object()

    def fake_model(weights, device, quant=None, offline=False):
        calls.append("model")
        return object(), Compute()

    monkeypatch.setattr(ojmod, "load_tokenizer", fake_tokenizer)
    monkeypatch.setattr(ojmod, "load_model", fake_model)
    policy = OpenJevPolicy(RuntimeConfig())
    policy.load()
    assert policy._loaded is True
    assert policy._oj_score is not None
    policy.load()
    assert calls == ["tok", "model"]  # second load is a no-op, not a reload


def test_learn_lets_model_drive_menus_and_gameplay():
    policy = _stubbed(learn=True, memory_turns=4)
    made = policy.decide(frame(), [1], game="game")
    assert made[1].state == state_from_choices("left", "A", NEUTRAL)
    assert policy.turn_log[0]["forced"] is False

    busy = frame()
    busy[0:16, 0:16] = 200  # a visibly different screen
    policy2 = _stubbed(learn=True, memory_turns=4)
    made2 = policy2.decide(busy, [1], game="game")
    assert made2[1].state == state_from_choices("left", "A", NEUTRAL)
    assert policy2.turn_log[0]["forced"] is False
    # Pixels reach the state: the two screens render different text.
    assert policy._oj_score.rows[0]["state"] != policy2._oj_score.rows[0]["state"]


def test_warm_latency_warning(monkeypatch, caplog):
    policy = _stubbed()
    times = iter([0.0, 0.01, 1.0, 1.5])
    monkeypatch.setattr(time, "monotonic", lambda: next(times))
    policy._ask("movement", "state")  # cold call: no warning
    assert "suboptimal AI performance" not in caplog.text
    with caplog.at_level(logging.WARNING, logger="secondplayer.policy.openjev"):
        out = policy._ask("movement", "state")  # warm call at 500ms
    assert "suboptimal AI performance" in caplog.text
    assert "500ms" in caplog.text
    assert out["latency_s"] == pytest.approx(0.5)
