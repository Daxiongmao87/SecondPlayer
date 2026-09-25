"""Tests for the learning scaffold: frame diff, episodic memory, policy wiring."""

from __future__ import annotations

from typing import Any

import numpy as np

from secondplayer.config import RuntimeConfig
from secondplayer.policy.laya import LEARN_SYSTEM, LayaVisionPolicy
from secondplayer.policy.memory import EpisodicMemory, FrameDiff, PollGate, fit_window


def frame(value: int = 0, size: int = 32) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_framediff_first_screen():
    assert FrameDiff.summarize(None, frame()).startswith("first screen")


def test_framediff_static():
    assert FrameDiff.summarize(frame(10), frame(10)).startswith("static")


def test_framediff_motion_names_sector():
    prev = frame()
    cur = frame()
    cur[20:28, 0:8] = 200  # bottom-left block appears
    summary = FrameDiff.summarize(prev, cur)
    assert summary.startswith("motion")
    assert "bottom-left" in summary


def test_framediff_full_flip():
    assert FrameDiff.summarize(frame(0), frame(255)).startswith("full-screen flip")


def test_memory_sweep_order():
    mem = EpisodicMemory()
    seen = []
    for _ in range(8):
        seen.append(mem.sweep_action())
        mem.record("x", None)
    assert seen == ["LEFT", "RIGHT", "UP", "DOWN", "A", "B", "START", "SELECT"]
    assert mem.sweep_action() is None


def test_memory_records_facts_and_caps_recent():
    mem = EpisodicMemory(memory_turns=2)
    mem.record("LEFT", None)  # turn 0: nothing to attribute yet
    # Each effect attributes to the PREVIOUS action (the cause).
    mem.record("RIGHT", "moved-left", 0.9)
    mem.record("A", "moved-right", 0.7)
    mem.record("B", "nothing", 0.5)
    assert mem.facts[("LEFT", "moved-left")].count == 1
    assert mem.facts[("RIGHT", "moved-right")].count == 1
    assert mem.facts[("A", "nothing")].count == 1
    assert len(mem.recent) == 2
    rendered = mem.render("motion middle (small)")
    assert rendered["turn"] == 4
    assert "LEFT->moved-left (1x" in rendered["memory"]
    assert len(rendered["memory"]) < 1200


def test_attribute_mapping():
    # Fresh memory per case: attribute() appends to the ambient window.
    assert EpisodicMemory().attribute(None, "motion middle (small)") is None
    assert EpisodicMemory().attribute("LEFT", "first screen, nothing to compare yet") is None
    assert EpisodicMemory().attribute("LEFT", "static, nothing moved") == "nothing"
    assert EpisodicMemory().attribute("A", "full-screen flip (menu change, death, or transition?)") == "menu"
    assert EpisodicMemory().attribute("LEFT", "motion middle-left (medium)") == "moved-left"
    assert EpisodicMemory().attribute("DOWN", "motion bottom (large)") == "moved-down"
    assert EpisodicMemory().attribute("LEFT", "motion middle (small)") == "unsure"
    assert EpisodicMemory().attribute("A", "motion middle (small)") == "acted"


def test_attribute_ambient_motion_under_distinct_buttons():
    mem = EpisodicMemory()
    motion = "motion bottom (small)"
    assert mem.attribute("LEFT", motion) == "unsure"  # only LEFT has seen it
    assert mem.attribute("RIGHT", motion) == "unsure"  # two buttons, still counts
    assert mem.attribute("UP", motion) == "ambient"  # third distinct button: autonomous
    # Same-button repetition still attributes: consistency under one button is evidence.
    mem2 = EpisodicMemory()
    assert mem2.attribute("LEFT", "motion middle-left (medium)") == "moved-left"
    assert mem2.attribute("LEFT", "motion middle-left (medium)") == "moved-left"


def test_attribute_ambient_matches_variant_summaries():
    mem = EpisodicMemory()
    mem.attribute("LEFT", "motion bottom (small)")
    mem.attribute("RIGHT", "motion bottom (small)")
    # Same size, overlapping region, different wording: still autonomous.
    assert mem.attribute("UP", "motion bottom-left, bottom (small)") == "ambient"


def test_attribute_ambient_needs_overlap():
    mem = EpisodicMemory()
    for button in ("LEFT", "RIGHT", "DOWN"):
        mem.attribute(button, "motion top (small)")
    # Same size but disjoint region: a genuinely new effect, not ambient.
    assert mem.attribute("A", "motion bottom (small)") == "acted"


def test_attribute_fixation_cannot_reset_ambient():
    mem = EpisodicMemory()
    mem.attribute("LEFT", "motion bottom (small)")
    mem.attribute("RIGHT", "motion bottom (small)")
    # Once autonomous, mashing one button keeps reporting ambient forever.
    assert [mem.attribute("A", "motion bottom (small)") for _ in range(3)] == ["ambient"] * 3


def test_record_skips_ambient_facts_but_logs_recent():
    mem = EpisodicMemory()
    mem.record("LEFT", None)
    mem.record("RIGHT", "ambient")
    assert mem.facts == {}
    assert list(mem.recent) == [(1, "LEFT", "ambient", 1.0)]


def test_record_skips_unsure_facts_but_logs_recent():
    mem = EpisodicMemory()
    mem.record("LEFT", None)
    mem.record("RIGHT", "unsure")
    assert mem.facts == {}
    assert list(mem.recent) == [(1, "LEFT", "unsure", 1.0)]


def test_fit_window_math():
    # 1024 - 256 - 3 - 314 text tokens leaves room for 6 frames at 67 tokens.
    assert fit_window(10, 314) == 6
    assert fit_window(3, 314) == 3
    assert fit_window(10, 70) == 10
    assert fit_window(10, 5000) == 1  # text alone overflows: current frame only


def test_poll_gate_first_poll_fires():
    gate = PollGate()
    fire, observed = gate.poll(frame(), 100.0)
    assert fire is True
    assert observed.startswith("first screen")


def test_poll_gate_suppresses_static_until_heartbeat():
    gate = PollGate(min_gap_s=0.1, heartbeat_s=1.0)
    gate.poll(frame(10), 100.0)
    for dt in (0.1, 0.2, 0.5, 0.99):
        fire, observed = gate.poll(frame(10), 100.0 + dt)
        assert fire is False
        assert observed.startswith("static")
    fire, _ = gate.poll(frame(10), 101.0)
    assert fire is True


def test_poll_gate_fires_on_change_with_min_gap():
    gate = PollGate(min_gap_s=0.5, heartbeat_s=60.0)
    gate.poll(frame(0), 100.0)
    moved = frame(0)
    moved[0:8, 0:8] = 200
    fire, _ = gate.poll(moved, 100.1)
    assert fire is False  # changed, but inside the min gap
    moved2 = frame(0)
    moved2[8:16, 8:16] = 200
    fire, observed = gate.poll(moved2, 100.6)
    assert fire is True
    assert observed.startswith("motion")


def test_policy_window_grows_with_history():
    policy = LayaVisionPolicy(RuntimeConfig(learn=False, window_frames=6))
    policy.agent = StubAgent()
    for value in (0, 10, 20):
        policy.decide(frame(value), [1], game="game")
    assert [len(call[0]["images"]) for call in policy.agent.calls] == [1, 2, 3]
    first_images = policy.agent.calls[0][0]["images"]
    assert len(first_images) == 1
    last_images = policy.agent.calls[2][0]["images"]
    assert last_images[-1][0, 0, 0] == 20  # current frame is last
    assert [entry[1][1] for entry in policy.history] == ["LEFT", "LEFT", "LEFT"]
    assert len(policy.history) == 3


def test_policy_window_caps_at_configured_size():
    policy = LayaVisionPolicy(RuntimeConfig(learn=False, window_frames=2))
    policy.agent = StubAgent()
    for value in (0, 10, 20):
        policy.decide(frame(value), [1], game="game")
    assert [len(call[0]["images"]) for call in policy.agent.calls] == [1, 2, 2]
    assert len(policy.history) == 2


def test_policy_window_trims_by_token_budget():
    policy = LayaVisionPolicy(RuntimeConfig(learn=False, window_frames=6))
    policy.agent = StubAgent()
    policy._per_image = 67  # as if calibrated
    for value in (0, 10, 20):
        policy.decide(frame(value), [1], game="game")
    # StubAgent has no tokenizer: falls back to untrimmed window.
    assert len(policy.agent.calls[2][0]["images"]) == 3


def test_memory_stuck_override_picks_least_tried():
    mem = EpisodicMemory()
    for _ in range(5):
        mem.note_observed("static, nothing moved")
    mem.tried.update(["LEFT", "RIGHT", "UP", "DOWN", "A"])
    assert mem.stuck_action() == "B"


def test_memory_stuck_can_reach_start():
    mem = EpisodicMemory()
    for _ in range(5):
        mem.note_observed("static, nothing moved")
    mem.tried.update(["LEFT", "RIGHT", "UP", "DOWN", "A", "B"])
    assert mem.stuck_action() == "START"


def test_memory_summary_shape():
    mem = EpisodicMemory()
    mem.record("LEFT", None)
    mem.record("RIGHT", "moved-left", 1.0)
    summary = mem.summary()
    assert summary["turns"] == 2
    assert summary["facts"]["LEFT->moved-left"] == {"count": 1, "conf": 1.0}


class StubAgent:
    def __init__(self):
        self.calls: list[tuple[dict, dict]] = []

    def predict(self, state, questions, n_permutations=1, batch_size=8):
        self.calls.append((dict(state), {k: dict(v) for k, v in questions.items()}))
        answers: dict[str, Any] = {}
        for qid, qdef in questions.items():
            if qdef["type"] == "noul":
                answers[qid] = {"noul": 0.8}
            elif qid.endswith("r_effect"):
                answers[qid] = {"choice": "moved-left", "confidence": 0.9}
            elif qdef["type"] == "score":
                answers[qid] = {"score": 2.0}
            elif "movement" in qid:
                answers[qid] = {"choice": "left", "confidence": 0.9}
            else:
                answers[qid] = {"choice": "none", "confidence": 0.9}
        return {"answers": answers}


def test_learn_off_path_has_no_memory():
    policy = LayaVisionPolicy(RuntimeConfig(learn=False))
    policy.agent = StubAgent()
    policy.decide(frame(), [1], game="game")
    state, questions = policy.agent.calls[0]
    assert set(questions) == {"p1_movement", "p1_buttons"}
    assert "memory" not in state
    assert "LEARN" not in state["system"] and state["system"] != LEARN_SYSTEM


def test_learn_on_sweeps_first_and_builds_memory():
    policy = LayaVisionPolicy(RuntimeConfig(learn=True, memory_turns=4))
    policy.agent = StubAgent()
    f = frame()
    first = policy.decide(f, [1], game="game")
    assert first[1].state.as_text() == "LEFT"  # sweep turn 0
    state0, questions0 = policy.agent.calls[0]
    assert state0["system"] == LEARN_SYSTEM
    assert state0["memory"] == "nothing learned yet"
    assert "r_effect" not in " ".join(questions0)  # no reflection before any action

    moved = f.copy()
    moved[12:20, 0:8] = 200  # fully inside the middle-left sector
    policy.decide(moved, [1], game="game")
    state1, questions1 = policy.agent.calls[1]
    assert set(questions1) == {"p1_movement", "p1_buttons"}
    assert state1["last_action"] == "LEFT"
    assert "motion" in state1["observed"]
    mem = policy.memories[1]
    assert mem.facts[("LEFT", "moved-left")].count == 1
    assert "LEFT->moved-left" in mem.render("x")["memory"]
    assert [e["applied"] for e in policy.turn_log] == ["LEFT", "RIGHT"]
    assert all(e["forced"] for e in policy.turn_log)
    assert policy.turn_log[1]["effect"] == "moved-left"
