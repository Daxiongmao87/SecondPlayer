import time

import numpy as np

from secondplayer.controller import ControllerState
from secondplayer.engine import SecondPlayerEngine
from secondplayer.policy.laya import PlayerDecision


class FakeAdapter:
    game_name = "Test Game"

    def __init__(self):
        self.running = False
        self.applied = []
        self.captures = 0

    def start(self):
        self.running = True

    def is_running(self):
        return self.running

    def capture(self):
        self.captures += 1
        if self.captures > 1:
            self.running = False
        return np.zeros((224, 256, 3), dtype=np.uint8)

    def apply(self, player, state):
        self.applied.append((player, state))

    def close(self):
        self.running = False


class FakePolicy:
    def decide(self, frame, players, *, game):
        time.sleep(0.001)
        return {
            p: PlayerDecision(
                state=ControllerState.from_buttons(["RIGHT", "B"]),
                movement_confidence=1.0,
                action_confidence=1.0,
                raw={},
            )
            for p in players
        }


def test_engine_runs_inference_without_queueing_frames():
    adapter = FakeAdapter()
    engine = SecondPlayerEngine(adapter, FakePolicy(), [2], 0)
    stats = engine.run()
    assert stats.failures == 0
    assert adapter.captures >= 1
    assert any(player == 2 for player, _ in adapter.applied)
