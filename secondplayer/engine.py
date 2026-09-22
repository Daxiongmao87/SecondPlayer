from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass

from .adapters.base import EmulatorAdapter
from .controller import NEUTRAL
from .policy.laya import LayaVisionPolicy, PlayerDecision

log = logging.getLogger(__name__)


@dataclass(slots=True)
class EngineStats:
    decisions: int = 0
    failures: int = 0
    last_inference_ms: float = 0.0


class SecondPlayerEngine:
    """Asynchronous bridge between a stock emulator adapter and local Laya.

    The emulator process never waits on this loop. A single worker performs
    visual inference while the virtual controllers continue holding the last
    successfully applied state.
    """

    def __init__(self, adapter: EmulatorAdapter, policy: LayaVisionPolicy, players: list[int], reaction_ms: int):
        self.adapter = adapter
        self.policy = policy
        self.players = players
        self.reaction_s = reaction_ms / 1000.0
        self.stats = EngineStats()

    @staticmethod
    def _infer(policy: LayaVisionPolicy, frame, players: list[int], game: str):
        t0 = time.perf_counter()
        decisions = policy.decide(frame, players, game=game)
        return decisions, (time.perf_counter() - t0) * 1000.0

    def run(self) -> EngineStats:
        self.adapter.start()
        future: concurrent.futures.Future | None = None
        next_submit = 0.0
        with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="secondplayer-laya") as pool:
            try:
                while self.adapter.is_running():
                    now = time.monotonic()
                    if future is not None and future.done():
                        try:
                            decisions, elapsed_ms = future.result()
                            self.stats.last_inference_ms = elapsed_ms
                            self.stats.decisions += 1
                            for player, decision in decisions.items():
                                assert isinstance(decision, PlayerDecision)
                                self.adapter.apply(player, decision.state)
                            log.info(
                                "decision #%d %.0fms %s",
                                self.stats.decisions,
                                elapsed_ms,
                                {p: d.state.as_text() for p, d in decisions.items()},
                            )
                        except Exception:
                            self.stats.failures += 1
                            log.exception("AI decision failed; releasing AI controllers")
                            for player in self.players:
                                self.adapter.apply(player, NEUTRAL)
                        future = None

                    if future is None and now >= next_submit:
                        try:
                            frame = self.adapter.capture()
                            future = pool.submit(self._infer, self.policy, frame, self.players, self.adapter.game_name)
                            next_submit = now + self.reaction_s
                        except Exception:
                            self.stats.failures += 1
                            log.exception("capture failed")
                            next_submit = now + max(0.1, self.reaction_s)
                    time.sleep(0.005)
            finally:
                for player in self.players:
                    try:
                        self.adapter.apply(player, NEUTRAL)
                    except Exception:
                        pass
                self.adapter.close()
        return self.stats
