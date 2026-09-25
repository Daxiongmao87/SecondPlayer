"""Game-agnostic episodic memory for System-1 (typed-decision) policies.

Laya/Jev-style models are stateless: each call gets ``(state, questions)``
and returns probabilities. Historical context therefore lives HERE, in the
harness, and is rendered into every call's state. Attribution is computed
harness-side from (button pressed, observed motion): the model reads memory,
it does not write it.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def fit_window(
    n_available: int,
    text_tokens: int,
    *,
    max_len: int = 1024,
    head_len: int = 256,
    zero_prefix: int = 3,
    per_image: int = 67,
) -> int:
    """How many recent frames fit in context next to ``text_tokens`` of state.

    Layout: image prefix + state text + question head. Always keeps at least
    the current frame; laya truncates overflowing state text itself.
    """
    room = max_len - head_len - zero_prefix - text_tokens
    return max(1, min(n_available, room // per_image))


@dataclass(slots=True)
class PollGate:
    """Fire inference only when the scene changed (plus a frozen-screen heartbeat).

    The loop captures at the poll rate; this gate diffs against the previous
    capture and fires at most every ``min_gap_s``. A fully static screen still
    fires every ``heartbeat_s`` so stuck recovery can act on pauses and menus.
    """

    min_gap_s: float = 0.1
    heartbeat_s: float = 1.0
    last_frame: np.ndarray | None = None
    last_fire_at: float = 0.0
    polls: int = 0
    fires: int = 0

    def poll(self, frame: np.ndarray, now: float) -> tuple[bool, str]:
        observed = FrameDiff.summarize(self.last_frame, frame)
        self.last_frame = frame.copy()
        self.polls += 1
        if observed.startswith("static") and now - self.last_fire_at < self.heartbeat_s:
            return False, observed
        if now - self.last_fire_at < self.min_gap_s:
            return False, observed
        self.last_fire_at = now
        self.fires += 1
        return True, observed

_SECTORS = ("top", "middle", "bottom")
_COLUMNS = ("left", "center", "right")


class FrameDiff:
    """Pixel-level change summary between two frames. No game knowledge."""

    @staticmethod
    def summarize(prev: np.ndarray | None, frame: np.ndarray) -> str:
        if prev is None or prev.shape != frame.shape:
            return "first screen, nothing to compare yet"
        diff = np.abs(frame.astype(int) - prev.astype(int)).sum(axis=2)
        changed = diff > 24
        frac = float(changed.mean())
        if frac < 0.0005:
            return "static, nothing moved"
        if frac > 0.45:
            return "full-screen flip (menu change, death, or transition?)"
        h, w = changed.shape
        strengths: list[tuple[float, str]] = []
        for i, row in enumerate(_SECTORS):
            for j, col in enumerate(_COLUMNS):
                sector = changed[i * h // 3 : (i + 1) * h // 3, j * w // 3 : (j + 1) * w // 3]
                strengths.append((float(sector.mean()), f"{row}-{col}" if col != "center" else row))
        strengths.sort(reverse=True)
        mag = "small" if frac < 0.01 else ("medium" if frac < 0.06 else "large")
        where = ", ".join(name for strength, name in strengths[:2] if strength > 0.002)
        return f"motion {where or 'scattered'} ({mag})"


@dataclass
class _Fact:
    count: int = 0
    conf_sum: float = 0.0


def _canon_motion(observed: str) -> tuple[str, frozenset[str]]:
    """Canonical motion shape: (magnitude, location set), order-free.

    Gravity reads as "motion bottom (small)" on one turn and "motion
    bottom-left, bottom (small)" on the next; the canonical shapes overlap,
    so run-level autonomy tracking treats them as the same motion.
    """
    if not observed.startswith("motion "):
        return observed, frozenset()
    body, _, mag = observed[len("motion ") :].partition(" (")
    places = frozenset(part.strip() for part in body.split(",") if part.strip())
    return mag.strip(") "), places


@dataclass(slots=True)
class EpisodicMemory:
    """Per-player cause-effect memory: one entry per decision turn."""

    memory_turns: int = 4
    max_facts: int = 12
    static_limit: int = 5

    SWEEP: tuple[str, ...] = ("LEFT", "RIGHT", "UP", "DOWN", "A", "B", "START", "SELECT")
    # Stuck recovery may also reach menu buttons: a static screen is often a
    # pause or a menu, and only START/SELECT can advance those.
    STUCK_POOL: tuple[str, ...] = ("LEFT", "RIGHT", "UP", "DOWN", "A", "B", "START", "SELECT")

    turn: int = 0
    last_action: str | None = None
    tried: Counter = field(default_factory=Counter)
    facts: dict[tuple[str, str], _Fact] = field(default_factory=dict)
    recent: deque = field(default_factory=deque)
    static_streak: int = 0
    # Run-level autonomy ledger: canonical motion shape -> buttons seen with it.
    # Unlike a recent window, this cannot be reset by fixating on one button.
    canon_buttons: dict[tuple[str, frozenset[str]], set[str]] = field(default_factory=dict)

    def note_observed(self, observed: str) -> None:
        if observed.startswith("static"):
            self.static_streak += 1
        else:
            self.static_streak = 0

    def sweep_action(self) -> str | None:
        """First turns: press each controller button once, in order."""
        if self.turn < len(self.SWEEP):
            return self.SWEEP[self.turn]
        return None

    def stuck_action(self) -> str | None:
        """Screen frozen for a while: retry the least-tried button, menus included."""
        if self.static_streak >= self.static_limit:
            return min(self.STUCK_POOL, key=lambda b: self.tried[b])
        return None

    def attribute(self, last_action: str | None, observed: str) -> str | None:
        """Harness-side cause-effect: what did last_action do, given observed?

        A directional button whose direction word appears in the motion keeps
        its moved-* claim. Anything else is checked against the run-level
        autonomy ledger: motion of the same size in an overlapping region,
        previously seen under two or more distinct OTHER buttons, is
        autonomous (gravity, timers, attract loops) and reported as ambient.
        The ledger is run-level so fixating on one button cannot reset it.
        """
        if last_action is None or observed.startswith("first screen"):
            return None
        if observed.startswith("static"):
            return "nothing"
        if observed.startswith("full-screen flip"):
            return "menu"
        direction = None
        for button in ("LEFT", "RIGHT", "UP", "DOWN"):
            if button in last_action.split("+"):
                direction = button
                break
        if direction is not None:
            want = {"LEFT": "left", "RIGHT": "right", "UP": "top", "DOWN": "bottom"}[direction]
            if want in observed:
                moved = {"LEFT": "moved-left", "RIGHT": "moved-right", "UP": "moved-up", "DOWN": "moved-down"}
                return moved[direction]
        mag, places = _canon_motion(observed)
        others: set[str] = set()
        for (seen_mag, seen_places), buttons in self.canon_buttons.items():
            if seen_mag == mag and seen_places & places:
                others |= buttons
        self.canon_buttons.setdefault((mag, places), set()).add(last_action)
        if len(others - {last_action}) >= 2:
            return "ambient"
        if direction is not None:
            return "unsure"
        return "acted"

    def record(self, action_text: str, effect: str | None, conf: float = 1.0) -> None:
        self.tried[action_text] += 1
        # The effect describes the prev->current transition, which was caused
        # by last_action -- not by the action being applied now.
        if effect is not None and self.last_action is not None:
            # Non-claims stay in the recent log only: ambient motion belongs
            # to no button, and unsure is the absence of a claim. Facts are
            # causal claims (moved-*, acted, nothing, menu) or nothing.
            if effect not in ("ambient", "unsure"):
                fact = self.facts.setdefault((self.last_action, effect), _Fact())
                fact.count += 1
                fact.conf_sum += conf
            self.recent.append((self.turn, self.last_action, effect, round(conf, 2)))
            while len(self.recent) > self.memory_turns:
                self.recent.popleft()
        self.last_action = action_text
        self.turn += 1

    def render(self, observed: str) -> dict[str, Any]:
        ranked = sorted(
            self.facts.items(), key=lambda kv: (kv[1].count, kv[1].conf_sum), reverse=True
        )
        facts = "; ".join(
            f"{button}->{effect} ({fact.count}x {fact.conf_sum / fact.count:.1f})"
            for (button, effect), fact in ranked[: self.max_facts]
        )
        recent = "; ".join(
            f"t{t} {action}: {effect} ({conf})" for t, action, effect, conf in self.recent
        )
        return {
            "turn": self.turn,
            "phase": "explore" if self.turn < len(self.SWEEP) else "exploit",
            "last_action": self.last_action or "none yet",
            "observed": observed,
            "memory": facts or "nothing learned yet",
            "recent": recent or "no turns yet",
        }

    def summary(self) -> dict[str, Any]:
        return {
            "turns": self.turn,
            "tried": dict(self.tried),
            "facts": {
                f"{button}->{effect}": {"count": fact.count, "conf": fact.conf_sum / fact.count}
                for (button, effect), fact in sorted(self.facts.items())
            },
        }
