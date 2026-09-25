"""SecondPlayer Tetris Benchmark v1 (Tetris & Dr. Mario, USA).

Loop: cold boot -> capture -> decode HUD -> Laya decides from pixels ->
apply -> repeat until GAME OVER (two consecutive reads, only after gameplay
was reached) or the 30-minute ceiling. The AI navigates everything itself,
menus included, exactly like product play mode. Writes a JSON report with
final score, survival time, decision count, and inference latency percentiles.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ...controller import NEUTRAL
from ...policy.laya import PlayerDecision
from ...policy.memory import PollGate
from . import decoder as D

log = logging.getLogger(__name__)

ROM_SHA256 = "62cef84f03ea6f3d0a05cc399731da193e01f0aee54e01b13ad08fb98b4a427c"
STATE_PATH = Path(__file__).with_name("bench0.state")
DEFAULT_TIMEOUT_S = 1800.0
GAMEOVER_CONFIRM_S = 1.0


class Policy(Protocol):
    def decide(self, frame, players: list[int], *, game: str) -> dict[int, PlayerDecision]: ...


class BenchmarkAdapter(Protocol):
    """Adapter surface a benchmark run needs (satisfied by MesenCEAdapter)."""

    @property
    def game_name(self) -> str: ...
    def start(self) -> None: ...
    def close(self) -> None: ...
    def capture(self): ...
    def apply(self, player: int, state) -> None: ...
    def wait_frames(self, frames: int) -> None: ...
    def load_state(self, state: bytes) -> None: ...


def percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile over an ascending-sorted list."""
    if not sorted_values:
        return 0.0
    rank = min(len(sorted_values) - 1, max(0, int((pct / 100.0) * len(sorted_values))))
    return float(sorted_values[rank])


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summarize(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": float(sum(ordered) / len(ordered)) if ordered else 0.0,
        "p50": percentile(ordered, 50),
        "p95": percentile(ordered, 95),
        "max": float(ordered[-1]) if ordered else 0.0,
    }


def _read_hud(mask) -> dict[str, Any]:
    return {
        "score": D.read_score(mask),
        "top": D.read_top(mask),
        "lines": D.read_lines(mask),
        "level": D.read_level(mask),
        "stats": D.read_stats(mask),
        "game_over": D.is_game_over(mask),
    }


def _save_snap(snap_dir: Path, polls: int, frame) -> None:
    try:
        from PIL import Image

        snap_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(snap_dir / f"poll-{polls:05d}.png")
    except Exception:
        log.warning("snapshot save failed", exc_info=True)


def run_benchmark(
    adapter: BenchmarkAdapter,
    policy: Policy,
    rom_path: Path,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    min_interval_s: float = 0.0,
    poll_interval_s: float = 0.1,
    from_state: Path | None = None,
    snap_every: int = 0,
    snap_dir: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Play one benchmark run from a cold boot; return the report."""
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    rom_sha = _sha256_file(rom_path)
    if rom_sha != ROM_SHA256:
        raise ValueError(f"ROM hash {rom_sha} does not match pinned Tetris benchmark ROM {ROM_SHA256}")
    if from_state is not None:
        state_bytes = Path(from_state).read_bytes()
        state_sha: str | None = hashlib.sha256(state_bytes).hexdigest()
    else:
        state_bytes = None
        state_sha = None

    infer_ms: list[float] = []
    move_conf: list[float] = []
    act_conf: list[float] = []
    trace: list[list[float | int | None]] = []
    decisions = 0
    failures = 0
    decode_gaps = 0
    polls = 0
    go_first_at: float | None = None
    reached_gameplay = False
    gameplay_at_s: float | None = None
    lines_start: int | None = None
    last_hud: dict[str, Any] = {}
    last_traced_score: int | None = None
    reason = "timeout"
    detail = ""
    gate = PollGate(min_gap_s=max(min_interval_s, 0.0))

    adapter.start()
    try:
        # Warmup: slow policies (model load + first-inference compile) go hot
        # before the timed game starts, else the first pieces fall unplayed.
        # The warmup decision is discarded, never applied; the clock starts
        # after it so warmup time is harness overhead, not game time.
        try:
            warm_frame = adapter.capture()
            policy.decide(warm_frame, [1], game=adapter.game_name)
        except Exception:
            log.exception("policy warmup failed; continuing cold")
        t0 = time.monotonic()
        if state_bytes is not None:
            adapter.load_state(state_bytes)
            # Settle to a frame edge: inputs applied immediately after a load
            # otherwise land +/-1 frame nondeterministically.
            adapter.wait_frames(30)
        deadline = t0 + timeout_s
        next_poll = time.monotonic()
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_poll:
                time.sleep(next_poll - now)
            tick_start = time.monotonic()
            try:
                frame = adapter.capture()
            except Exception as exc:
                failures += 1
                log.warning("capture failed: %s", exc)
                next_poll = tick_start + max(poll_interval_s, 0.1)
                continue
            polls += 1
            if snap_every > 0 and snap_dir is not None and polls % snap_every == 0:
                _save_snap(snap_dir, polls, frame)
            hud = _read_hud(D.white_mask(frame))
            if hud["score"] is None:
                decode_gaps += 1
            else:
                last_hud = hud
                if not reached_gameplay:
                    reached_gameplay = True
                    gameplay_at_s = round(tick_start - t0, 2)
                    lines_start = hud["lines"]
            fire, _ = gate.poll(frame, tick_start)
            if fire or hud["score"] != last_traced_score:
                trace.append([round(tick_start - t0, 2), hud["score"]])
                last_traced_score = hud["score"]
            # Menus can hold enough white pixels to trip the splash detector,
            # so GAME OVER only counts after gameplay was reached, confirmed
            # across reads at least a second apart (a line-clear flash is not).
            if hud["game_over"] and reached_gameplay:
                if go_first_at is None:
                    go_first_at = tick_start
                elif tick_start - go_first_at >= GAMEOVER_CONFIRM_S:
                    reason = "game_over"
                    break
            else:
                go_first_at = None
            if fire:
                try:
                    infer_t0 = time.perf_counter()
                    made = policy.decide(frame, [1], game=adapter.game_name)
                    infer_ms.append((time.perf_counter() - infer_t0) * 1000.0)
                    move_conf.append(float(getattr(made[1], "movement_confidence", 0.0)))
                    act_conf.append(float(getattr(made[1], "action_confidence", 0.0)))
                    adapter.apply(1, made[1].state)
                    decisions += 1
                except Exception:
                    failures += 1
                    log.exception("AI decision failed; releasing controller")
                    try:
                        adapter.apply(1, NEUTRAL)
                    except Exception:
                        pass
            next_poll = tick_start + poll_interval_s
    except KeyboardInterrupt:
        reason = "interrupted"
    except Exception as exc:
        reason = "error"
        detail = str(exc)
        log.exception("benchmark run errored")
    finally:
        try:
            adapter.apply(1, NEUTRAL)
        except Exception:
            pass
        # Final re-read on the (now stable) terminal screen.
        final = dict(last_hud)
        if reason == "game_over":
            for _ in range(5):
                try:
                    hud = _read_hud(D.white_mask(adapter.capture()))
                except Exception:
                    break
                if hud["score"] is not None:
                    final = hud
                    break
                try:
                    adapter.wait_frames(30)
                except Exception:
                    break
        adapter.close()

    duration_s = time.monotonic() - t0
    # The cold-boot menu lottery can land in A-TYPE (lines count up from 0)
    # or B-TYPE (lines count down from 25): the raw display is ambiguous,
    # so infer the mode and report lines actually cleared.
    lines_end = final.get("lines")
    if lines_start == 25:
        mode = "b-type"
    elif lines_start == 0:
        mode = "a-type"
    elif lines_start is not None and lines_end is not None and lines_end != lines_start:
        mode = "b-type" if lines_end < lines_start else "a-type"
    else:
        mode = "unknown"
    if lines_start is not None and lines_end is not None:
        lines_cleared: int | None = abs(lines_end - lines_start)
    else:
        lines_cleared = None
    learning: dict[str, Any] | None = None
    memories = getattr(policy, "memories", None)
    if memories:
        mem = memories.get(1)
        turn_log = getattr(policy, "turn_log", []) or []
        if mem is not None:
            effects: dict[str, int] = {}
            for entry in turn_log:
                key = str(entry.get("effect", "none"))
                effects[key] = effects.get(key, 0) + 1
            learning = {
                "turns": mem.turn,
                "facts": mem.summary()["facts"],
                "tried": dict(sorted(mem.tried.items())),
                "forced": sum(1 for entry in turn_log if entry.get("forced")),
                "effects": effects,
                "log": turn_log,
            }
    report = {
        "benchmark": "tetris-v1",
        "rom_sha256": rom_sha,
        "rom_bytes": rom_path.stat().st_size,
        "state_sha256": state_sha,
        "started_at": started.isoformat(),
        "duration_s": round(duration_s, 2),
        "reason": reason,
        "detail": detail,
        "final": {
            "score": final.get("score"),
            "top": final.get("top"),
            "lines": final.get("lines"),
            "level": final.get("level"),
            "stats": final.get("stats"),
        },
        "decisions": decisions,
        "cadence_hz": round(decisions / duration_s, 3) if duration_s > 0 else 0.0,
        "polls": polls,
        "poll_hz": round(polls / duration_s, 3) if duration_s > 0 else 0.0,
        "snaps": polls // snap_every if snap_every > 0 and snap_dir is not None else 0,
        "reached_gameplay": reached_gameplay,
        "gameplay_at_s": gameplay_at_s,
        "mode": mode,
        "lines_start": lines_start,
        "lines_cleared": lines_cleared,
        "failures": failures,
        "decode_gaps": decode_gaps,
        "infer_ms": _summarize(infer_ms),
        "confidence": {"movement": _summarize(move_conf), "buttons": _summarize(act_conf)},
        "learning": learning,
        "trace": trace,
    }
    if extra:
        report["config"] = extra
    return report


def write_report(report: dict[str, Any], out_dir: Path | None = None) -> Path:
    directory = out_dir or (Path(__file__).with_name("results"))
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"tetris-v1-{stamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
