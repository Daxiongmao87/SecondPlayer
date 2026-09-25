"""Tests for benchmark runner helpers (no emulator required)."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from secondplayer.benchmarks.tetris import run as run_mod
from secondplayer.benchmarks.tetris.run import _summarize, percentile, run_benchmark
from secondplayer.controller import NEUTRAL


def test_percentile_nearest_rank():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 50) == 30.0
    assert percentile(values, 95) == 40.0
    assert percentile(values, 0) == 10.0
    assert percentile([], 50) == 0.0


def test_summarize_empty():
    assert _summarize([]) == {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}


def test_run_rejects_unpinned_rom(tmp_path: Path):
    rom = tmp_path / "other.smc"
    rom.write_bytes(b"not the benchmark rom")
    with pytest.raises(ValueError, match="does not match pinned"):
        run_benchmark(object(), object(), rom, timeout_s=1)


class StubLoopAdapter:
    game_name = "tetris"

    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames
        self.i = 0
        self.loaded: bytes | None = None
        self.applied: list[Any] = []

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def capture(self) -> np.ndarray:
        frame = self.frames[self.i % len(self.frames)]
        self.i += 1
        return frame

    def apply(self, player: int, state: Any) -> None:
        self.applied.append((player, state))

    def wait_frames(self, frames: int) -> None:
        pass

    def load_state(self, state: bytes) -> None:
        self.loaded = state


class StubLoopPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, frame: Any, players: list[int], *, game: str) -> dict[int, Any]:
        self.calls += 1
        return {1: SimpleNamespace(state=NEUTRAL)}


def _hud(score: int | None, game_over: bool = False, lines: int | None = 0) -> dict[str, Any]:
    return {
        "score": score,
        "top": 100,
        "lines": lines,
        "level": 0,
        "stats": [0] * 7,
        "game_over": game_over,
    }


@pytest.fixture
def fake_clock(monkeypatch):
    t = [100.0]

    def tick() -> float:
        t[0] += 0.05
        return t[0]

    monkeypatch.setattr(time, "monotonic", tick)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return t


def _pinned_rom(tmp_path: Path, monkeypatch) -> Path:
    rom = tmp_path / "game.smc"
    rom.write_bytes(b"fake rom bytes")
    monkeypatch.setattr(run_mod, "ROM_SHA256", hashlib.sha256(b"fake rom bytes").hexdigest())
    return rom


def test_cold_boot_skips_load_and_ignores_menu_game_over(tmp_path: Path, monkeypatch, fake_clock):
    rom = _pinned_rom(tmp_path, monkeypatch)
    huds = [_hud(None, True)] * 3 + [_hud(0), _hud(0)]
    calls = {"n": 0}

    def scripted(mask: Any) -> dict[str, Any]:
        hud = huds[min(calls["n"], len(huds) - 1)]
        calls["n"] += 1
        return hud

    monkeypatch.setattr(run_mod, "_read_hud", scripted)
    alternating = [np.zeros((32, 32, 3), dtype=np.uint8), np.full((32, 32, 3), 255, dtype=np.uint8)]
    adapter = StubLoopAdapter(alternating)
    report = run_benchmark(adapter, StubLoopPolicy(), rom, timeout_s=2.0, min_interval_s=0.0, poll_interval_s=0.0)
    assert adapter.loaded is None  # cold boot: no savestate loaded
    assert report["state_sha256"] is None
    assert report["reason"] == "timeout"  # pre-gameplay GAME OVER reads ignored
    assert report["reached_gameplay"] is True
    assert report["gameplay_at_s"] is not None
    assert report["final"]["score"] == 0
    assert report["decisions"] == report["polls"]  # every poll changed -> decided


def test_from_state_loads_and_game_over_ends_after_gameplay(tmp_path: Path, monkeypatch, fake_clock):
    rom = _pinned_rom(tmp_path, monkeypatch)
    state_path = tmp_path / "debug.state"
    state_path.write_bytes(b"debug-state")
    huds = [_hud(0), _hud(0, True), _hud(0, True), _hud(0, True)]
    calls = {"n": 0}

    def scripted(mask: Any) -> dict[str, Any]:
        hud = huds[min(calls["n"], len(huds) - 1)]
        calls["n"] += 1
        return hud

    monkeypatch.setattr(run_mod, "_read_hud", scripted)
    monkeypatch.setattr(run_mod, "GAMEOVER_CONFIRM_S", 0.2)
    alternating = [np.zeros((32, 32, 3), dtype=np.uint8), np.full((32, 32, 3), 255, dtype=np.uint8)]
    adapter = StubLoopAdapter(alternating)
    report = run_benchmark(
        adapter, StubLoopPolicy(), rom, timeout_s=30.0, min_interval_s=0.0, poll_interval_s=0.0,
        from_state=state_path,
    )
    assert adapter.loaded == b"debug-state"
    assert report["state_sha256"] == hashlib.sha256(b"debug-state").hexdigest()
    assert report["reason"] == "game_over"
    assert report["reached_gameplay"] is True


def test_static_polls_skip_inference_in_loop(tmp_path: Path, monkeypatch, fake_clock):
    rom = _pinned_rom(tmp_path, monkeypatch)
    monkeypatch.setattr(run_mod, "_read_hud", lambda mask: _hud(0))
    adapter = StubLoopAdapter([np.zeros((32, 32, 3), dtype=np.uint8)])
    report = run_benchmark(adapter, StubLoopPolicy(), rom, timeout_s=2.0, min_interval_s=0.0, poll_interval_s=0.0)
    assert report["polls"] > 5
    assert report["decisions"] < report["polls"]  # frozen screen: heartbeat only


def test_mode_inference_a_type_counts_up(tmp_path: Path, monkeypatch, fake_clock):
    rom = _pinned_rom(tmp_path, monkeypatch)
    huds = [_hud(None), _hud(0, lines=0), _hud(100, lines=2)]
    calls = {"n": 0}

    def scripted(mask: Any) -> dict[str, Any]:
        hud = huds[min(calls["n"], len(huds) - 1)]
        calls["n"] += 1
        return hud

    monkeypatch.setattr(run_mod, "_read_hud", scripted)
    alternating = [np.zeros((32, 32, 3), dtype=np.uint8), np.full((32, 32, 3), 255, dtype=np.uint8)]
    report = run_benchmark(
        StubLoopAdapter(alternating), StubLoopPolicy(), rom,
        timeout_s=2.0, min_interval_s=0.0, poll_interval_s=0.0,
    )
    assert report["reached_gameplay"] is True
    assert report["mode"] == "a-type"
    assert report["lines_start"] == 0
    assert report["lines_cleared"] == 2


def test_mode_inference_b_type_counts_down(tmp_path: Path, monkeypatch, fake_clock):
    rom = _pinned_rom(tmp_path, monkeypatch)
    huds = [_hud(None), _hud(0, lines=25), _hud(0, lines=25), _hud(50, lines=22)]
    calls = {"n": 0}

    def scripted(mask: Any) -> dict[str, Any]:
        hud = huds[min(calls["n"], len(huds) - 1)]
        calls["n"] += 1
        return hud

    monkeypatch.setattr(run_mod, "_read_hud", scripted)
    alternating = [np.zeros((32, 32, 3), dtype=np.uint8), np.full((32, 32, 3), 255, dtype=np.uint8)]
    report = run_benchmark(
        StubLoopAdapter(alternating), StubLoopPolicy(), rom,
        timeout_s=2.0, min_interval_s=0.0, poll_interval_s=0.0,
    )
    assert report["mode"] == "b-type"
    assert report["lines_start"] == 25
    assert report["lines_cleared"] == 3


def test_snap_every_saves_poll_frames(tmp_path: Path, monkeypatch, fake_clock):
    rom = _pinned_rom(tmp_path, monkeypatch)
    monkeypatch.setattr(run_mod, "_read_hud", lambda mask: _hud(0))
    alternating = [np.zeros((32, 32, 3), dtype=np.uint8), np.full((32, 32, 3), 255, dtype=np.uint8)]
    adapter = StubLoopAdapter(alternating)
    snap_dir = tmp_path / "snaps"
    report = run_benchmark(
        adapter, StubLoopPolicy(), rom, timeout_s=2.0, min_interval_s=0.0, poll_interval_s=0.0,
        snap_every=2, snap_dir=snap_dir,
    )
    assert report["snaps"] == report["polls"] // 2
    assert len(list(snap_dir.glob("poll-*.png"))) == report["snaps"]
