from __future__ import annotations

import json
from pathlib import Path

import pytest

from secondplayer.adapters.mesence.adapter import MesenCEAdapter, MesenCEOptions
from secondplayer.config import SessionConfig
from secondplayer.controller import ControllerState
from secondplayer.errors import AdapterError


def make_rom(tmp_path: Path) -> Path:
    rom = tmp_path / "game.sfc"
    rom.write_bytes(b"not-a-real-rom")
    return rom


def test_options_reject_unknown_keys():
    with pytest.raises(AdapterError, match="unknown MesenCE"):
        MesenCEOptions.from_mapping({"made_up": True})


def test_v1_rejects_player_three(tmp_path: Path):
    with pytest.raises(AdapterError, match="players 1 and 2"):
        MesenCEAdapter(
            MesenCEOptions(),
            SessionConfig(players={1: "human", 3: "ai"}),
            make_rom(tmp_path),
        )


def test_prepare_session_copies_and_patches_settings(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "settings.json").write_text(
        json.dumps({
            "Audio": {"MasterVolume": 42},
            "Snes": {"Port1": {"Type": "SnesMouse", "Mapping1": {"A": 123}}},
        }),
        encoding="utf-8",
    )
    root = tmp_path / "session"
    adapter = MesenCEAdapter(
        MesenCEOptions(source_config_home=str(source), session_root=str(root)),
        SessionConfig(players={1: "human", 2: "ai"}),
        make_rom(tmp_path),
    )
    adapter._prepare_session()
    settings = json.loads((root / "home/.config/MesenCE/settings.json").read_text())

    assert settings["Audio"]["MasterVolume"] == 42
    assert settings["Snes"]["Port1"]["Type"] == "SnesController"
    assert settings["Snes"]["Port1"]["Mapping1"]["A"] == 123
    assert settings["Snes"]["Port2"]["Type"] == "SnesController"
    assert settings["Debug"]["ScriptWindow"]["AllowIoOsAccess"] is True
    assert settings["Debug"]["ScriptWindow"]["AllowNetworkAccess"] is True
    assert (root / "secondplayer.lua").is_file()


def test_build_headless_command(tmp_path: Path, monkeypatch):
    adapter = MesenCEAdapter(
        MesenCEOptions(headless=True, xvfb=True, session_root=str(tmp_path / "session")),
        SessionConfig(players={1: "human", 2: "ai"}),
        make_rom(tmp_path),
    )
    adapter._prepare_session()
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr("secondplayer.adapters.mesence.adapter.shutil.which", lambda name: "/usr/bin/xvfb-run")
    cmd = adapter._build_command(Path("/opt/Mesen"))
    assert cmd[:3] == ["/usr/bin/xvfb-run", "-a", "/opt/Mesen"]
    assert "--testrunner" in cmd
    assert str(adapter.rom) in cmd
    assert str(adapter._bridge_path) in cmd


def test_build_graphical_command_is_stock_cli(tmp_path: Path, monkeypatch):
    adapter = MesenCEAdapter(
        MesenCEOptions(headless=False, session_root=str(tmp_path / "session")),
        SessionConfig(players={1: "human", 2: "ai"}),
        make_rom(tmp_path),
    )
    adapter._prepare_session()
    monkeypatch.setenv("DISPLAY", ":0")
    cmd = adapter._build_command(Path("/opt/Mesen"))
    assert cmd == ["/opt/Mesen", "--doNotSaveSettings", str(adapter.rom), str(adapter._bridge_path)]


def test_build_graphical_command_uses_xvfb_without_display(tmp_path: Path, monkeypatch):
    adapter = MesenCEAdapter(
        MesenCEOptions(headless=False, xvfb=True, session_root=str(tmp_path / "session")),
        SessionConfig(players={1: "human", 2: "ai"}),
        make_rom(tmp_path),
    )
    adapter._prepare_session()
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr("secondplayer.adapters.mesence.adapter.shutil.which", lambda name: "/usr/bin/xvfb-run")
    cmd = adapter._build_command(Path("/opt/Mesen"))
    assert cmd[:3] == ["/usr/bin/xvfb-run", "-a", "/opt/Mesen"]
    assert "--doNotSaveSettings" in cmd


def test_controller_mask_matches_bridge_order(tmp_path: Path):
    adapter = MesenCEAdapter(
        MesenCEOptions(), SessionConfig(players={1: "human", 2: "ai"}), make_rom(tmp_path)
    )
    state = ControllerState.from_buttons(["UP", "A", "R", "START"])
    assert adapter._mask(state) == (1 << 0) | (1 << 4) | (1 << 9) | (1 << 10)


def _write_script(tmp_path: Path, name: str, body: str) -> Path:
    script = tmp_path / name
    script.write_text(body, encoding="utf-8")
    script.chmod(0o755)
    return script


def test_failed_start_kills_child_and_removes_temp_session(tmp_path: Path, monkeypatch):
    import os
    import subprocess

    sleeper = _write_script(tmp_path, "sleeper.sh", "#!/bin/sh\nsleep 30\n")
    pids: list[int] = []
    real_popen = subprocess.Popen

    def spy(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        pids.append(proc.pid)
        return proc

    monkeypatch.setattr(subprocess, "Popen", spy)
    adapter = MesenCEAdapter(
        MesenCEOptions(binary=str(sleeper), headless=True, xvfb=False, startup_timeout_s=2),
        SessionConfig(players={1: "human", 2: "ai"}),
        make_rom(tmp_path),
    )
    with pytest.raises(AdapterError, match="timed out"):
        adapter.start()
    assert pids, "expected the fake emulator to be spawned"
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)
    assert adapter.session_root is not None
    assert not adapter.session_root.exists()


def test_exited_child_reports_startup_error(tmp_path: Path):
    exiter = _write_script(tmp_path, "exiter.sh", "#!/bin/sh\nexit 3\n")
    adapter = MesenCEAdapter(
        MesenCEOptions(binary=str(exiter), headless=True, xvfb=False),
        SessionConfig(players={1: "human", 2: "ai"}),
        make_rom(tmp_path),
    )
    with pytest.raises(AdapterError, match="exited before"):
        adapter.start()
    assert adapter.session_root is not None
    assert not adapter.session_root.exists()


def test_pulse_frames_reaches_bridge_env(tmp_path: Path):
    assert MesenCEOptions().pulse_frames == 0
    adapter = MesenCEAdapter(
        MesenCEOptions(pulse_frames=9, session_root=str(tmp_path / "session")),
        SessionConfig(players={1: "human", 2: "ai"}),
        make_rom(tmp_path),
    )
    adapter._prepare_session()
    env = adapter._bridge_env(43111)
    assert env["SECONDPLAYER_PULSE_FRAMES"] == "9"
    assert env["SECONDPLAYER_BRIDGE_PORT"] == "43111"


def test_pulse_frames_rejects_negative():
    with pytest.raises(AdapterError, match="pulse_frames"):
        MesenCEOptions.from_mapping({"pulse_frames": -1})


def test_wait_frames_round_trip(tmp_path: Path):
    import socket
    import threading

    adapter = MesenCEAdapter(
        MesenCEOptions(), SessionConfig(players={1: "human", 2: "ai"}), make_rom(tmp_path)
    )
    local, remote = socket.socketpair()
    adapter._sock = local
    adapter._reader = local.makefile("rb")

    def bridge():
        with remote:
            reader = remote.makefile("rb")
            assert reader.readline().decode() == "WAITFRAMES 1 50\n"
            remote.sendall(b"WAITED 1\n")

    worker = threading.Thread(target=bridge)
    worker.start()
    try:
        adapter.wait_frames(50)
    finally:
        worker.join(timeout=10)
        adapter._reader.close()
    assert not worker.is_alive()


def test_wait_frames_rejects_non_positive(tmp_path: Path):
    adapter = MesenCEAdapter(
        MesenCEOptions(), SessionConfig(players={1: "human", 2: "ai"}), make_rom(tmp_path)
    )
    with pytest.raises(AdapterError, match="at least 1 frame"):
        adapter.wait_frames(0)


def _loopback_adapter(tmp_path: Path):
    import socket

    adapter = MesenCEAdapter(
        MesenCEOptions(), SessionConfig(players={1: "human", 2: "ai"}), make_rom(tmp_path)
    )
    local, remote = socket.socketpair()
    adapter._sock = local
    adapter._reader = local.makefile("rb")
    return adapter, remote


def _run_bridge(remote, fn):
    import contextlib
    import threading

    @contextlib.contextmanager
    def run():
        worker = threading.Thread(target=fn)
        worker.start()
        try:
            yield worker
        finally:
            worker.join(timeout=10)
        assert not worker.is_alive()

    return run()


def _nasty_blob() -> bytes:
    return b"\x00\n\r\xffMESS\x00\n" + bytes(range(256)) * 4 + b"\nSTATE 9 9\n"


def test_save_state_round_trip_is_binary_safe(tmp_path: Path):
    import contextlib

    blob = _nasty_blob()
    adapter, remote = _loopback_adapter(tmp_path)

    def bridge():
        with remote:
            reader = remote.makefile("rb")
            assert reader.readline().decode() == "SAVESTATE 1\n"
            remote.sendall(b"STATE 1 " + str(len(blob)).encode() + b"\n" + blob)

    target = tmp_path / "bench0.state"
    with contextlib.ExitStack() as stack:
        stack.enter_context(_run_bridge(remote, bridge))
        stack.enter_context(remote)
        try:
            assert adapter.save_state(target) == blob
        finally:
            adapter._reader.close()
    assert target.read_bytes() == blob


def test_save_state_rejects_bad_lengths(tmp_path: Path):
    import contextlib

    for header in (b"STATE 1 0\n", b"STATE 1 67108865\n", b"STATE 1 nope\n", b"STATE 1\n"):
        adapter, remote = _loopback_adapter(tmp_path)

        def bridge(header=header):
            with remote:
                reader = remote.makefile("rb")
                reader.readline()
                remote.sendall(header)

        with contextlib.ExitStack() as stack:
            stack.enter_context(_run_bridge(remote, bridge))
            stack.enter_context(remote)
            try:
                with pytest.raises(AdapterError, match="STATE|savestate"):
                    adapter.save_state(tmp_path / "x.state")
            finally:
                adapter._reader.close()


def test_load_state_round_trip_is_binary_safe(tmp_path: Path):
    import contextlib

    blob = _nasty_blob()
    adapter, remote = _loopback_adapter(tmp_path)
    seen: dict[str, bytes] = {}

    def bridge():
        with remote:
            reader = remote.makefile("rb")
            seen["header"] = reader.readline()
            body = b""
            while len(body) < len(blob):
                chunk = reader.read(len(blob) - len(body))
                assert chunk
                body += chunk
            seen["body"] = body
            remote.sendall(b"OK 1\n")

    with contextlib.ExitStack() as stack:
        stack.enter_context(_run_bridge(remote, bridge))
        stack.enter_context(remote)
        try:
            adapter.load_state(blob)
        finally:
            adapter._reader.close()
    assert seen["header"] == b"LOADSTATE 1 " + str(len(blob)).encode() + b"\n"
    assert seen["body"] == blob


def test_load_state_rejects_empty_and_oversize(tmp_path: Path):
    adapter, _ = _loopback_adapter(tmp_path)
    try:
        with pytest.raises(AdapterError, match="empty"):
            adapter.load_state(b"")
        with pytest.raises(AdapterError, match="too large"):
            adapter.load_state(b"\x01" * (64 * 1024 * 1024 + 1))
    finally:
        adapter._reader.close()
