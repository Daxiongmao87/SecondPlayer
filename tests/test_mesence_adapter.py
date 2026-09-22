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
    monkeypatch.setattr("secondplayer.adapters.mesence.adapter.shutil.which", lambda name: "/usr/bin/xvfb-run")
    cmd = adapter._build_command(Path("/opt/Mesen"))
    assert cmd[:3] == ["/usr/bin/xvfb-run", "-a", "/opt/Mesen"]
    assert "--testrunner" in cmd
    assert str(adapter.rom) in cmd
    assert str(adapter._bridge_path) in cmd


def test_build_graphical_command_is_stock_cli(tmp_path: Path):
    adapter = MesenCEAdapter(
        MesenCEOptions(headless=False, session_root=str(tmp_path / "session")),
        SessionConfig(players={1: "human", 2: "ai"}),
        make_rom(tmp_path),
    )
    adapter._prepare_session()
    cmd = adapter._build_command(Path("/opt/Mesen"))
    assert cmd == ["/opt/Mesen", "--doNotSaveSettings", str(adapter.rom), str(adapter._bridge_path)]


def test_controller_mask_matches_bridge_order(tmp_path: Path):
    adapter = MesenCEAdapter(
        MesenCEOptions(), SessionConfig(players={1: "human", 2: "ai"}), make_rom(tmp_path)
    )
    state = ControllerState.from_buttons(["UP", "A", "R", "START"])
    assert adapter._mask(state) == (1 << 0) | (1 << 4) | (1 << 9) | (1 << 10)
