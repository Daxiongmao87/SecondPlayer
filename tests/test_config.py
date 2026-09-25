from pathlib import Path

from secondplayer.config import RuntimeConfig, load_config


def test_player_ids_are_not_limited_by_core(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text('[session.players]\n1="human"\n7="ai"\n', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.session.players[1] == "human"
    assert cfg.session.players[7] == "ai"
    assert cfg.session.ai_players == [7]


def test_adapter_options_remain_opaque(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[adapter]\nname="mesence"\n[adapter.options]\nbinary="custom-mesen"\nbridge_port=43111\n'
        '[session.players]\n1="human"\n2="ai"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.adapter.name == "mesence"
    assert cfg.adapter.options == {"binary": "custom-mesen", "bridge_port": 43111}


def test_runtime_poll_and_window_fields(tmp_path: Path):
    assert RuntimeConfig().poll_ms == 100
    assert RuntimeConfig().window_frames == 6
    p = tmp_path / "c.toml"
    p.write_text(
        '[runtime]\npoll_ms = 50\nwindow_frames = 3\n[session.players]\n1 = "ai"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.runtime.poll_ms == 50
    assert cfg.runtime.window_frames == 3
