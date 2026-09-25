import sys
import types

from secondplayer.config import RuntimeConfig
from secondplayer.controller import NEUTRAL
from secondplayer.policy.laya import LayaVisionPolicy


class _FakeLayaModule(types.ModuleType):
    def __init__(self):
        super().__init__("laya")
        self.calls: list[dict] = []
        self.agent = types.SimpleNamespace(cfg={}, source={"id": "r", "revision": "resolved-sha"})

    def load_vlm(self, model, **kwargs):
        self.calls.append({"model": model, **kwargs})
        return self.agent


def test_player1_leads_shared_menus():
    for kind in ("movement", "buttons"):
        instructions = LayaVisionPolicy._question(1, kind, NEUTRAL)["instructions"]
        assert "SNES Player 1" in instructions
        assert "main player" in instructions
        assert "lead shared menu navigation" in instructions


def test_other_players_defer_shared_menus_to_player1():
    for player in (2, 3):
        for kind in ("movement", "buttons"):
            instructions = LayaVisionPolicy._question(player, kind, NEUTRAL)["instructions"]
            assert f"SNES Player {player}" in instructions
            assert "not the main player" in instructions
            assert "let Player 1 drive shared menus" in instructions
            assert "strictly for you" in instructions


def _load_with_fake_laya(monkeypatch, **cfg_kwargs):
    fake = _FakeLayaModule()
    monkeypatch.setitem(sys.modules, "laya", fake)
    cfg = RuntimeConfig(model="org/repo", device="cpu", **cfg_kwargs)
    policy = LayaVisionPolicy(cfg)
    policy.load()
    return policy, fake


def test_load_forwards_model_revision(monkeypatch):
    _, fake = _load_with_fake_laya(monkeypatch, model_revision="0b6228f")
    assert fake.calls and fake.calls[0]["revision"] == "0b6228f"
    assert fake.calls[0]["model"] == "org/repo"


def test_load_omits_blank_revision(monkeypatch):
    _, fake = _load_with_fake_laya(monkeypatch)
    assert fake.calls and "revision" not in fake.calls[0]


def test_model_info_reports_provenance(monkeypatch):
    policy, _ = _load_with_fake_laya(monkeypatch, model_revision="requested-sha")
    info = policy.model_info()
    assert info["repo"] == "org/repo"
    assert info["revision_requested"] == "requested-sha"
    assert info["revision_resolved"] == "resolved-sha"
    sha = info["laya_code_sha"]
    assert sha is None or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha))
    assert "params" in info
