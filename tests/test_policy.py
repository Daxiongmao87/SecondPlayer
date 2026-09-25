from secondplayer.controller import NEUTRAL
from secondplayer.policy.laya import LayaVisionPolicy


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
