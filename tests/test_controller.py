from secondplayer.controller import ControllerState, state_from_choices


def test_opposing_directions_are_sanitized():
    s = ControllerState.from_buttons(["UP", "DOWN", "A"])
    assert s.buttons == frozenset({"A"})


def test_choice_composition_and_keep():
    first = state_from_choices("right", "B+Y")
    assert first.buttons == frozenset({"RIGHT", "B", "Y"})
    second = state_from_choices("keep", "none", first)
    assert second.buttons == frozenset({"RIGHT"})
