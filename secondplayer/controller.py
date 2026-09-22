from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

SNES_BUTTONS = (
    "UP", "DOWN", "LEFT", "RIGHT",
    "A", "B", "X", "Y", "L", "R", "START", "SELECT",
)

MOVEMENT_CHOICES = (
    "neutral", "up", "down", "left", "right",
    "up-left", "up-right", "down-left", "down-right", "keep",
)

BUTTON_CHOICES = (
    "none", "A", "B", "X", "Y", "L", "R",
    "A+B", "A+Y", "B+Y", "X+Y", "START", "SELECT", "keep",
)

_MOVEMENT_TO_BUTTONS = {
    "neutral": (),
    "up": ("UP",),
    "down": ("DOWN",),
    "left": ("LEFT",),
    "right": ("RIGHT",),
    "up-left": ("UP", "LEFT"),
    "up-right": ("UP", "RIGHT"),
    "down-left": ("DOWN", "LEFT"),
    "down-right": ("DOWN", "RIGHT"),
}


@dataclass(frozen=True, slots=True)
class ControllerState:
    buttons: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_buttons(cls, buttons: Iterable[str]) -> "ControllerState":
        normalized = {str(b).upper() for b in buttons}
        unknown = normalized.difference(SNES_BUTTONS)
        if unknown:
            raise ValueError(f"unknown SNES buttons: {sorted(unknown)}")
        # Opposing D-pad directions are not a valid physical SNES pad state.
        if "UP" in normalized and "DOWN" in normalized:
            normalized.discard("UP")
            normalized.discard("DOWN")
        if "LEFT" in normalized and "RIGHT" in normalized:
            normalized.discard("LEFT")
            normalized.discard("RIGHT")
        return cls(frozenset(normalized))

    @property
    def movement(self) -> frozenset[str]:
        return frozenset(self.buttons.intersection({"UP", "DOWN", "LEFT", "RIGHT"}))

    @property
    def actions(self) -> frozenset[str]:
        return frozenset(self.buttons.difference({"UP", "DOWN", "LEFT", "RIGHT"}))

    def as_text(self) -> str:
        return "+".join(b for b in SNES_BUTTONS if b in self.buttons) or "neutral"


NEUTRAL = ControllerState()


def state_from_choices(
    movement: str,
    action: str,
    previous: ControllerState = NEUTRAL,
) -> ControllerState:
    movement = movement.strip().lower()
    action = action.strip()

    if movement == "keep":
        move_buttons = set(previous.movement)
    else:
        if movement not in _MOVEMENT_TO_BUTTONS:
            raise ValueError(f"unknown movement choice: {movement}")
        move_buttons = set(_MOVEMENT_TO_BUTTONS[movement])

    if action.lower() == "keep":
        action_buttons = set(previous.actions)
    elif action.lower() == "none":
        action_buttons = set()
    else:
        action_buttons = {part.strip().upper() for part in action.split("+") if part.strip()}

    return ControllerState.from_buttons(move_buttons | action_buttons)
