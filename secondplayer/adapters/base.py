from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..controller import ControllerState


class EmulatorAdapter(ABC):
    @property
    @abstractmethod
    def game_name(self) -> str: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def is_running(self) -> bool: ...

    @abstractmethod
    def capture(self) -> np.ndarray: ...

    @abstractmethod
    def apply(self, player: int, state: ControllerState) -> None: ...

    @abstractmethod
    def close(self) -> None: ...
