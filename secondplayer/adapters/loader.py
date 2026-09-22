from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

from ..config import SessionConfig
from ..errors import AdapterError
from .base import EmulatorAdapter


def _module_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if not normalized or not normalized.replace("_", "").isalnum():
        raise AdapterError(f"invalid adapter name: {name!r}")
    return f"secondplayer.adapters.{normalized}"


def load_adapter_module(name: str) -> ModuleType:
    module_name = _module_name(name)
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise AdapterError(f"adapter {name!r} is not installed") from exc
        raise AdapterError(
            f"adapter {name!r} could not be imported because dependency {exc.name!r} is unavailable"
        ) from exc


def create_adapter(
    name: str,
    options: dict[str, Any],
    session: SessionConfig,
    rom: str | Path,
) -> EmulatorAdapter:
    module = load_adapter_module(name)
    factory = getattr(module, "create_adapter", None)
    if not callable(factory):
        raise AdapterError(f"adapter {name!r} does not expose create_adapter(options, session, rom)")
    adapter = factory(dict(options), session, Path(rom))
    if not isinstance(adapter, EmulatorAdapter):
        raise AdapterError(f"adapter {name!r} factory returned {type(adapter).__name__}, expected EmulatorAdapter")
    return adapter
