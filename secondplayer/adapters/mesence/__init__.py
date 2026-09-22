from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ...config import SessionConfig
from .adapter import MesenCEAdapter, MesenCEOptions


def create_adapter(options: dict[str, Any], session: SessionConfig, rom):
    return MesenCEAdapter(MesenCEOptions.from_mapping(options), session, rom)


def doctor(options: dict[str, Any], session: SessionConfig):
    parsed = MesenCEOptions.from_mapping(options)
    return [SimpleNamespace(name=name, ok=ok, detail=detail) for name, ok, detail in MesenCEAdapter.doctor(parsed, session)]


__all__ = ["MesenCEAdapter", "MesenCEOptions", "create_adapter", "doctor"]
