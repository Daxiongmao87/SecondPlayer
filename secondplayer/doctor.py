from __future__ import annotations

from dataclasses import dataclass

from .adapters.loader import load_adapter_module
from .config import AppConfig
from .platform.devices import resolve_device


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str


def run_checks(cfg: AppConfig) -> list[Check]:
    checks: list[Check] = []

    try:
        import laya  # noqa: F401
        checks.append(Check("Laya", True, "importable"))
    except Exception as exc:
        checks.append(Check("Laya", False, str(exc)))

    try:
        compute = resolve_device(cfg.runtime.device)
        checks.append(Check("compute", True, f"{compute.label} ({compute.reason})"))
    except Exception as exc:
        checks.append(Check("compute", False, str(exc)))

    try:
        module = load_adapter_module(cfg.adapter.name)
        checks.append(Check("adapter", True, cfg.adapter.name))
        doctor = getattr(module, "doctor", None)
        if callable(doctor):
            for result in doctor(dict(cfg.adapter.options), cfg.session):
                checks.append(Check(str(result.name), bool(result.ok), str(result.detail)))
    except Exception as exc:
        checks.append(Check("adapter", False, f"{cfg.adapter.name}: {exc}"))

    return checks
