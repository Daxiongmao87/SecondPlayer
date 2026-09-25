"""Compute device autodetection for local model backends.

Resolution order for ``auto``: NVIDIA CUDA -> AMD ROCm (via the CUDA API on
ROCm torch builds) -> Intel XPU -> Apple MPS -> CPU. CUDA devices must pass a
tiny compute probe (a visible GPU the torch build cannot run kernels on does
not count); among usable ones ``auto`` picks the most free memory. Callers
that know their footprint pass ``min_free_bytes`` and CUDA devices below it
are skipped as if unusable, so a crowded GPU falls back to CPU instead of
OOMing mid-load. An explicit device that is unavailable falls back to CPU
with the reason recorded; only an unknown device name is an error. ``torch``
itself is optional: without it everything resolves to CPU so config parsing,
doctor, and unit tests work anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ConfigurationError

_KNOWN_EXPLICIT = ("cpu", "cuda", "xpu", "mps")


@dataclass(frozen=True, slots=True)
class Resolution:
    device: str
    dtype: str | None
    label: str
    reason: str


def _get_torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def _hardware_gpus() -> list[str]:
    """Best-effort GPU hardware names via lspci. Never raises; [] when unknown."""
    try:
        import shutil
        import subprocess

        lspci = shutil.which("lspci")
        if not lspci:
            return []
        out = subprocess.run([lspci], capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.splitlines():
            low = line.lower()
            if "vga" in low or "3d controller" in low or "display controller" in low:
                _, _, desc = line.partition(": ")
                gpus.append(desc.strip() or line.strip())
        return gpus
    except Exception:
        return []


def _hardware_note(need: str | None = None) -> str:
    gpus = _hardware_gpus()
    if not gpus:
        return ""
    if need is None:
        joined = ", ".join(gpus).lower()
        if "nvidia" in joined:
            need = "needs CUDA torch"
        elif "intel" in joined:
            need = "needs XPU torch + Level Zero"
        elif "amd" in joined or "radeon" in joined:
            need = "needs ROCm torch"
        else:
            need = "needs a matching torch build"
    return f" (hardware present: {', '.join(gpus)}; {need})"


def _probe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return False


def _is_hip(torch) -> bool:
    return bool(getattr(getattr(torch, "version", None), "hip", None))


def _gpu_name(torch, index: int = 0) -> str | None:
    try:
        return str(torch.cuda.get_device_name(index))
    except Exception:
        return None


def _cuda_free_if_usable(torch, index: int) -> int | None:
    """Free bytes on a CUDA device, or None when no kernel runs there."""
    try:
        with torch.cuda.device(index):
            check = torch.zeros(1, device="cuda") + 1
            if int(check.sum()) != 1:
                return None
            free, _ = torch.cuda.mem_get_info(index)
        return int(free)
    except Exception:
        return None


def _usable_cuda_devices(torch) -> list[tuple[int, int]]:
    """(index, free bytes) for each CUDA device that runs kernels."""
    try:
        if not torch.cuda.is_available():
            return []
        count = torch.cuda.device_count()
    except Exception:
        return []
    usable = []
    for index in range(count or 0):
        free = _cuda_free_if_usable(torch, index)
        if free is not None:
            usable.append((index, free))
    return usable


def _cuda_bf16(torch) -> bool:
    return bool(_probe(getattr(torch.cuda, "is_bf16_supported", lambda: False)))


def _gb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.1f}GB"


def resolve_device(preference: str | None = None, min_free_bytes: int = 0) -> Resolution:
    pref = (preference or "auto").strip().lower()
    torch = _get_torch()

    usable_cuda = _usable_cuda_devices(torch) if torch else []
    small: list[tuple[int, int]] = []
    if min_free_bytes > 0 and pref in ("auto", "cuda"):
        # An explicitly indexed cuda:N is honored as an operator override;
        # automatic picks must fit the model.
        small = [(i, f) for i, f in usable_cuda if f < min_free_bytes]
        usable_cuda = [(i, f) for i, f in usable_cuda if f >= min_free_bytes]
    xpu_mod = getattr(torch, "xpu", None) if torch else None
    xpu_ok = bool(xpu_mod) and bool(_probe(xpu_mod.is_available))
    mps_mod = getattr(getattr(torch, "backends", None), "mps", None) if torch else None
    mps_ok = bool(mps_mod) and bool(_probe(mps_mod.is_available))

    if pref not in ("auto", *_KNOWN_EXPLICIT) and not pref.startswith("cuda:"):
        raise ConfigurationError(
            f"unknown compute device {preference!r}: expected auto, cpu, cuda, xpu, or mps"
        )

    note = _hardware_note()
    if pref != "auto":
        if pref == "cpu" or not torch:
            return Resolution("cpu", None, "CPU", f"explicit {pref}" if torch else "torch unavailable")
        if pref.startswith("cuda"):
            if ":" in pref:
                try:
                    wanted = int(pref.split(":", 1)[1])
                except ValueError:
                    wanted = -1
                hit = next((f for i, f in usable_cuda if i == wanted), None)
                if hit is not None:
                    return _cuda_resolution(torch, wanted, hit, f"explicit {pref}")
                return Resolution("cpu", None, "CPU", f"{pref} unusable, fell back to CPU{note}")
            if usable_cuda:
                index, free = max(usable_cuda, key=lambda pair: pair[1])
                return _cuda_resolution(torch, index, free, "explicit cuda")
            if small:
                skipped = "; ".join(f"cuda:{i} has {_gb(f)} free" for i, f in small)
                return Resolution(
                    "cpu",
                    None,
                    "CPU",
                    f"cuda GPUs too small ({skipped}, needs {_gb(min_free_bytes)}), "
                    f"fell back to CPU{_hardware_note('needs free VRAM')}",
                )
            return Resolution("cpu", None, "CPU", f"cuda unavailable, fell back to CPU{note}")
        if pref == "xpu":
            if xpu_ok:
                return Resolution("xpu", None, "Intel XPU", "explicit xpu")
            return Resolution("cpu", None, "CPU", f"xpu unavailable, fell back to CPU{note}")
        if pref == "mps":
            if mps_ok:
                return Resolution("mps", None, "Apple MPS", "explicit mps")
            return Resolution("cpu", None, "CPU", f"mps unavailable, fell back to CPU{note}")

    if usable_cuda:
        index, free = max(usable_cuda, key=lambda pair: pair[1])
        return _cuda_resolution(torch, index, free, "auto-detected")
    if xpu_ok:
        return Resolution("xpu", None, "Intel XPU", "auto-detected")
    if mps_ok:
        return Resolution("mps", None, "Apple MPS", "auto-detected")
    base = "no GPU backend available" if torch else "torch unavailable"
    if small:
        skipped = "; ".join(f"cuda:{i} has {_gb(f)} free" for i, f in small)
        base += f" ({skipped}, needs {_gb(min_free_bytes)})"
        return Resolution("cpu", None, "CPU", base + _hardware_note("needs free VRAM"))
    return Resolution("cpu", None, "CPU", base + note)


def _cuda_resolution(torch, index: int, free_bytes: int, why: str) -> Resolution:
    if _is_hip(torch):
        name = _gpu_name(torch, index)
        label = f"AMD ROCm ({name})" if name else "AMD ROCm"
    else:
        name = _gpu_name(torch, index)
        label = f"NVIDIA CUDA ({name})" if name else "NVIDIA CUDA"
    dtype = "bf16" if _cuda_bf16(torch) else None
    gb = free_bytes / (1024**3)
    return Resolution(f"cuda:{index}", dtype, label, f"{why} ({gb:.1f}GB free)")
