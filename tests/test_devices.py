"""Tests for compute device autodetection (torch is faked; never required)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from secondplayer.errors import ConfigurationError
from secondplayer.platform import devices


GB = 1024**3


class FakeTensor:
    def __add__(self, other):
        return self

    def sum(self):
        return 1


class FakeCuda:
    """Mimics the torch.cuda surface the selector probes."""

    def __init__(self, free=(), fail=(), name="Fake GPU", bf16=False):
        self._free = list(free)
        self._fail = set(fail)
        self._name = name
        self._bf16 = bf16
        self._current = 0

    def is_available(self):
        return True

    def device_count(self):
        return len(self._free)

    def device(self, index):
        outer = self

        class Ctx:
            def __enter__(self):
                outer._current = index

            def __exit__(self, *args):
                return False

        return Ctx()

    def get_device_name(self, index=0):
        return self._name

    def is_bf16_supported(self):
        return self._bf16

    def mem_get_info(self, index):
        return (self._free[index], 32 * GB)


def fake_torch(cuda=False, hip=False, xpu=False, mps=False, bf16=False, name="Fake GPU"):
    if cuda is False:
        cuda_ns = SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            get_device_name=lambda *a: name,
            is_bf16_supported=lambda: bf16,
        )
        zeros = None
    else:
        free, fail = cuda if isinstance(cuda, tuple) else (cuda, ())
        cuda_ns = FakeCuda(free, fail, name, bf16)

        def zeros(n, device=None):
            if cuda_ns._current in cuda_ns._fail:
                raise RuntimeError("no kernel for this GPU")
            return FakeTensor()

    return SimpleNamespace(
        cuda=cuda_ns,
        zeros=zeros if cuda is not False else None,
        xpu=SimpleNamespace(is_available=lambda: xpu) if xpu else None,
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        version=SimpleNamespace(hip="7.0" if hip else None),
    )


def test_no_torch_resolves_cpu(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: None)
    resolved = devices.resolve_device("auto")
    assert (resolved.device, resolved.dtype, resolved.label) == ("cpu", None, "CPU")


def test_nvidia_cuda_with_bf16(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch(cuda=[8 * GB], bf16=True))
    resolved = devices.resolve_device("auto")
    assert resolved.device == "cuda:0"
    assert resolved.dtype == "bf16"
    assert resolved.label.startswith("NVIDIA CUDA")
    assert "8.0GB free" in resolved.reason


def test_nvidia_cuda_without_bf16_stays_fp32(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch(cuda=[8 * GB], bf16=False))
    resolved = devices.resolve_device("auto")
    assert (resolved.device, resolved.dtype) == ("cuda:0", None)


def test_rocm_build_labels_amd(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch(cuda=[8 * GB], hip=True))
    resolved = devices.resolve_device("auto")
    assert resolved.device == "cuda:0"
    assert resolved.label.startswith("AMD ROCm")


def test_intel_xpu_beats_cpu(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch(xpu=True))
    resolved = devices.resolve_device("auto")
    assert (resolved.device, resolved.label) == ("xpu", "Intel XPU")


def test_cuda_beats_xpu(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch(cuda=[8 * GB], xpu=True))
    assert devices.resolve_device("auto").device == "cuda:0"


def test_apple_mps_before_cpu(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch(mps=True))
    assert devices.resolve_device("auto").device == "mps"


def test_explicit_cpu_honored(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch(cuda=[8 * GB]))
    assert devices.resolve_device("cpu").device == "cpu"


def test_explicit_cuda_keeps_index(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch(cuda=[8 * GB, 8 * GB]))
    assert devices.resolve_device("cuda:1").device == "cuda:1"


def test_auto_picks_most_free_cuda(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch(cuda=[2 * GB, 9 * GB, 4 * GB]))
    resolved = devices.resolve_device("auto")
    assert resolved.device == "cuda:1"
    assert "9.0GB free" in resolved.reason


def test_auto_skips_cuda_without_kernels(monkeypatch):
    monkeypatch.setattr(
        devices, "_get_torch", lambda: fake_torch(cuda=([8 * GB, 8 * GB], (0, 1)), xpu=True)
    )
    assert devices.resolve_device("auto").device == "xpu"


def test_unusable_cuda_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch(cuda=([8 * GB], (0,))))
    resolved = devices.resolve_device("auto")
    assert resolved.device == "cpu"
    assert "no GPU backend" in resolved.reason


def test_explicit_bad_cuda_index_falls_back(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch(cuda=[8 * GB]))
    resolved = devices.resolve_device("cuda:9")
    assert resolved.device == "cpu"
    assert "fell back" in resolved.reason


def test_missing_explicit_gpu_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch())
    for pref in ("cuda", "xpu", "mps"):
        resolved = devices.resolve_device(pref)
        assert resolved.device == "cpu"
        assert "fell back" in resolved.reason


def test_unknown_device_is_an_error(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch())
    with pytest.raises(ConfigurationError, match="unknown compute device"):
        devices.resolve_device("tpu")


def test_auto_skips_cuda_below_min_free(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch(cuda=[2 * GB]))
    resolved = devices.resolve_device("auto", min_free_bytes=4 * GB)
    assert resolved.device == "cpu"
    assert "no GPU backend" in resolved.reason
    assert "cuda:0 has 2.0GB free" in resolved.reason
    assert "needs 4.0GB" in resolved.reason
    assert "needs free VRAM" in resolved.reason
    assert "needs CUDA torch" not in resolved.reason


def test_auto_picks_first_cuda_clearing_min_free(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch(cuda=[2 * GB, 9 * GB]))
    resolved = devices.resolve_device("auto", min_free_bytes=4 * GB)
    assert resolved.device == "cuda:1"


def test_bare_cuda_below_min_free_falls_back(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch(cuda=[2 * GB]))
    resolved = devices.resolve_device("cuda", min_free_bytes=4 * GB)
    assert resolved.device == "cpu"
    assert "fell back" in resolved.reason
    assert "needs 4.0GB" in resolved.reason


def test_explicit_cuda_index_ignores_min_free(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch(cuda=[2 * GB]))
    assert devices.resolve_device("cuda:0", min_free_bytes=4 * GB).device == "cuda:0"


def test_cpu_fallback_names_stranded_hardware(monkeypatch):
    monkeypatch.setattr(devices, "_get_torch", lambda: fake_torch())
    monkeypatch.setattr(devices, "_hardware_gpus", lambda: ["Intel Corporation DG2 [Arc A770M]"])
    resolved = devices.resolve_device("auto")
    assert resolved.device == "cpu"
    assert "Arc A770M" in resolved.reason
    assert "XPU torch" in resolved.reason
