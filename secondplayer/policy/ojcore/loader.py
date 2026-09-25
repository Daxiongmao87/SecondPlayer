"""In-process model loading for the baked-in readout.

``quant`` selects the backend: ``none`` (unquantized baseline), ``quanto_qint4``
(pre-quantized safetensors dir), ``torchao_int4`` (quantize-at-load),
``awq`` (pre-quantized AWQ checkpoint). Unknown names fail before anything
loads; missing libraries fail loud with the package name.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ...errors import ConfigurationError, ModelError
from ...platform.devices import Resolution, resolve_device

log = logging.getLogger(__name__)

BASELINE_QUANT = "none"

# Automatic GPU picks must clear this. The 4B qint4 decider peaks at 2.79GB
# and the captioner shares its device (measured combined decide peak ~3.6GB),
# so anything below a 4GB-class GPU falls back to CPU instead of OOMing.
_MIN_GPU_FREE_BYTES = 4 * 1024**3


def load_tokenizer(source: str, offline: bool = False) -> Any:
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import transformers
    except ImportError as exc:
        raise ModelError(f"baked-in readout needs transformers: {exc}") from exc
    try:
        return transformers.AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    except Exception as exc:
        raise ModelError(f"failed to load tokenizer {source!r}: {exc}") from exc


QUANT_BACKENDS = ("none", "quanto_qint4", "torchao_int4", "awq")


def _place(model: Any, compute: Resolution, torch: Any) -> Any:
    try:
        from quanto import QTensor

        fixed_types: tuple[type, ...] = (QTensor,)
    except ImportError:
        fixed_types = ()
    params = list(model.parameters())
    buffers = list(model.buffers())
    # Quantized storage fixes the math dtype (dequantized output matches the
    # file's float tensors); casting only the float side breaks matmul dtype
    # consistency, so a model containing QTensors moves device but keeps dtype.
    quantized = any(isinstance(t, fixed_types) for t in params)
    model.to(compute.device)
    if not quantized:
        dtype = getattr(torch, compute.dtype) if compute.dtype else None
        if dtype is None:
            dtype = torch.float32 if compute.device == "cpu" else torch.bfloat16
        for tensor in params + buffers:
            if tensor.is_floating_point():
                tensor.data = tensor.data.to(dtype)
    model.eval()
    return model


def _load_baseline(source: str, compute: Resolution) -> Any:
    import torch
    import transformers

    dtype = getattr(torch, compute.dtype) if compute.dtype else None
    if dtype is None:
        dtype = torch.float32 if compute.device == "cpu" else torch.bfloat16
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            source,
            dtype=dtype,
            device_map={"": compute.device},
            trust_remote_code=False,
        )
        model.eval()
    except Exception as exc:
        raise ModelError(f"failed to load model {source!r} on {compute.device}: {exc}") from exc
    return model


def _file_float_dtype(weights: str, torch: Any) -> Any:
    """Read the float dtype anchor from a quanto safetensors header (no torch load)."""
    import json
    import struct

    try:
        with open(weights, "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
    except (OSError, ValueError) as exc:
        raise ModelError(f"cannot read quanto header {weights!r}: {exc}") from exc
    names = {"F32": "float32", "BF16": "bfloat16", "F16": "float16"}
    entry = header.get("model.embed_tokens._scale", {})
    name = names.get(entry.get("dtype", ""), "")
    if not name:
        for meta in header.values():
            if isinstance(meta, dict) and meta.get("dtype") in names:
                name = names[meta["dtype"]]
                break
    return getattr(torch, name or "float32")


def _load_quanto(source: str, compute: Resolution) -> Any:
    try:
        import torch
        import transformers
        from quanto import freeze, qint4, quantize, safe_load

        from .embed import quantize_embeddings
    except ImportError as exc:
        raise ModelError(f"quanto backend needs torch+transformers+quanto: {exc}") from exc
    weights = os.path.join(source, "model.safetensors")
    if not os.path.isfile(weights):
        raise ModelError(f"quanto backend needs a quanto dir with model.safetensors: {source!r}")
    try:
        config = transformers.AutoConfig.from_pretrained(source, trust_remote_code=False)
        # Build the arch in the file's float dtype so the load casts neither
        # side (quantized scales keep file dtype either way).
        model = transformers.AutoModelForCausalLM.from_config(
            config, dtype=_file_float_dtype(weights, torch), trust_remote_code=False
        )
        quantize(model, weights=qint4, activations=None)
        quantize_embeddings(model)
        model.load_state_dict(safe_load(weights))
    except Exception as exc:
        raise ModelError(f"failed to load quanto model {source!r}: {exc}") from exc
    model = _place(model, compute, torch)
    freeze(model)
    from .scoring import HEAD_ROWS_FILE

    if os.path.isfile(os.path.join(source, HEAD_ROWS_FILE)):
        # The subset readout serves the slot rows from the sidecar, so the
        # 248k-row int4 head (~350MB on 4B) is dead weight; drop it. Without
        # a sidecar the head stays for the live-slice fallback.
        model.lm_head = None  # type: ignore[assignment]
        log.info("ojcore dropped lm_head; slot rows come from %s", HEAD_ROWS_FILE)
    return model


def _load_torchao(source: str, compute: Resolution) -> Any:
    try:
        import torch
        import transformers
        from torchao.quantization import Int4WeightOnlyConfig, quantize_
    except ImportError as exc:
        raise ModelError(f"torchao backend needs torch+transformers+torchao: {exc}") from exc
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(source, trust_remote_code=False)
        quantize_(model, Int4WeightOnlyConfig())
    except Exception as exc:
        raise ModelError(f"failed to quantize model {source!r} with torchao: {exc}") from exc
    return _place(model, compute, torch)


def _load_awq(source: str, compute: Resolution) -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise ModelError(f"awq backend needs transformers: {exc}") from exc
    try:
        import awq  # noqa: F401 (transformers integration needs it importable)
    except ImportError as exc:
        raise ModelError(f"awq backend needs the autoawq package: {exc}") from exc
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            source,
            device_map={"": compute.device},
            trust_remote_code=False,
        )
        model.eval()
    except Exception as exc:
        raise ModelError(f"failed to load AWQ model {source!r} on {compute.device}: {exc}") from exc
    return model


def load_model(
    source: str,
    device_preference: str | None = None,
    quant: str | None = None,
    offline: bool = False,
) -> tuple[Any, Resolution]:
    """Load weights in-process on the resolved device. Returns (model, resolution)."""
    name = (quant or BASELINE_QUANT).strip().lower()
    if name not in QUANT_BACKENDS:
        raise ConfigurationError(f"unknown quant backend {quant!r}: expected one of {QUANT_BACKENDS}")
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    compute = resolve_device(device_preference, min_free_bytes=_MIN_GPU_FREE_BYTES)
    log.info("ojcore compute device: %s (%s)", compute.label, compute.reason)
    if name == "quanto_qint4":
        model = _load_quanto(source, compute)
    elif name == "torchao_int4":
        model = _load_torchao(source, compute)
    elif name == "awq":
        model = _load_awq(source, compute)
    else:
        model = _load_baseline(source, compute)
    try:
        model._ojcore_source = source
    except Exception:
        pass
    return model, compute
