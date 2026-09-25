"""In-process single-token letter readout over a causal LM. No network calls."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

from ...errors import ModelError
from .messages import LETTERS, direct_messages


def slot_ids(tokenizer: Any, count: int) -> list[int]:
    """Verified single-token answer slots A.. (same checks as the oracle readout)."""
    result = []
    for letter in LETTERS[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ModelError(f"answer slot {letter!r} is not one exact round-trip token")
        result.append(encoded[0])
    if len(result) != len(set(result)):
        raise ModelError("answer-slot tokens collide")
    return result


def _softmax(logps: list[float]) -> list[float]:
    if len(logps) < 2 or any(not math.isfinite(v) for v in logps):
        raise ModelError("need at least two finite scores")
    peak = max(logps)
    exps = [math.exp(v - peak) for v in logps]
    total = sum(exps)
    return [v / total for v in exps]


def slot_probs(top_logprobs: list[dict[str, Any]], ids: list[int]) -> list[float]:
    """Softmax over the verified slots from a top-logprobs listing; fail loud on gaps."""
    by_id = {entry["id"]: entry["logprob"] for entry in top_logprobs}
    missing = [i for i, slot in enumerate(ids) if slot not in by_id]
    if missing:
        raise ModelError(f"readout omitted answer slots {missing} from top logprobs")
    return _softmax([by_id[slot] for slot in ids])


def slot_probs_from_logits(logits: Any, ids: list[int]) -> list[float]:
    """Softmax over the verified slots from a final-position logits vector."""
    if hasattr(logits, "detach"):
        logits = logits.detach()
    return _softmax([float(logits[slot]) for slot in ids])


def _apply_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _model_device(model: Any):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None


PREFILL_CHUNK = 256


HEAD_ROWS_FILE = "head_rows.pt"


def _sidecar_rows(source: Any, ids: list[int], torch: Any) -> Any | None:
    """Precomputed slot rows from quant time; None when missing or short."""
    import os

    if not source or not isinstance(source, str):
        return None
    path = os.path.join(source, HEAD_ROWS_FILE)
    if not os.path.isfile(path):
        return None
    blob = torch.load(path, map_location="cpu", weights_only=True)
    pos = {token: i for i, token in enumerate(blob.get("ids", []))}
    if any(token not in pos for token in ids):
        return None
    return blob["rows"][[pos[token] for token in ids]]


def _subset_rows(model: Any, ids: list[int], torch: Any, device: Any = None) -> Any:
    """Dequantized lm_head rows for the answer slots only; cached on the model.

    Quanto unpacks the whole 248k-row head on any slice (~2GB transient),
    so the sidecar written at quant time is the primary source; slicing the
    live weight is the fallback for backends without one.
    """
    cache = getattr(model, "_ojcore_subset_cache", None)
    if cache is None:
        cache = {}
        model._ojcore_subset_cache = cache
    key = tuple(ids)
    hit = cache.get(key)
    if hit is None:
        hit = _sidecar_rows(getattr(model, "_ojcore_source", None), ids, torch)
        if hit is None:
            sub = model.lm_head.weight[list(ids)]
            hit = sub.dequantize() if hasattr(sub, "dequantize") else sub
        if device is not None:
            hit = hit.to(device)
        cache[key] = hit
    return hit


def _forward_hidden(backbone: Any, inputs: dict[str, Any], params: Any, torch: Any, context: Any) -> Any:
    """Backbone-only forward; returns the final-position hidden state."""
    base: dict[str, Any] = {"return_dict": True}
    ids = inputs["input_ids"]
    cacheable = torch is not None and "past_key_values" in params and "cache_position" in params
    shape = getattr(ids, "shape", None)
    if shape is None or shape[-1] <= PREFILL_CHUNK or not cacheable:
        with context:
            out = backbone(**inputs, use_cache=False, **base)
            return out.last_hidden_state[0][-1]
    past: Any = None
    hidden: Any = None
    with context:
        for start in range(0, ids.shape[-1], PREFILL_CHUNK):
            stop = min(start + PREFILL_CHUNK, ids.shape[-1])
            chunk = {"input_ids": ids[..., start:stop]}
            position = torch.arange(start, stop, device=ids.device)
            out = backbone(
                **chunk, past_key_values=past, cache_position=position, use_cache=True, **base
            )
            past = out.past_key_values
            hidden = out.last_hidden_state[0][-1]
    return hidden


def _readout_logits(
    model: Any, inputs: dict[str, Any], params: Any, torch: Any, context: Any, ids: list[int]
) -> tuple[Any, list[int]]:
    """Slot logits via the subset head; falls back to a full forward.

    Backends whose head rows cannot be sliced (or stub models without a
    backbone) take the original full-logits path unchanged. A model whose
    head was dropped at load has no full path and needs its sidecar.
    """
    backbone = getattr(model, "model", None)
    if torch is not None and backbone is not None:
        import inspect

        try:
            bparams = inspect.signature(backbone.forward).parameters
            hidden = _forward_hidden(backbone, inputs, bparams, torch, context)
        except Exception:
            if getattr(model, "lm_head", None) is None:
                raise
        else:
            try:
                rows = _subset_rows(model, ids, torch, getattr(hidden, "device", None))
            except Exception as exc:
                cache = getattr(model, "_ojcore_subset_cache", None)
                if isinstance(cache, dict):
                    cache.clear()
                raise ModelError(
                    "dropped-head model needs its head_rows.pt sidecar for these slots"
                ) from exc
            logits = torch.matmul(hidden.to(rows.dtype), rows.t())
            return logits, list(range(len(ids)))
    out = _forward(model, inputs, params, torch, context)
    return out.logits[0][-1], ids


def _forward(model: Any, inputs: dict[str, Any], params: Any, torch: Any, context: Any) -> Any:
    """Single forward, or cache-threaded chunked prefill for long prompts."""
    base: dict[str, Any] = {"return_dict": True}
    if "logits_to_keep" in params:
        base["logits_to_keep"] = 1
    ids = inputs["input_ids"]
    cacheable = torch is not None and "past_key_values" in params and "cache_position" in params
    shape = getattr(ids, "shape", None)
    if shape is None or shape[-1] <= PREFILL_CHUNK or not cacheable:
        with context:
            return model(**inputs, use_cache=False, **base)
    # Chunked prefill: constant memory. Inputs here are never padded, so the
    # all-ones mask is dropped and position comes from cache_position alone.
    past: Any = None
    out: Any = None
    with context:
        for start in range(0, ids.shape[-1], PREFILL_CHUNK):
            stop = min(start + PREFILL_CHUNK, ids.shape[-1])
            chunk = {"input_ids": ids[..., start:stop]}
            position = torch.arange(start, stop, device=ids.device)
            out = model(**chunk, past_key_values=past, cache_position=position, use_cache=True, **base)
            past = out.past_key_values
    return out


def score(
    model: Any, tokenizer: Any, row: dict[str, Any], metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """One forward pass; softmax over the option letters at the first position."""
    try:
        import torch
    except ImportError:
        torch = None  # stubs exercise the readout path without torch
    options = row.get("options", [])
    names = [opt.get("id", opt.get("name")) for opt in options]
    prompt = _apply_template(tokenizer, direct_messages(row))
    inputs = tokenizer(prompt, return_tensors="pt")
    device = _model_device(model)
    if device is not None:
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    import inspect

    try:
        params = inspect.signature(model.forward).parameters
    except (AttributeError, TypeError, ValueError):
        params = {}
    # no_grad, not inference_mode: quantized backends (quanto) bump version
    # counters that inference tensors forbid.
    context = torch.no_grad() if torch is not None else nullcontext()
    ids = slot_ids(tokenizer, len(names))
    logits, slots = _readout_logits(model, inputs, params, torch, context, ids)
    return {
        "id": row.get("id"),
        "option_ids": names,
        "probabilities": slot_probs_from_logits(logits, slots),
    }
