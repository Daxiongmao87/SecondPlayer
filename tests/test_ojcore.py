"""Tests for the baked-in ojcore readout: prompt shape, scoring, loader gate, no-network."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest

from secondplayer.errors import ConfigurationError, ModelError
from secondplayer.policy.ojcore import (
    PROMPT_VERSION,
    direct_messages,
    load_model,
    options_block,
    render_suffix,
    score,
    slot_ids,
    slot_probs,
    slot_probs_from_logits,
)


def row() -> dict[str, Any]:
    return {
        "id": "movement",
        "state": "Game: test.\nScreen: a blank screen.",
        "question": "Which direction?",
        "options": [
            {"id": "left", "description": "Hold LEFT."},
            {"id": "right", "description": "Hold RIGHT."},
            {"id": "keep"},
        ],
    }


def test_prompt_version_pinned():
    assert PROMPT_VERSION == "ojcore-choice-v1"


def test_options_block_lettering():
    assert options_block(row()["options"]) == "A. left: Hold LEFT.\nB. right: Hold RIGHT.\nC. keep"


def test_render_suffix_shape():
    suffix = render_suffix("Which direction?", row()["options"])
    assert suffix.startswith("Question: Which direction?\n\nOptions:\nA. left: Hold LEFT.")
    assert suffix.endswith("\n\nAnswer with one letter: A, B, C.")


def test_direct_messages_two_roles():
    state = row()["state"]
    messages = direct_messages(row())
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "exactly one letter" in messages[0]["content"]
    assert messages[1]["content"].startswith(state + "\n\nQuestion: Which direction?")


def test_direct_messages_rejects_bad_rows():
    with pytest.raises(ModelError, match="at most|exceed"):
        direct_messages({"state": "s", "question": "q", "options": [{"id": str(i)} for i in range(17)]})
    with pytest.raises(ModelError, match="no options"):
        direct_messages({"state": "s", "question": "q", "options": []})
    with pytest.raises(ModelError, match="question"):
        direct_messages({"state": "s", "options": [{"id": "a"}]})
    with pytest.raises(ModelError, match="id"):
        direct_messages({"state": "s", "question": "q", "options": [{"description": "x"}]})


class StubTokenizer:
    def __init__(self):
        self.table = {"A": 32, "B": 33, "C": 34}

    def encode(self, text, add_special_tokens=False):
        return [self.table[text]] if text in self.table else [1, 2]

    def decode(self, ids):
        inv = {v: k for k, v in self.table.items()}
        return "".join(inv.get(i, "?") for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def __call__(self, text, return_tensors=None):
        return {"input_ids": [[1, 2, 3]]}


class StubOut:
    def __init__(self, logits):
        self.logits = logits


class StubModel:
    device = "cpu"

    def __init__(self, last):
        self.last = last
        self.seen: dict[str, Any] = {}

    def forward(self, **kwargs):
        raise AssertionError("forward must not run directly in tests")

    def __call__(self, **kwargs):
        self.seen = kwargs
        return StubOut([[self.last]])


def test_slot_ids_and_probs():
    assert slot_ids(StubTokenizer(), 3) == [32, 33, 34]
    with pytest.raises(ModelError, match="round-trip"):
        slot_ids(StubTokenizer(), 4)
    probs = slot_probs(
        [{"id": 32, "logprob": 0.0}, {"id": 33, "logprob": -1.0}, {"id": 99, "logprob": -0.1}],
        [32, 33],
    )
    assert probs == pytest.approx([0.7311, 0.2689], abs=1e-3)
    with pytest.raises(ModelError, match="omitted answer slots"):
        slot_probs([{"id": 32, "logprob": 0.0}], [32, 77])
    assert slot_probs_from_logits([0.0] * 32 + [0.0, -1.0], [32, 33]) == pytest.approx(
        [0.7311, 0.2689], abs=1e-3
    )
    with pytest.raises(ModelError, match="at least two finite"):
        slot_probs_from_logits([1.0], [0])


def test_slot_probs_detach_grad_tensors():
    torch = pytest.importorskip("torch")
    import warnings

    logits = torch.tensor([2.0, 1.0, 0.0], requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        probs = slot_probs_from_logits(logits, [0, 1, 2])
    assert probs == pytest.approx([0.665, 0.245, 0.090], abs=1e-3)


def test_score_reads_first_position_slots():
    last = [0.0] * 50
    last[32], last[33], last[34] = 2.0, 1.0, 0.0
    result = score(StubModel(last), StubTokenizer(), row())
    assert result["id"] == "movement"
    assert result["option_ids"] == ["left", "right", "keep"]
    assert result["probabilities"] == pytest.approx([0.665, 0.245, 0.090], abs=1e-3)
    assert sum(result["probabilities"]) == pytest.approx(1.0)


def test_loader_rejects_unknown_quant_before_loading():
    with pytest.raises(ConfigurationError, match="unknown quant backend"):
        load_model("any/model", quant="q4", offline=True)


def test_quanto_backend_needs_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "quanto", None)
    with pytest.raises(ModelError, match="quanto"):
        load_model("any/model", "cpu", quant="quanto_qint4")


def test_quanto_backend_needs_file(tmp_path):
    with pytest.raises(ModelError, match="model.safetensors"):
        load_model(str(tmp_path), "cpu", quant="quanto_qint4")


def test_load_model_gates_small_gpus(monkeypatch):
    from types import SimpleNamespace

    from secondplayer.platform.devices import Resolution
    from secondplayer.policy.ojcore import loader as loader_mod

    seen: dict[str, Any] = {}

    def fake_resolve(pref=None, **kwargs):
        seen["pref"] = pref
        seen["kwargs"] = kwargs
        return Resolution("cpu", None, "CPU", "test")

    monkeypatch.setattr(loader_mod, "resolve_device", fake_resolve)
    monkeypatch.setattr(loader_mod, "_load_baseline", lambda source, compute: SimpleNamespace())
    load_model("any/model", "auto", quant="none")
    assert seen["pref"] == "auto"
    assert seen["kwargs"]["min_free_bytes"] == loader_mod._MIN_GPU_FREE_BYTES
    assert loader_mod._MIN_GPU_FREE_BYTES >= 3 * 1024**3


def test_torchao_backend_needs_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "torchao", None)
    monkeypatch.setitem(sys.modules, "torchao.quantization", None)
    with pytest.raises(ModelError, match="torchao"):
        load_model("any/model", "cpu", quant="torchao_int4")


def test_awq_backend_needs_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "awq", None)
    with pytest.raises(ModelError, match="autoawq"):
        load_model("any/model", "cpu", quant="awq")


def test_place_skips_cast_with_quantized_weights():
    import torch
    from quanto import QTensor, freeze, qint4, quantize

    from secondplayer.platform.devices import Resolution
    from secondplayer.policy.ojcore.loader import _place

    net = torch.nn.Sequential(torch.nn.Linear(8, 8))
    quantize(net, weights=qint4, activations=None)
    freeze(net)
    assert isinstance(net[0].weight, QTensor)
    compute = Resolution(device="cpu", dtype="bfloat16", label="CPU", reason="test")
    _place(net, compute, torch)  # must not raise and must not cast anything
    assert isinstance(net[0].weight, QTensor)
    assert net[0].bias is not None and net[0].bias.dtype == torch.float32


def test_qembed_roundtrip_and_storage():
    torch = pytest.importorskip("torch")
    from secondplayer.policy.ojcore.embed import QEmbed, quantize_embeddings

    torch.manual_seed(0)
    ref = torch.nn.Embedding(64, 16)
    q = QEmbed.from_float(ref)
    assert q._data.dtype == torch.int8
    assert q._scale.shape == (64, 1)
    ids = torch.tensor([0, 7, 63])
    assert torch.allclose(q(ids), ref(ids), atol=0.05, rtol=0.05)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(64, 16)
            self.fc = torch.nn.Linear(16, 16)

    net = Tiny()
    assert quantize_embeddings(net) == 1
    assert isinstance(net.emb, QEmbed)
    assert not isinstance(net.fc, QEmbed)
    keys = set(net.state_dict())
    assert "emb._data" in keys and "emb._scale" in keys


def test_qembed_keeps_source_dtype():
    torch = pytest.importorskip("torch")
    from secondplayer.policy.ojcore.embed import QEmbed

    torch.manual_seed(1)
    ref = torch.nn.Embedding(32, 8, dtype=torch.bfloat16)
    q = QEmbed.from_float(ref)
    assert q._scale.dtype == torch.bfloat16
    ids = torch.tensor([0, 31])
    out = q(ids)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), ref(ids).float(), atol=0.1, rtol=0.1)


def test_file_float_dtype_from_header(tmp_path):
    torch = pytest.importorskip("torch")
    import json
    import struct

    from secondplayer.policy.ojcore.loader import _file_float_dtype

    def header_file(entries):
        header = json.dumps(entries).encode()
        path = tmp_path / "model.safetensors"
        path.write_bytes(struct.pack("<Q", len(header)) + header)
        return str(path)

    bf16 = header_file({"model.embed_tokens._scale": {"dtype": "BF16", "shape": [4, 1]}})
    assert _file_float_dtype(bf16, torch) is torch.bfloat16
    plain = header_file({"model.embed_tokens.weight": {"dtype": "F32", "shape": [4, 4]}})
    assert _file_float_dtype(plain, torch) is torch.float32


def test_place_casts_plain_model():
    import torch

    from secondplayer.platform.devices import Resolution
    from secondplayer.policy.ojcore.loader import _place

    net = torch.nn.Sequential(torch.nn.Linear(8, 8))
    compute = Resolution(device="cpu", dtype="bfloat16", label="CPU", reason="test")
    _place(net, compute, torch)
    assert net[0].weight.dtype == torch.bfloat16
    assert net[0].bias.dtype == torch.bfloat16


def test_forward_chunks_long_prefill_with_cache(monkeypatch):
    torch = pytest.importorskip("torch")
    import inspect
    from contextlib import nullcontext
    from types import SimpleNamespace

    from secondplayer.policy.ojcore import scoring

    monkeypatch.setattr(scoring, "PREFILL_CHUNK", 4)
    calls = []

    class CacheModel(torch.nn.Module):
        def forward(
            self,
            input_ids,
            past_key_values=None,
            cache_position=None,
            use_cache=False,
            return_dict=True,
            logits_to_keep=0,
        ):
            calls.append(
                {
                    "len": input_ids.shape[-1],
                    "pos0": int(cache_position[0]),
                    "past": past_key_values,
                    "cache": use_cache,
                    "keep": logits_to_keep,
                }
            )
            return SimpleNamespace(logits=torch.zeros(1, 1, 8), past_key_values=("cache", len(calls)))

    model = CacheModel()
    inputs = {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10),
    }
    out = scoring._forward(model, inputs, inspect.signature(model.forward).parameters, torch, nullcontext())
    assert [c["len"] for c in calls] == [4, 4, 2]
    assert [c["pos0"] for c in calls] == [0, 4, 8]
    assert calls[0]["past"] is None
    assert calls[1]["past"] == ("cache", 1)
    assert calls[2]["past"] == ("cache", 2)
    assert all(c["cache"] for c in calls)
    assert all(c["keep"] == 1 for c in calls)
    assert out.logits.shape == (1, 1, 8)


def test_readout_uses_subset_head_when_available():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    hidden = torch.tensor([[[1.0, 0.0, -1.0, 2.0]]])
    rows = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )

    class SliceWeight:
        def __getitem__(self, ids):
            assert list(ids) == [32, 33, 34]
            return rows  # already float: exercises the no-dequantize branch

    class Backbone(torch.nn.Module):
        def forward(self, input_ids, use_cache=False, return_dict=True):
            return SimpleNamespace(last_hidden_state=hidden)

    class HeadModel:
        model = Backbone()
        lm_head = SimpleNamespace(weight=SliceWeight())

    model = HeadModel()
    result = score(model, StubTokenizer(), row())
    assert result["probabilities"] == pytest.approx([0.665, 0.245, 0.090], abs=1e-3)
    assert model._ojcore_subset_cache[(32, 33, 34)] is rows


def test_readout_prefers_sidecar_rows(tmp_path):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from secondplayer.policy.ojcore.scoring import HEAD_ROWS_FILE

    hidden = torch.tensor([[[1.0, 0.0, -1.0, 2.0]]])
    torch.save(
        {
            "ids": [32, 33, 34],
            "rows": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            ),
        },
        tmp_path / HEAD_ROWS_FILE,
    )

    class ExplodingWeight:
        def __getitem__(self, ids):
            raise AssertionError("live head must not be sliced when a sidecar exists")

    class Backbone(torch.nn.Module):
        def forward(self, input_ids, use_cache=False, return_dict=True):
            return SimpleNamespace(last_hidden_state=hidden)

    class SidecarModel:
        model = Backbone()
        lm_head = SimpleNamespace(weight=ExplodingWeight())
        _ojcore_source = str(tmp_path)

    model = SidecarModel()
    result = score(model, StubTokenizer(), row())
    assert result["probabilities"] == pytest.approx([0.665, 0.245, 0.090], abs=1e-3)
    assert model._ojcore_subset_cache[(32, 33, 34)].shape == (3, 4)


def test_readout_scores_without_lm_head_when_sidecar_covers(tmp_path):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from secondplayer.policy.ojcore.scoring import HEAD_ROWS_FILE

    hidden = torch.tensor([[[1.0, 0.0, -1.0, 2.0]]])
    torch.save(
        {
            "ids": [32, 33, 34],
            "rows": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            ),
        },
        tmp_path / HEAD_ROWS_FILE,
    )

    class Backbone(torch.nn.Module):
        def forward(self, input_ids, use_cache=False, return_dict=True):
            return SimpleNamespace(last_hidden_state=hidden)

    class DroppedHeadModel:
        model = Backbone()
        lm_head = None
        _ojcore_source = str(tmp_path)

    result = score(DroppedHeadModel(), StubTokenizer(), row())
    assert result["probabilities"] == pytest.approx([0.665, 0.245, 0.090], abs=1e-3)


def test_readout_fails_loud_without_head_or_sidecar(tmp_path):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    hidden = torch.tensor([[[1.0, 0.0, -1.0, 2.0]]])

    class Backbone(torch.nn.Module):
        def forward(self, input_ids, use_cache=False, return_dict=True):
            return SimpleNamespace(last_hidden_state=hidden)

    class BareModel:
        model = Backbone()
        lm_head = None
        _ojcore_source = str(tmp_path)

    with pytest.raises(ModelError, match="head_rows"):
        score(BareModel(), StubTokenizer(), row())


def test_readout_surfaces_backbone_errors_without_head():
    torch = pytest.importorskip("torch")

    class BrokenBackbone(torch.nn.Module):
        def forward(self, **kwargs):
            raise RuntimeError("boom")

    class BareModel:
        model = BrokenBackbone()
        lm_head = None

    with pytest.raises(RuntimeError, match="boom"):
        score(BareModel(), StubTokenizer(), row())


def test_readout_falls_back_to_full_logits():
    from types import SimpleNamespace

    last = [0.0] * 50
    last[32], last[33], last[34] = 2.0, 1.0, 0.0

    class BrokenBackbone:
        def forward(self, **kwargs):
            raise RuntimeError("no backbone here")

    class FallbackModel(StubModel):
        model = BrokenBackbone()
        lm_head = SimpleNamespace(weight=None)

    result = score(FallbackModel(last), StubTokenizer(), row())
    assert result["probabilities"] == pytest.approx([0.665, 0.245, 0.090], abs=1e-3)


def test_ojcore_has_no_network_hooks():
    root = Path(__file__).resolve().parent.parent / "secondplayer" / "policy" / "ojcore"
    sources = [p.read_text() for p in sorted(root.glob("*.py"))]
    assert sources, "ojcore package is empty"
    blob = "\n".join(sources)
    for banned in ("urllib", "http", "18081", "localhost", "socket"):
        assert banned not in blob
    assert not re.search(r"\bserver\b", blob, re.IGNORECASE)
