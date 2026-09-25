"""Symmetric per-row int8 embeddings for the sub-3GB quant file.

quanto 0.2.0 quantizes Linear/Conv/Norm only; the 636M-param fp32 embedding
table would bust the budget alone. QEmbed stores int8 rows plus one fp32
scale per row and dequantizes only gathered rows, so VRAM holds ~0.64GB.
Math dtype stays fp32 end to end (embedding output fp32, matching the
file's dequantized linears).
"""

from __future__ import annotations

from typing import Any

import torch


class QEmbed(torch.nn.Module):
    """Drop-in int8 replacement for nn.Embedding (lookup + row dequantize)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.register_buffer("_data", torch.zeros(num_embeddings, embedding_dim, dtype=torch.int8))
        self.register_buffer("_scale", torch.ones(num_embeddings, 1, dtype=dtype))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        gathered = self._data[ids].to(self._scale.dtype)
        return gathered * self._scale[ids]

    @classmethod
    def from_float(cls, embedding: torch.nn.Embedding) -> QEmbed:
        dtype = embedding.weight.dtype
        if dtype not in (torch.float32, torch.bfloat16, torch.float16):
            dtype = torch.float32
        weight = embedding.weight.detach().to(torch.float32)
        peak = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        scale = (peak / 127.0).to(dtype)
        codes = torch.round(weight / peak.clamp_min(1e-12) * 127.0).clamp(-127, 127).to(torch.int8)
        mod = cls(weight.shape[0], weight.shape[1], dtype=dtype)
        mod._data.copy_(codes)
        mod._scale.copy_(scale)
        return mod


def quantize_embeddings(model: torch.nn.Module) -> int:
    """Replace every nn.Embedding with its QEmbed equivalent. Returns the count."""
    swapped = 0
    for name, module in list(model.named_modules()):
        if type(module) is not torch.nn.Embedding:
            continue
        parent: Any = model
        *path, attr = name.split(".")
        for part in path:
            parent = getattr(parent, part)
        setattr(parent, attr, QEmbed.from_float(module))
        swapped += 1
    return swapped
