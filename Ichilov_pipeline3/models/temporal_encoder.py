"""
Temporal encoder for per-view frame embeddings.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """
    Input:
      - x: [B, T, D]
      - mask: [B, T] (optional, True for valid)
    Output:
      - pooled: [B, D]
    """

    def __init__(
        self,
        dim: int = 384,
        num_layers: int = 2,
        num_heads: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_seq_len = int(max_seq_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.attn_pool = nn.Linear(self.dim, 1)
        self.norm = nn.LayerNorm(self.dim)

        pos = self._build_sinusoidal(self.max_seq_len, self.dim)
        self.register_buffer("positional", pos, persistent=False)

    @staticmethod
    def _build_sinusoidal(length: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,D], got {tuple(x.shape)}")
        bsz, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got dim={dim}")

        if seq_len > self.positional.shape[0]:
            pos = self._build_sinusoidal(seq_len, self.dim).to(x.device)
        else:
            pos = self.positional[:seq_len].to(x.device)
        h = x + pos.unsqueeze(0)

        if mask is None:
            mask = torch.ones(bsz, seq_len, device=x.device, dtype=torch.bool)
        if mask.ndim != 2:
            raise ValueError(f"Expected mask [B,T], got {tuple(mask.shape)}")

        h = self.encoder(h, src_key_padding_mask=~mask)
        h = self.norm(h)

        logits = self.attn_pool(h).squeeze(-1)
        logits = logits.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(logits, dim=1)
        attn = torch.where(mask, attn, torch.zeros_like(attn))
        denom = attn.sum(dim=1, keepdim=True).clamp(min=1e-6)
        attn = attn / denom
        pooled = torch.sum(h * attn.unsqueeze(-1), dim=1)
        return pooled

