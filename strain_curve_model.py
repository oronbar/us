"""Temporal models for DICOM/view-level strain curve prediction."""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :].to(dtype=x.dtype, device=x.device)


class AttentionPooling(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        if mask is not None:
            weights = weights * mask.to(weights.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    w = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)


class StrainCurveModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        curve_length: int,
        model_type: str = "temporal_transformer",
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        derive_peak_from_curve: bool = False,
        use_view_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.curve_length = int(curve_length)
        self.model_type = str(model_type)
        self.derive_peak_from_curve = bool(derive_peak_from_curve)
        self.use_view_embedding = bool(use_view_embedding)

        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.view_embedding = nn.Embedding(4, hidden_dim) if use_view_embedding else None

        if self.model_type == "temporal_mean_pool":
            self.temporal = nn.Identity()
            self.pool = masked_mean
        elif self.model_type == "temporal_transformer":
            self.pos = SinusoidalPositionEncoding(hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.pooler = AttentionPooling(hidden_dim)
            self.pool = self.pooler
        elif self.model_type == "gru":
            self.temporal = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.gru_out = nn.Linear(hidden_dim * 2, hidden_dim)
            self.pool = masked_mean
        else:
            raise ValueError(
                "model_type must be one of: temporal_mean_pool, temporal_transformer, gru"
            )

        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.curve_head = nn.Linear(hidden_dim, self.curve_length)
        self.peak_head = nn.Linear(hidden_dim, 1)
        self.ttp_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.quality_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(
        self,
        embeddings: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
        view_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.input_proj(embeddings)
        if self.view_embedding is not None and view_id is not None:
            view_emb = self.view_embedding(view_id.clamp(0, 3)).unsqueeze(1)
            x = x + view_emb

        if self.model_type == "temporal_transformer":
            x = self.pos(x)
            key_padding_mask = frame_mask <= 0 if frame_mask is not None else None
            x = self.temporal(x, src_key_padding_mask=key_padding_mask)
            pooled = self.pool(x, frame_mask)
        elif self.model_type == "gru":
            x, _ = self.temporal(x)
            x = self.gru_out(x)
            pooled = self.pool(x, frame_mask)
        else:
            pooled = self.pool(x, frame_mask)

        h = self.shared(pooled)
        curve = self.curve_head(h)
        peak_head = self.peak_head(h).squeeze(-1)
        peak_from_curve = torch.min(curve, dim=1).values
        peak = peak_from_curve if self.derive_peak_from_curve else peak_head
        return {
            "pred_curve": curve,
            "pred_peak_gls": peak,
            "pred_peak_gls_head": peak_head,
            "pred_peak_gls_from_curve": peak_from_curve,
            "pred_time_to_peak": self.ttp_head(h).squeeze(-1),
            "quality_score": self.quality_head(h).squeeze(-1),
        }


def build_strain_curve_model(config: Dict[str, object]) -> StrainCurveModel:
    return StrainCurveModel(
        input_dim=int(config["input_dim"]),
        curve_length=int(config["curve_length"]),
        model_type=str(config.get("model_type", "temporal_transformer")),
        hidden_dim=int(config.get("hidden_dim", 256)),
        num_layers=int(config.get("num_layers", 2)),
        num_heads=int(config.get("num_heads", 4)),
        dropout=float(config.get("dropout", 0.1)),
        derive_peak_from_curve=bool(config.get("derive_peak_from_curve", False)),
        use_view_embedding=bool(config.get("use_view_embedding", True)),
    )
