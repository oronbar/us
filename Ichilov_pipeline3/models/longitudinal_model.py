"""
Longitudinal visit model (GRU or Transformer) for cardiotoxicity modeling.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class LongitudinalModel(nn.Module):
    """
    Input:
      - x: [B, V, D]
      - visit_mask: [B, V] (optional)
      - visit_times: [B, V] (optional)

    Returns:
      {
        "delta_gls": [B],
        "severity_score": [B, V],
        "risk_prob": [B],
      }
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        model_type: str = "gru",
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_time_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.model_type = str(model_type).lower()
        self.use_time_encoding = bool(use_time_encoding)

        self.input_proj = (
            nn.Identity() if self.input_dim == self.hidden_dim else nn.Linear(self.input_dim, self.hidden_dim)
        )
        self.time_proj = nn.Linear(1, self.hidden_dim) if self.use_time_encoding else None

        if self.model_type == "gru":
            self.encoder = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=int(num_layers),
                batch_first=True,
                dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            )
        elif self.model_type == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(num_heads),
                dropout=float(dropout),
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        else:
            raise ValueError(f"Unknown longitudinal model type: {model_type}")

        self.delta_head = nn.Linear(self.hidden_dim, 1)
        self.severity_head = nn.Linear(self.hidden_dim, 1)
        self.risk_head = nn.Linear(self.hidden_dim, 1)

    def _add_time_encoding(
        self,
        h: torch.Tensor,
        visit_times: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.use_time_encoding:
            return h
        bsz, n_visits, _ = h.shape
        if visit_times is None:
            visit_times = torch.arange(n_visits, device=h.device).unsqueeze(0).expand(bsz, -1).float()
        t = visit_times.float()
        t = t - t[:, :1]
        scale = t.abs().max(dim=1, keepdim=True).values.clamp(min=1.0)
        t = t / scale
        return h + self.time_proj(t.unsqueeze(-1))

    @staticmethod
    def _last_valid(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hidden: [B,V,D], mask: [B,V]
        lengths = mask.long().sum(dim=1).clamp(min=1) - 1
        gather_idx = lengths.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
        return hidden.gather(dim=1, index=gather_idx).squeeze(1)

    def forward(
        self,
        x: torch.Tensor,
        visit_mask: Optional[torch.Tensor] = None,
        visit_times: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,V,D], got {tuple(x.shape)}")
        bsz, n_visits, _ = x.shape
        if visit_mask is None:
            visit_mask = torch.ones(bsz, n_visits, dtype=torch.bool, device=x.device)
        if visit_mask.ndim != 2:
            raise ValueError(f"Expected visit_mask [B,V], got {tuple(visit_mask.shape)}")

        h = self.input_proj(x)
        h = self._add_time_encoding(h, visit_times)

        if self.model_type == "gru":
            lengths = visit_mask.long().sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths=lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.encoder(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=n_visits
            )
        else:
            h = self.encoder(h, src_key_padding_mask=~visit_mask)

        severity = self.severity_head(h).squeeze(-1)
        severity = severity.masked_fill(~visit_mask, 0.0)

        pooled = self._last_valid(h, visit_mask)
        delta = self.delta_head(pooled).squeeze(-1)
        risk_logit = self.risk_head(pooled).squeeze(-1)
        risk_prob = torch.sigmoid(risk_logit)

        return {
            "delta_gls": delta,
            "severity_score": severity,
            "risk_prob": risk_prob,
        }

