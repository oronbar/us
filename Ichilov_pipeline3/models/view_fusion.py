"""
Attention-based fusion across A2C/A3C/A4C view embeddings.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class ViewFusion(nn.Module):
    """
    Input per visit:
      {
        "A2C": [B, D] or None,
        "A3C": [B, D] or None,
        "A4C": [B, D] or None,
      }
    Output:
      - fused: [B, D]
    """

    VIEW_ORDER = ("A2C", "A3C", "A4C")

    def __init__(self, dim: int = 384) -> None:
        super().__init__()
        self.dim = int(dim)
        self.view_tokens = nn.Parameter(torch.zeros(len(self.VIEW_ORDER), self.dim))
        self.score = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, 1),
        )
        nn.init.trunc_normal_(self.view_tokens, std=0.02)

    def forward(
        self,
        views: Dict[str, Optional[torch.Tensor]],
        view_masks: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        first_tensor = None
        for key in self.VIEW_ORDER:
            t = views.get(key)
            if t is not None:
                first_tensor = t
                break
        if first_tensor is None:
            raise ValueError("ViewFusion received no view tensors.")

        bsz = first_tensor.shape[0]
        embs = []
        scores = []
        masks = []
        device = first_tensor.device

        for view_idx, view_name in enumerate(self.VIEW_ORDER):
            emb = views.get(view_name)
            if emb is None:
                continue
            if emb.ndim != 2 or emb.shape[1] != self.dim:
                raise ValueError(
                    f"{view_name} embedding expected [B,{self.dim}], got {tuple(emb.shape)}"
                )
            token = self.view_tokens[view_idx].unsqueeze(0).expand_as(emb)
            score = self.score(emb + token).squeeze(-1)
            if view_masks and view_name in view_masks and view_masks[view_name] is not None:
                valid = view_masks[view_name].to(device=device, dtype=torch.bool)
            else:
                valid = torch.ones(bsz, device=device, dtype=torch.bool)
            score = score.masked_fill(~valid, float("-inf"))
            embs.append(emb)
            scores.append(score)
            masks.append(valid)

        emb_stack = torch.stack(embs, dim=1)  # [B, Nv, D]
        score_stack = torch.stack(scores, dim=1)  # [B, Nv]
        mask_stack = torch.stack(masks, dim=1)  # [B, Nv]

        attn = torch.softmax(score_stack, dim=1)
        attn = torch.where(mask_stack, attn, torch.zeros_like(attn))
        denom = attn.sum(dim=1, keepdim=True).clamp(min=1e-6)
        attn = attn / denom
        fused = torch.sum(emb_stack * attn.unsqueeze(-1), dim=1)
        return fused

