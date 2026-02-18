"""
Pairwise ranking loss for within-patient visit severity ordering.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseRankingLoss(nn.Module):
    """
    Computes:
      log(1 + exp(-(s_i - s_j)))
    for valid ordered pairs, with sign inferred from target ordering.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        pred_severity: torch.Tensor,
        target_severity: torch.Tensor,
        visit_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pred_severity.ndim != 2 or target_severity.ndim != 2:
            raise ValueError("PairwiseRankingLoss expects [B,V] tensors.")

        bsz, _ = pred_severity.shape
        losses = []
        for bidx in range(bsz):
            if visit_mask is None:
                valid = torch.ones_like(pred_severity[bidx], dtype=torch.bool)
            else:
                valid = visit_mask[bidx].to(dtype=torch.bool, device=pred_severity.device)
            idx = valid.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() < 2:
                continue
            p = pred_severity[bidx, idx]
            t = target_severity[bidx, idx]
            n = p.numel()
            for i in range(n - 1):
                for j in range(i + 1, n):
                    delta_t = t[j] - t[i]
                    if torch.abs(delta_t) <= self.eps:
                        continue
                    sign = torch.sign(delta_t)
                    losses.append(F.softplus(-(p[i] - p[j]) * sign))

        if not losses:
            return torch.tensor(0.0, device=pred_severity.device, dtype=pred_severity.dtype)
        return torch.stack(losses).mean()

