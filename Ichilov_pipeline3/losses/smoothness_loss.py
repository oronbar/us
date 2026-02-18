"""
Temporal smoothness regularization over visit severity scores.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothnessLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        severity_score: torch.Tensor,
        visit_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if severity_score.ndim != 2:
            raise ValueError("SmoothnessLoss expects severity_score shape [B,V].")
        if severity_score.shape[1] < 2:
            return torch.tensor(0.0, device=severity_score.device, dtype=severity_score.dtype)

        diffs = torch.abs(severity_score[:, 1:] - severity_score[:, :-1])
        if visit_mask is None:
            return diffs.mean()
        valid = visit_mask[:, 1:] & visit_mask[:, :-1]
        if valid.any():
            return F.l1_loss(diffs[valid], torch.zeros_like(diffs[valid]))
        return torch.tensor(0.0, device=severity_score.device, dtype=severity_score.dtype)

