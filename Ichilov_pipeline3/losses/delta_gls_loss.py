"""
Huber loss for delta GLS prediction.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaGLSLoss(nn.Module):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = float(beta)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=pred.device)
            pred = pred[mask]
            target = target[mask]
        if pred.numel() == 0:
            return torch.tensor(0.0, device=target.device, dtype=target.dtype)
        return F.smooth_l1_loss(pred, target, beta=self.beta)

