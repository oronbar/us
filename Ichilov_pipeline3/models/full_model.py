"""
End-to-end Ichilov pipeline3 model:
frames -> frame encoder -> temporal encoder -> view fusion -> longitudinal model.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .frame_encoder import FrameEncoder
from .longitudinal_model import LongitudinalModel
from .temporal_encoder import TemporalEncoder
from .view_fusion import ViewFusion


class IchilovPipeline3Model(nn.Module):
    VIEW_ORDER = ("A2C", "A3C", "A4C")

    def __init__(
        self,
        frame_encoder: Optional[FrameEncoder] = None,
        temporal_encoder: Optional[TemporalEncoder] = None,
        view_fusion: Optional[ViewFusion] = None,
        longitudinal_model: Optional[LongitudinalModel] = None,
        frame_dim: int = 384,
        temporal_layers: int = 2,
        temporal_heads: int = 6,
        temporal_dropout: float = 0.1,
        backbone_name: str = "vit_small_patch16_dinov2.lvd142m",
        backbone_pretrained: bool = True,
        backbone_freeze: bool = True,
        unfreeze_last_blocks: int = 0,
        longitudinal_hidden: int = 256,
        longitudinal_model_type: str = "gru",
        longitudinal_layers: int = 1,
        longitudinal_heads: int = 4,
        longitudinal_dropout: float = 0.1,
        use_time_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.frame_encoder = frame_encoder or FrameEncoder(
            backbone_name=backbone_name,
            pretrained=backbone_pretrained,
            freeze_backbone=backbone_freeze,
            unfreeze_last_blocks=unfreeze_last_blocks,
        )
        d_model = int(getattr(self.frame_encoder, "output_dim", frame_dim))
        self.temporal_encoder = temporal_encoder or TemporalEncoder(
            dim=d_model,
            num_layers=temporal_layers,
            num_heads=temporal_heads,
            dropout=temporal_dropout,
        )
        self.view_fusion = view_fusion or ViewFusion(dim=d_model)
        self.longitudinal_model = longitudinal_model or LongitudinalModel(
            input_dim=d_model,
            hidden_dim=longitudinal_hidden,
            model_type=longitudinal_model_type,
            num_layers=longitudinal_layers,
            num_heads=longitudinal_heads,
            dropout=longitudinal_dropout,
            use_time_encoding=use_time_encoding,
        )
        self.embedding_dim = d_model

    def _encode_view(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        # frames: [B, V, T, C, H, W], frame_mask: [B, V, T]
        bsz, n_visits, n_frames, c, h, w = frames.shape
        flat_frames = frames.reshape(bsz * n_visits * n_frames, c, h, w)
        flat_emb = self.frame_encoder(flat_frames)
        view_seq = flat_emb.reshape(bsz * n_visits, n_frames, -1)
        seq_mask = frame_mask.reshape(bsz * n_visits, n_frames)
        pooled = self.temporal_encoder(view_seq, seq_mask)
        return pooled.reshape(bsz, n_visits, -1)

    def _encode_view_from_embeddings(
        self,
        embeddings: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        # embeddings: [B, V, T, D], frame_mask: [B, V, T]
        if embeddings.ndim != 4:
            raise ValueError(f"Expected embeddings [B,V,T,D], got {tuple(embeddings.shape)}")
        bsz, n_visits, n_frames, _ = embeddings.shape
        seq = embeddings.reshape(bsz * n_visits, n_frames, -1)
        seq_mask = frame_mask.reshape(bsz * n_visits, n_frames)
        pooled = self.temporal_encoder(seq, seq_mask)
        return pooled.reshape(bsz, n_visits, -1)

    def forward(
        self,
        frames_by_view: Dict[str, torch.Tensor],
        frame_masks_by_view: Dict[str, torch.Tensor],
        visit_mask: torch.Tensor,
        visit_times: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        view_embeddings: Dict[str, Optional[torch.Tensor]] = {}
        view_valid_masks: Dict[str, Optional[torch.Tensor]] = {}

        for view in self.VIEW_ORDER:
            frames = frames_by_view.get(view)
            frame_mask = frame_masks_by_view.get(view)
            if frames is None or frame_mask is None:
                view_embeddings[view] = None
                view_valid_masks[view] = None
                continue
            emb = self._encode_view(frames, frame_mask)
            valid = frame_mask.any(dim=-1) & visit_mask
            view_embeddings[view] = emb
            view_valid_masks[view] = valid

        bsz, n_visits = visit_mask.shape
        fused_visits = []
        for visit_idx in range(n_visits):
            per_visit_emb = {}
            per_visit_mask = {}
            for view in self.VIEW_ORDER:
                emb = view_embeddings.get(view)
                msk = view_valid_masks.get(view)
                if emb is None or msk is None:
                    per_visit_emb[view] = None
                    per_visit_mask[view] = None
                else:
                    per_visit_emb[view] = emb[:, visit_idx, :]
                    per_visit_mask[view] = msk[:, visit_idx]
            fused = self.view_fusion(per_visit_emb, per_visit_mask)
            fused_visits.append(fused)

        visit_embeddings = torch.stack(fused_visits, dim=1) if fused_visits else torch.zeros(
            bsz, 0, self.embedding_dim, device=visit_mask.device
        )
        outputs = self.longitudinal_model(
            visit_embeddings,
            visit_mask=visit_mask,
            visit_times=visit_times,
        )
        outputs["visit_embedding"] = visit_embeddings
        return outputs

    def forward_from_frame_embeddings(
        self,
        embeddings_by_view: Dict[str, torch.Tensor],
        frame_masks_by_view: Dict[str, torch.Tensor],
        visit_mask: torch.Tensor,
        visit_times: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        view_embeddings: Dict[str, Optional[torch.Tensor]] = {}
        view_valid_masks: Dict[str, Optional[torch.Tensor]] = {}

        for view in self.VIEW_ORDER:
            emb = embeddings_by_view.get(view)
            frame_mask = frame_masks_by_view.get(view)
            if emb is None or frame_mask is None:
                view_embeddings[view] = None
                view_valid_masks[view] = None
                continue
            pooled = self._encode_view_from_embeddings(emb, frame_mask)
            valid = frame_mask.any(dim=-1) & visit_mask
            view_embeddings[view] = pooled
            view_valid_masks[view] = valid

        bsz, n_visits = visit_mask.shape
        fused_visits = []
        for visit_idx in range(n_visits):
            per_visit_emb = {}
            per_visit_mask = {}
            for view in self.VIEW_ORDER:
                emb = view_embeddings.get(view)
                msk = view_valid_masks.get(view)
                if emb is None or msk is None:
                    per_visit_emb[view] = None
                    per_visit_mask[view] = None
                else:
                    per_visit_emb[view] = emb[:, visit_idx, :]
                    per_visit_mask[view] = msk[:, visit_idx]
            fused_visits.append(self.view_fusion(per_visit_emb, per_visit_mask))

        visit_embeddings = torch.stack(fused_visits, dim=1) if fused_visits else torch.zeros(
            bsz, 0, self.embedding_dim, device=visit_mask.device
        )
        outputs = self.longitudinal_model(
            visit_embeddings,
            visit_mask=visit_mask,
            visit_times=visit_times,
        )
        outputs["visit_embedding"] = visit_embeddings
        return outputs
