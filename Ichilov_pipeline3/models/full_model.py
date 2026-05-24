"""
End-to-end Ichilov pipeline3 model:
frames -> frame encoder -> temporal encoder -> view fusion -> longitudinal model.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

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
        temporal_mode: str = "attn_pool",
        view_fusion_mode: str = "attn",
        single_best_view: str = "A4C",
    ) -> None:
        super().__init__()
        self.temporal_mode = str(temporal_mode).lower()
        self.view_fusion_mode = str(view_fusion_mode).lower()
        self.single_best_view = str(single_best_view).upper()
        if self.temporal_mode not in {"attn_pool", "mean_pool", "max_pool", "first_frame", "last_frame"}:
            raise ValueError(f"Unsupported temporal_mode: {self.temporal_mode}")
        if self.view_fusion_mode not in {"attn", "mean", "max", "single_best_view", "concat_then_linear"}:
            raise ValueError(f"Unsupported view_fusion_mode: {self.view_fusion_mode}")
        if self.single_best_view not in self.VIEW_ORDER:
            raise ValueError(f"single_best_view must be one of {self.VIEW_ORDER}, got {self.single_best_view}")

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
        self.concat_view_proj = nn.Linear(d_model * len(self.VIEW_ORDER), d_model)
        self.embedding_dim = d_model

    @staticmethod
    def _masked_mean(seq: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = mask.to(seq.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        attn = weights / denom
        pooled = (seq * attn.unsqueeze(-1)).sum(dim=1)
        return pooled, attn

    @staticmethod
    def _masked_max(seq: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        masked = seq.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        pooled, _ = masked.max(dim=1)
        pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
        norms = torch.norm(seq, dim=-1)
        norms = norms.masked_fill(~mask, float("-inf"))
        top_idx = norms.argmax(dim=1)
        attn = torch.zeros_like(mask, dtype=seq.dtype)
        valid_any = mask.any(dim=1)
        if torch.any(valid_any):
            rows = torch.where(valid_any)[0]
            attn[rows, top_idx[rows]] = 1.0
        return pooled, attn

    @staticmethod
    def _masked_first(seq: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n_frames, _ = seq.shape
        idx = torch.zeros(bsz, dtype=torch.long, device=seq.device)
        found = torch.zeros(bsz, dtype=torch.bool, device=seq.device)
        for t in range(n_frames):
            take = (~found) & mask[:, t]
            idx = torch.where(take, torch.full_like(idx, t), idx)
            found = found | take
        gather = idx.view(-1, 1, 1).expand(-1, 1, seq.shape[-1])
        pooled = seq.gather(dim=1, index=gather).squeeze(1)
        pooled = torch.where(found.unsqueeze(-1), pooled, torch.zeros_like(pooled))
        attn = torch.zeros_like(mask, dtype=seq.dtype)
        if torch.any(found):
            rows = torch.where(found)[0]
            attn[rows, idx[rows]] = 1.0
        return pooled, attn

    @staticmethod
    def _masked_last(seq: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n_frames, _ = seq.shape
        idx = torch.zeros(bsz, dtype=torch.long, device=seq.device)
        for t in range(n_frames):
            take = mask[:, t]
            idx = torch.where(take, torch.full_like(idx, t), idx)
        found = mask.any(dim=1)
        gather = idx.view(-1, 1, 1).expand(-1, 1, seq.shape[-1])
        pooled = seq.gather(dim=1, index=gather).squeeze(1)
        pooled = torch.where(found.unsqueeze(-1), pooled, torch.zeros_like(pooled))
        attn = torch.zeros_like(mask, dtype=seq.dtype)
        if torch.any(found):
            rows = torch.where(found)[0]
            attn[rows, idx[rows]] = 1.0
        return pooled, attn

    def _pool_temporal(
        self,
        seq: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.temporal_mode == "attn_pool":
            pooled, attn = self.temporal_encoder(seq, mask, return_attention=True)
            return pooled, attn
        if self.temporal_mode == "mean_pool":
            return self._masked_mean(seq, mask)
        if self.temporal_mode == "max_pool":
            return self._masked_max(seq, mask)
        if self.temporal_mode == "first_frame":
            return self._masked_first(seq, mask)
        if self.temporal_mode == "last_frame":
            return self._masked_last(seq, mask)
        raise ValueError(f"Unsupported temporal mode: {self.temporal_mode}")

    def _fuse_views(
        self,
        per_visit_emb: Dict[str, Optional[torch.Tensor]],
        per_visit_mask: Dict[str, Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first = None
        for view in self.VIEW_ORDER:
            if per_visit_emb.get(view) is not None:
                first = per_visit_emb[view]
                break
        if first is None:
            raise ValueError("No per-view embeddings available for fusion.")
        bsz, dim = first.shape
        device = first.device

        emb_list = []
        mask_list = []
        for view in self.VIEW_ORDER:
            emb = per_visit_emb.get(view)
            if emb is None:
                emb = torch.zeros(bsz, dim, device=device, dtype=first.dtype)
                valid = torch.zeros(bsz, device=device, dtype=torch.bool)
            else:
                valid = per_visit_mask.get(view)
                if valid is None:
                    valid = torch.ones(bsz, device=device, dtype=torch.bool)
                else:
                    valid = valid.to(device=device, dtype=torch.bool)
            emb_list.append(emb)
            mask_list.append(valid)

        emb_stack = torch.stack(emb_list, dim=1)  # [B,3,D]
        mask_stack = torch.stack(mask_list, dim=1)  # [B,3]

        if self.view_fusion_mode == "attn":
            fused, attn, attn_mask = self.view_fusion(
                per_visit_emb, per_visit_mask, return_attention=True
            )
            return fused, attn, attn_mask

        if self.view_fusion_mode == "mean":
            weights = mask_stack.to(emb_stack.dtype)
            denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
            attn = weights / denom
            fused = torch.sum(emb_stack * attn.unsqueeze(-1), dim=1)
            return fused, attn, mask_stack

        if self.view_fusion_mode == "max":
            masked = emb_stack.masked_fill(~mask_stack.unsqueeze(-1), float("-inf"))
            fused, _ = masked.max(dim=1)
            fused = torch.where(torch.isfinite(fused), fused, torch.zeros_like(fused))
            norms = torch.norm(emb_stack, dim=-1).masked_fill(~mask_stack, float("-inf"))
            top_idx = norms.argmax(dim=1)
            attn = torch.zeros_like(norms)
            valid_any = mask_stack.any(dim=1)
            if torch.any(valid_any):
                rows = torch.where(valid_any)[0]
                attn[rows, top_idx[rows]] = 1.0
            return fused, attn, mask_stack

        if self.view_fusion_mode == "single_best_view":
            best_idx = self.VIEW_ORDER.index(self.single_best_view)
            idx = torch.full((bsz,), best_idx, dtype=torch.long, device=device)
            valid_best = mask_stack[:, best_idx]
            fallback = mask_stack.float().argmax(dim=1)
            idx = torch.where(valid_best, idx, fallback)
            valid_any = mask_stack.any(dim=1)
            gather = idx.view(-1, 1, 1).expand(-1, 1, dim)
            fused = emb_stack.gather(dim=1, index=gather).squeeze(1)
            fused = torch.where(valid_any.unsqueeze(-1), fused, torch.zeros_like(fused))
            attn = torch.zeros(bsz, len(self.VIEW_ORDER), device=device, dtype=emb_stack.dtype)
            if torch.any(valid_any):
                rows = torch.where(valid_any)[0]
                attn[rows, idx[rows]] = 1.0
            return fused, attn, mask_stack

        if self.view_fusion_mode == "concat_then_linear":
            concat = emb_stack.reshape(bsz, -1)
            fused = self.concat_view_proj(concat)
            weights = mask_stack.to(emb_stack.dtype)
            denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
            attn = weights / denom
            return fused, attn, mask_stack

        raise ValueError(f"Unsupported view_fusion_mode: {self.view_fusion_mode}")

    def _encode_view(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # frames: [B, V, T, C, H, W], frame_mask: [B, V, T]
        bsz, n_visits, n_frames, c, h, w = frames.shape
        flat_frames = frames.reshape(bsz * n_visits * n_frames, c, h, w)
        flat_emb = self.frame_encoder(flat_frames)
        view_seq = flat_emb.reshape(bsz * n_visits, n_frames, -1)
        seq_mask = frame_mask.reshape(bsz * n_visits, n_frames)
        pooled, attn = self._pool_temporal(view_seq, seq_mask)
        return pooled.reshape(bsz, n_visits, -1), attn.reshape(bsz, n_visits, n_frames)

    def _encode_view_from_embeddings(
        self,
        embeddings: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # embeddings: [B, V, T, D], frame_mask: [B, V, T]
        if embeddings.ndim != 4:
            raise ValueError(f"Expected embeddings [B,V,T,D], got {tuple(embeddings.shape)}")
        bsz, n_visits, n_frames, _ = embeddings.shape
        seq = embeddings.reshape(bsz * n_visits, n_frames, -1)
        seq_mask = frame_mask.reshape(bsz * n_visits, n_frames)
        pooled, attn = self._pool_temporal(seq, seq_mask)
        return pooled.reshape(bsz, n_visits, -1), attn.reshape(bsz, n_visits, n_frames)

    def forward(
        self,
        frames_by_view: Dict[str, torch.Tensor],
        frame_masks_by_view: Dict[str, torch.Tensor],
        visit_mask: torch.Tensor,
        visit_times: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        view_embeddings: Dict[str, Optional[torch.Tensor]] = {}
        view_valid_masks: Dict[str, Optional[torch.Tensor]] = {}
        frame_attn_by_view: Dict[str, Optional[torch.Tensor]] = {}

        for view in self.VIEW_ORDER:
            frames = frames_by_view.get(view)
            frame_mask = frame_masks_by_view.get(view)
            if frames is None or frame_mask is None:
                view_embeddings[view] = None
                view_valid_masks[view] = None
                frame_attn_by_view[view] = None
                continue
            emb, frame_attn = self._encode_view(frames, frame_mask)
            valid = frame_mask.any(dim=-1) & visit_mask
            view_embeddings[view] = emb
            view_valid_masks[view] = valid
            frame_attn_by_view[view] = frame_attn

        bsz, n_visits = visit_mask.shape
        fused_visits = []
        view_attn_visits = []
        view_attn_masks = []
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
            fused, v_attn, v_mask = self._fuse_views(per_visit_emb, per_visit_mask)
            fused_visits.append(fused)
            view_attn_visits.append(v_attn)
            view_attn_masks.append(v_mask)

        visit_embeddings = torch.stack(fused_visits, dim=1) if fused_visits else torch.zeros(
            bsz, 0, self.embedding_dim, device=visit_mask.device
        )
        view_attn = torch.stack(view_attn_visits, dim=1) if view_attn_visits else torch.zeros(
            bsz, 0, len(self.VIEW_ORDER), device=visit_mask.device
        )
        view_attn_mask = torch.stack(view_attn_masks, dim=1) if view_attn_masks else torch.zeros(
            bsz, 0, len(self.VIEW_ORDER), device=visit_mask.device, dtype=torch.bool
        )
        outputs = self.longitudinal_model(
            visit_embeddings,
            visit_mask=visit_mask,
            visit_times=visit_times,
        )
        outputs["visit_embedding"] = visit_embeddings
        outputs["diagnostics"] = {
            "frame_attn_by_view": {
                k: (v.detach() if v is not None else None) for k, v in frame_attn_by_view.items()
            },
            "frame_mask_by_view": {
                k: (frame_masks_by_view[k].detach() if k in frame_masks_by_view else None)
                for k in self.VIEW_ORDER
            },
            "view_attn": view_attn.detach(),
            "view_attn_mask": view_attn_mask.detach(),
        }
        return outputs

    def forward_from_frame_embeddings(
        self,
        embeddings_by_view: Dict[str, torch.Tensor],
        frame_masks_by_view: Dict[str, torch.Tensor],
        visit_mask: torch.Tensor,
        visit_times: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        for view, emb in embeddings_by_view.items():
            if emb.ndim != 4:
                raise ValueError(f"embeddings_by_view[{view}] must be [B,V,T,D], got {tuple(emb.shape)}")
            fmask = frame_masks_by_view.get(view)
            if fmask is not None:
                if fmask.ndim != 3:
                    raise ValueError(
                        f"frame_masks_by_view[{view}] must be [B,V,T], got {tuple(fmask.shape)}"
                    )
                if tuple(fmask.shape[:3]) != tuple(emb.shape[:3]):
                    raise ValueError(
                        f"Shape mismatch for {view}: embeddings {tuple(emb.shape[:3])} vs mask {tuple(fmask.shape[:3])}"
                    )
        view_embeddings: Dict[str, Optional[torch.Tensor]] = {}
        view_valid_masks: Dict[str, Optional[torch.Tensor]] = {}
        frame_attn_by_view: Dict[str, Optional[torch.Tensor]] = {}

        for view in self.VIEW_ORDER:
            emb = embeddings_by_view.get(view)
            frame_mask = frame_masks_by_view.get(view)
            if emb is None or frame_mask is None:
                view_embeddings[view] = None
                view_valid_masks[view] = None
                frame_attn_by_view[view] = None
                continue
            pooled, frame_attn = self._encode_view_from_embeddings(emb, frame_mask)
            valid = frame_mask.any(dim=-1) & visit_mask
            view_embeddings[view] = pooled
            view_valid_masks[view] = valid
            frame_attn_by_view[view] = frame_attn

        bsz, n_visits = visit_mask.shape
        fused_visits = []
        view_attn_visits = []
        view_attn_masks = []
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
            fused, v_attn, v_mask = self._fuse_views(per_visit_emb, per_visit_mask)
            fused_visits.append(fused)
            view_attn_visits.append(v_attn)
            view_attn_masks.append(v_mask)

        visit_embeddings = torch.stack(fused_visits, dim=1) if fused_visits else torch.zeros(
            bsz, 0, self.embedding_dim, device=visit_mask.device
        )
        view_attn = torch.stack(view_attn_visits, dim=1) if view_attn_visits else torch.zeros(
            bsz, 0, len(self.VIEW_ORDER), device=visit_mask.device
        )
        view_attn_mask = torch.stack(view_attn_masks, dim=1) if view_attn_masks else torch.zeros(
            bsz, 0, len(self.VIEW_ORDER), device=visit_mask.device, dtype=torch.bool
        )
        outputs = self.longitudinal_model(
            visit_embeddings,
            visit_mask=visit_mask,
            visit_times=visit_times,
        )
        outputs["visit_embedding"] = visit_embeddings
        outputs["diagnostics"] = {
            "frame_attn_by_view": {
                k: (v.detach() if v is not None else None) for k, v in frame_attn_by_view.items()
            },
            "frame_mask_by_view": {
                k: (frame_masks_by_view[k].detach() if k in frame_masks_by_view else None)
                for k in self.VIEW_ORDER
            },
            "view_attn": view_attn.detach(),
            "view_attn_mask": view_attn_mask.detach(),
        }
        return outputs
