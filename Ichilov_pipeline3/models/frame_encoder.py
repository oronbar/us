"""
Frame-level encoder backed by a DINOv2 ViT-S model.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("ichilov_pipeline3.frame_encoder")


def _resolve_blocks(module: nn.Module) -> List[nn.Module]:
    def _to_blocks(obj: object) -> List[nn.Module]:
        if isinstance(obj, (nn.ModuleList, list, tuple)):
            return [m for m in obj if isinstance(m, nn.Module)]
        if isinstance(obj, nn.Sequential):
            return list(obj.children())
        if isinstance(obj, nn.Module):
            children = list(obj.children())
            if children:
                return children
        return []

    for attr in ("blocks", "layers"):
        blocks = _to_blocks(getattr(module, attr, None))
        if blocks:
            return blocks

    encoder = getattr(module, "encoder", None)
    if encoder is not None:
        for attr in ("layer", "layers", "blocks"):
            blocks = _to_blocks(getattr(encoder, attr, None))
            if blocks:
                return blocks
    return []


class FrameEncoder(nn.Module):
    """
    Encode a single frame into a CLS embedding.

    Input:
      - x: [B, C, H, W]
    Output:
      - cls: [B, D]
    """

    def __init__(
        self,
        backbone_name: str = "vit_small_patch16_dinov2.lvd142m",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_last_blocks: int = 0,
        hf_fallback_name: str = "facebook/dinov2-small",
        input_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.hf_fallback_name = hf_fallback_name
        self.backend = "timm"
        self.backbone: nn.Module
        self.output_dim: int
        self.expected_img_size: Optional[tuple[int, int]] = None

        try:
            import timm

            create_kwargs = {"pretrained": pretrained, "num_classes": 0}
            if input_size is not None:
                create_kwargs["img_size"] = int(input_size)
            self.backbone = timm.create_model(backbone_name, **create_kwargs)
            if hasattr(self.backbone, "reset_classifier"):
                self.backbone.reset_classifier(0)
            self.output_dim = int(getattr(self.backbone, "num_features", 384))
            patch_embed = getattr(self.backbone, "patch_embed", None)
            img_size = getattr(patch_embed, "img_size", None) if patch_embed is not None else None
            if isinstance(img_size, int):
                self.expected_img_size = (img_size, img_size)
            elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
                self.expected_img_size = (int(img_size[0]), int(img_size[1]))
            logger.info("Loaded DINOv2 backbone from timm: %s", backbone_name)
        except Exception as exc_timm:
            logger.warning(
                "Failed loading timm model '%s' (%s). Falling back to HuggingFace '%s'.",
                backbone_name,
                exc_timm,
                hf_fallback_name,
            )
            try:
                from transformers import AutoModel

                self.backbone = AutoModel.from_pretrained(hf_fallback_name)
                self.backend = "hf"
                self.output_dim = int(self.backbone.config.hidden_size)
                hf_size = getattr(self.backbone.config, "image_size", None)
                if isinstance(hf_size, int):
                    self.expected_img_size = (hf_size, hf_size)
            except Exception as exc_hf:  # pragma: no cover - runtime dependency/environment
                raise RuntimeError(
                    "Unable to load DINOv2 backbone from timm or HuggingFace."
                ) from exc_hf

        self.freeze_backbone(freeze_backbone, unfreeze_last_blocks=unfreeze_last_blocks)

    def freeze_backbone(self, freeze: bool, unfreeze_last_blocks: int = 0) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

        if freeze and unfreeze_last_blocks > 0:
            blocks = _resolve_blocks(self.backbone)
            if not blocks:
                logger.warning(
                    "No transformer blocks found for selective unfreezing; keeping full freeze."
                )
                return
            for block in blocks[-int(unfreeze_last_blocks) :]:
                for p in block.parameters():
                    p.requires_grad = True

    def _forward_timm(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone.forward_features(x)
        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]
            if "x_prenorm" in out:
                prenorm = out["x_prenorm"]
                if prenorm.ndim == 3:
                    return prenorm[:, 0, :]
            if "features" in out:
                feats = out["features"]
                if feats.ndim == 2:
                    return feats
                if feats.ndim == 3:
                    return feats[:, 0, :]
        if isinstance(out, (list, tuple)) and out:
            out = out[0]
        if out.ndim == 2:
            return out
        if out.ndim == 3:
            return out[:, 0, :]
        raise RuntimeError(f"Unexpected timm feature shape: {tuple(out.shape)}")

    def _forward_hf(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        if hasattr(out, "last_hidden_state"):
            h = out.last_hidden_state
            if h.ndim == 3:
                return h[:, 0, :]
            if h.ndim == 2:
                return h
        raise RuntimeError("Unexpected HuggingFace DINOv2 output format.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,C,H,W], got {tuple(x.shape)}")
        if self.backend == "hf":
            return self._forward_hf(x)
        return self._forward_timm(x)

