"""
HiLoS-Cardio: Hierarchical Longitudinal transmural Strain Transformer/GRU
-----------------------------------------------------------------------------
This script trains and evaluates a three-level model on the VVI-derived dataset
produced by `vvi_xml_preprocess.py`. It follows the design discussed for using
transmural Δ-strain (endo - epi) curves, spatial encoding across segments/views,
and temporal encoding across patient visits with early-biased risk prediction.

Levels
------
1) Segment encoder (temporal 1D CNN) over multi-channel curves:
   - channels: Δ (endo-epi), dΔ/dτ, optionally endo and epi.
2) Visit spatial encoder (Transformer) over 8 tokens (4 segments for 2C + 4 for 4C)
   with learned segment/view embeddings → visit embedding.
3) Patient temporal encoder (GRU) over visits with relative-time embeddings →
   per-visit cardiotoxicity risk head (+ optional GLS auxiliary regression).

Losses
------
- Weighted BCE per visit (early-positive visits get higher weight).
- Monotonic regularizer for positive patients (risk should not drop over time).
- Optional auxiliary GLS regression when `gls` is present.

Data expectations
-----------------
Parquet from `vvi_xml_preprocess.py` with rows:
  short_id, study_datetime, view ("2C"/"4C"), cycle_index, segment,
  endo, epi, delta (lists), gls (float), n_samples, ...
An external labels CSV is required with columns:
  short_id, outcome (0/1), [optional] event_visit_index (0-based; earliest visit with event)

Quick start
-----------
python hilos_cardio_model.py \
  --parquet path/to/strain_dataset.parquet \
  --labels-csv path/to/patient_labels.csv \
  --outdir ./models/hilos \
  --epochs 40 --batch 8 --lr 1e-3

Optional SSL pretraining (contrastive on visit embeddings):
python hilos_cardio_model.py --stage ssl ...

Notes
-----
- Visits are defined by (short_id, study_datetime). We require both 2C and 4C
  views with all 4 kept segments each; incomplete visits are skipped.
- For multiple cycles per view, the earliest cycle_index is selected.
- Curve length is resampled to T (default 64).
- Everything is ASCII; minimal dependencies: torch, pandas, numpy, scikit-learn.
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

# ----------------------------- Constants ----------------------------------
SEG_ORDER_4C = [
    "basal_inferoseptal",
    "mid_inferoseptal",
    "mid_anterolateral",
    "basal_anterolateral",
]
SEG_ORDER_2C = [
    "basal_inferior",
    "mid_inferior",
    "mid_anterior",
    "basal_anterior",
]
ALL_SEGMENTS = SEG_ORDER_4C + SEG_ORDER_2C
SEG_ID_MAP = {name: i for i, name in enumerate(ALL_SEGMENTS)}
VIEW_ID_MAP = {"4C": 0, "2C": 1}
N_SEGMENTS_PER_VISIT = 8

# ----------------------------- Utilities ----------------------------------


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resample_curve(y: Sequence[float], T: int) -> np.ndarray:
    """Linearly resample a 1D curve to length T on [0, 1]."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return np.zeros(T, dtype=np.float32)
    if y.size == T:
        return y.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, num=y.size)
    x_new = np.linspace(0.0, 1.0, num=T)
    return np.interp(x_new, x_old, y).astype(np.float32)


def finite_diff(x: np.ndarray) -> np.ndarray:
    """First derivative with simple finite differences, preserving length."""
    if x.size < 2:
        return np.zeros_like(x)
    dx = np.gradient(x)
    return dx.astype(np.float32)


def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean()
    sigma = x.std()
    return (x - mu) / (sigma + eps)


# ----------------------------- Data prep ----------------------------------
@dataclass
class Visit:
    curves: np.ndarray  # [8, T, C]
    gls: Optional[float]
    time_delta: float  # months since first visit
    study_datetime: pd.Timestamp


@dataclass
class PatientSeq:
    short_id: str
    outcome: int
    visits: List[Visit]
    event_visit_index: Optional[int] = None  # earliest visit with event (if known)


def load_labels(labels_csv: Path) -> Dict[str, Tuple[int, Optional[int]]]:
    df = pd.read_csv(labels_csv)
    req = ["short_id", "outcome"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"labels_csv missing column '{c}'")
    labels: Dict[str, Tuple[int, Optional[int]]] = {}
    for _, row in df.iterrows():
        sid = str(row["short_id"])
        outcome = int(row["outcome"])
        event_idx = None
        if "event_visit_index" in df.columns and not pd.isna(row["event_visit_index"]):
            event_idx = int(row["event_visit_index"])
        labels[sid] = (outcome, event_idx)
    return labels


def build_visit_tensor(
    df_visit: pd.DataFrame,
    T: int,
    include_raw_layers: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Build [8, T, C] tensor for a single visit (both views). Returns (tensor, gls_mean).
    Skips if any required segment is missing.
    """
    views = {"4C": SEG_ORDER_4C, "2C": SEG_ORDER_2C}
    segments: List[np.ndarray] = []
    gls_vals: List[float] = []

    for view, order in views.items():
        view_df = df_visit[df_visit["view"] == view]
        if view_df.empty:
            return None, None
        # pick earliest cycle_index for this view
        best_cycle = view_df["cycle_index"].min()
        view_df = view_df[view_df["cycle_index"] == best_cycle]
        # ensure all 4 segments
        present = set(view_df["segment"].tolist())
        if not all(seg in present for seg in order):
            return None, None
        for seg in order:
            row = view_df[view_df["segment"] == seg].iloc[0]
            delta = resample_curve(row["delta"], T)
            d1 = finite_diff(delta)
            channels = [delta, d1]
            if include_raw_layers:
                endo = resample_curve(row["endo"], T)
                epi = resample_curve(row["epi"], T)
                channels.extend([endo, epi])
            feat = np.stack(channels, axis=-1)  # [T, C]
            feat = zscore(feat, eps=1e-5)
            segments.append(feat.astype(np.float32))
            if not pd.isna(row.get("gls", np.nan)):
                gls_vals.append(float(row["gls"]))

    if len(segments) != N_SEGMENTS_PER_VISIT:
        return None, None
    visit_tensor = np.stack(segments, axis=0)  # [8, T, C]
    gls_mean = float(np.mean(gls_vals)) if gls_vals else None
    return visit_tensor, gls_mean


def build_patient_sequences(
    parquet_path: Path,
    labels_csv: Path,
    T: int,
    include_raw_layers: bool = True,
) -> List[PatientSeq]:
    df = pd.read_parquet(parquet_path)
    if "study_datetime" not in df.columns:
        raise ValueError("Parquet missing 'study_datetime'")
    df["study_datetime"] = pd.to_datetime(df["study_datetime"])
    labels = load_labels(labels_csv)

    patients: List[PatientSeq] = []
    for sid, sub in df.groupby("short_id"):
        if sid not in labels:
            continue
        outcome, event_idx = labels[sid]
        visits: List[Visit] = []
        for dt, df_visit in sub.groupby("study_datetime"):
            tensor, gls_val = build_visit_tensor(df_visit, T, include_raw_layers=include_raw_layers)
            if tensor is None:
                continue
            visits.append((dt, tensor, gls_val))
        if not visits:
            continue
        visits.sort(key=lambda x: x[0])
        t0 = visits[0][0]
        visit_objs: List[Visit] = []
        for dt, tensor, gls_val in visits:
            months = (dt - t0).days / 30.0
            visit_objs.append(Visit(curves=tensor, gls=gls_val, time_delta=months, study_datetime=dt))
        patients.append(PatientSeq(short_id=str(sid), outcome=int(outcome), visits=visit_objs, event_visit_index=event_idx))
    return patients


# ----------------------------- Dataset classes ----------------------------
class TransmuralSequenceDataset(Dataset):
    """
    Yields patient-level sequences of visits with padding for batching.
    """

    def __init__(self, patients: List[PatientSeq]):
        self.patients = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int):
        p = self.patients[idx]
        V = len(p.visits)
        T = p.visits[0].curves.shape[1]
        C = p.visits[0].curves.shape[2]
        curves = np.stack([v.curves for v in p.visits], axis=0)  # [V, 8, T, C]
        times = np.array([v.time_delta for v in p.visits], dtype=np.float32)  # [V]
        gls = np.array([v.gls if v.gls is not None else np.nan for v in p.visits], dtype=np.float32)
        return {
            "curves": torch.tensor(curves, dtype=torch.float32),  # [V, 8, T, C]
            "times": torch.tensor(times, dtype=torch.float32),    # [V]
            "gls": torch.tensor(gls, dtype=torch.float32),        # [V]
            "length": V,
            "outcome": p.outcome,
            "event_idx": -1 if p.event_visit_index is None else int(p.event_visit_index),
            "short_id": p.short_id,
        }


def collate_sequences(batch: List[dict]):
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)
    lengths = [b["length"] for b in batch]
    max_len = max(lengths)
    curves = []
    times = []
    gls = []
    mask = []
    outcomes = []
    event_idx = []
    short_ids = []
    for b in batch:
        V, S, T, C = b["curves"].shape
        pad_shape = (max_len - V, S, T, C)
        pad = torch.zeros(pad_shape, dtype=torch.float32)
        curves.append(torch.cat([b["curves"], pad], dim=0))

        pad_t = torch.zeros(max_len - V, dtype=torch.float32)
        times.append(torch.cat([b["times"], pad_t], dim=0))

        pad_g = torch.full((max_len - V,), float("nan"), dtype=torch.float32)
        gls.append(torch.cat([b["gls"], pad_g], dim=0))

        mask.append(torch.tensor([1] * V + [0] * (max_len - V), dtype=torch.float32))
        outcomes.append(b["outcome"])
        event_idx.append(b["event_idx"])
        short_ids.append(b["short_id"])

    curves = torch.stack(curves, dim=0)  # [B, V, 8, T, C]
    times = torch.stack(times, dim=0)    # [B, V]
    gls = torch.stack(gls, dim=0)        # [B, V]
    mask = torch.stack(mask, dim=0)      # [B, V]
    outcomes = torch.tensor(outcomes, dtype=torch.float32)  # [B]
    event_idx = torch.tensor(event_idx, dtype=torch.long)   # [B]
    return {
        "curves": curves,
        "times": times,
        "gls": gls,
        "mask": mask,
        "lengths": lengths,
        "outcomes": outcomes,
        "event_idx": event_idx,
        "short_ids": short_ids,
    }


# ----------------------------- Model components ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 5, dilation: int = 1):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation)
        self.norm = nn.GroupNorm(4, out_ch)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SegmentEncoder(nn.Module):
    def __init__(self, in_ch: int, d_model: int = 64):
        super().__init__()
        self.block1 = ConvBlock(in_ch, 64, k=5, dilation=1)
        self.block2 = ConvBlock(64, 96, k=5, dilation=2)
        self.block3 = ConvBlock(96, d_model, k=3, dilation=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B*V*8, C, T]
        returns: [B*V*8, d_model]
        """
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.pool(h).squeeze(-1)
        return h


class SpatialEncoder(nn.Module):
    def __init__(self, d_model: int = 64, d_visit: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.seg_emb = nn.Embedding(len(ALL_SEGMENTS), d_model)
        self.view_emb = nn.Embedding(len(VIEW_ID_MAP), d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Sequential(
            nn.Linear(2 * d_model, d_visit),
            nn.PReLU(),
            nn.Linear(d_visit, d_visit),
        )

    def forward(self, z_seg: torch.Tensor, seg_ids: torch.Tensor, view_ids: torch.Tensor) -> torch.Tensor:
        """
        z_seg: [B, V, 8, d_model]
        seg_ids: [8] tensor with global segment ids
        view_ids: [8] tensor with view ids (aligned to segments)
        return: visit embeddings [B, V, d_visit]
        """
        B, V, S, D = z_seg.shape
        seg_tokens = z_seg + self.seg_emb(seg_ids).view(1, 1, S, D) + self.view_emb(view_ids).view(1, 1, S, D)
        tokens = seg_tokens.view(B * V, S, D)
        tokens = self.transformer(tokens)
        mean_tok = tokens.mean(dim=1)
        max_tok, _ = tokens.max(dim=1)
        visit = torch.cat([mean_tok, max_tok], dim=-1)
        visit = self.proj(visit)
        visit = visit.view(B, V, -1)
        return visit


class TimeEmbedding(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.PReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B, V] months since baseline
        return self.mlp(t.unsqueeze(-1))


class TemporalEncoder(nn.Module):
    def __init__(self, d_visit: int = 128, d_hidden: int = 128, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=d_visit, hidden_size=d_hidden, num_layers=num_layers, batch_first=True)

    def forward(self, visits: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        visits: [B, V, d_visit]
        lengths: list of actual lengths
        returns: padded outputs [B, V, d_hidden], last_hidden [num_layers, B, d_hidden]
        """
        packed = pack_padded_sequence(visits, lengths=lengths, batch_first=True, enforce_sorted=True)
        packed_out, h = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return out, h


class RiskHead(nn.Module):
    def __init__(self, d_hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h).squeeze(-1)  # logits


class GLSHead(nn.Module):
    def __init__(self, d_hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.PReLU(),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h).squeeze(-1)


class HiLoSCardio(nn.Module):
    def __init__(self, in_ch: int = 4, d_seg: int = 64, d_visit: int = 128, d_hidden: int = 128):
        super().__init__()
        self.segment_encoder = SegmentEncoder(in_ch=in_ch, d_model=d_seg)
        self.spatial = SpatialEncoder(d_model=d_seg, d_visit=d_visit)
        self.time_emb = TimeEmbedding(d_model=d_visit)
        self.temporal = TemporalEncoder(d_visit=d_visit, d_hidden=d_hidden)
        self.risk_head = RiskHead(d_hidden=d_hidden)
        self.gls_head = GLSHead(d_hidden=d_hidden)
        # Static segment/view ids (S=8)
        seg_ids = [SEG_ID_MAP[s] for s in ALL_SEGMENTS]
        view_ids = [VIEW_ID_MAP["4C"]] * 4 + [VIEW_ID_MAP["2C"]] * 4
        self.register_buffer("seg_ids", torch.tensor(seg_ids, dtype=torch.long), persistent=False)
        self.register_buffer("view_ids", torch.tensor(view_ids, dtype=torch.long), persistent=False)

    def forward(self, curves: torch.Tensor, times: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        curves: [B, V, 8, T, C]
        times: [B, V] (months since baseline)
        lengths: list of actual visit counts
        returns:
          risk_logits: [B, V]
          gls_pred: [B, V]
          mask: [B, V] (1 for valid, 0 for pad)
        """
        B, V, S, T, C = curves.shape
        x = curves.permute(0, 1, 3, 2, 4)  # [B, V, T, S, C]
        x = x.reshape(B * V * S, T, C).permute(0, 2, 1)  # [B*V*S, C, T]
        z = self.segment_encoder(x)  # [B*V*S, d_seg]
        z = z.view(B, V, S, -1)
        visit = self.spatial(z, self.seg_ids, self.view_ids)  # [B, V, d_visit]
        visit = visit + self.time_emb(times)  # time conditioning
        h, _ = self.temporal(visit, lengths)
        risk_logits = self.risk_head(h)
        gls_pred = self.gls_head(h)
        max_len = h.size(1)
        mask = torch.zeros(B, max_len, device=h.device)
        for i, L in enumerate(lengths):
            mask[i, :L] = 1.0
        return risk_logits, gls_pred, mask


# ----------------------------- Losses -------------------------------------
def weighted_bce_per_visit(
    logits: torch.Tensor,
    outcomes: torch.Tensor,
    mask: torch.Tensor,
    lengths: List[int],
    early_weight_pos: float = 2.0,
    late_weight_pos: float = 1.0,
) -> torch.Tensor:
    """
    Assign higher weight to early visits of positive patients.
    logits: [B, V], outcomes: [B], mask: [B, V]
    """
    B, V = logits.shape
    weights = torch.ones_like(logits)
    for i in range(B):
        if outcomes[i] > 0.5:
            L = lengths[i]
            if L <= 1:
                weights[i, 0] = early_weight_pos
            else:
                w = torch.linspace(early_weight_pos, late_weight_pos, steps=L, device=logits.device)
                weights[i, :L] = w
    loss = F.binary_cross_entropy_with_logits(logits, outcomes.view(-1, 1).expand_as(logits), weight=weights, reduction="none")
    loss = (loss * mask).sum() / (mask.sum() + 1e-6)
    return loss


def monotonic_regularizer(probs: torch.Tensor, outcomes: torch.Tensor, lengths: List[int]) -> torch.Tensor:
    """
    Encourage non-decreasing risk over visits for positive patients.
    probs: [B, V]
    """
    reg = 0.0
    count = 0
    for i in range(probs.size(0)):
        if outcomes[i] < 0.5:
            continue
        L = lengths[i]
        if L < 2:
            continue
        diff = probs[i, :L - 1] - probs[i, 1:L]
        reg = reg + F.relu(diff).mean()
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=probs.device)
    return reg / count


def gls_aux_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred: [B, V], target: [B, V] (nan where missing), mask: [B, V]
    """
    valid = mask * (~torch.isnan(target))
    if valid.sum() < 1:
        return torch.tensor(0.0, device=pred.device)
    loss = F.smooth_l1_loss(pred[valid.bool()], target[valid.bool()])
    return loss


# ----------------------------- Training -----------------------------------
@torch.no_grad()
def evaluate(model: HiLoSCardio, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs = []
    all_labels = []
    for batch in loader:
        curves = batch["curves"].to(device)
        times = batch["times"].to(device)
        lengths = batch["lengths"]
        outcomes = batch["outcomes"].to(device)
        logits, _, _ = model(curves, times, lengths)
        probs = torch.sigmoid(logits)
        # patient-level score = max over visits
        max_prob = torch.stack([probs[i, :lengths[i]].max() for i in range(probs.size(0))], dim=0)
        all_probs.append(max_prob.cpu())
        all_labels.append(outcomes.cpu())
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float("nan")
    return auc


def run_supervised(
    model: HiLoSCardio,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_mono: float,
    lambda_gls: float,
    outdir: Path,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_auc = -1.0
    outdir.mkdir(parents=True, exist_ok=True)
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            curves = batch["curves"].to(device)
            times = batch["times"].to(device)
            lengths = batch["lengths"]
            mask = batch["mask"].to(device)
            outcomes = batch["outcomes"].to(device)
            gls_tgt = batch["gls"].to(device)

            logits, gls_pred, _ = model(curves, times, lengths)
            probs = torch.sigmoid(logits)
            bce = weighted_bce_per_visit(logits, outcomes, mask, lengths)
            mono = monotonic_regularizer(probs, outcomes, lengths)
            aux = gls_aux_loss(gls_pred, gls_tgt, mask) if lambda_gls > 0 else torch.tensor(0.0, device=device)
            loss = bce + lambda_mono * mono + lambda_gls * aux

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        val_auc = evaluate(model, val_loader, device)
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"[Supervised] Epoch {ep:03d} loss={avg_loss:.4f} val_auc={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            ckpt = outdir / "supervised_best.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  -> Saved best checkpoint to {ckpt}")


# ----------------------------- SSL Pretrain -------------------------------
def visit_augment(x: torch.Tensor, time_mask_frac: float = 0.15, jitter_std: float = 0.01) -> torch.Tensor:
    """
    x: [B, V, 8, T, C]
    Applies light jitter and random temporal masking.
    """
    y = x.clone()
    y = y + torch.randn_like(y) * jitter_std
    B, V, S, T, C = y.shape
    mask_len = max(1, int(time_mask_frac * T))
    for b in range(B):
        for v in range(V):
            start = np.random.randint(0, max(1, T - mask_len + 1))
            y[b, v, :, start : start + mask_len, :] = 0.0
    return y


def run_ssl_pretrain(
    model: HiLoSCardio,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    outdir: Path,
    temperature: float = 0.1,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    outdir.mkdir(parents=True, exist_ok=True)
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for batch in train_loader:
            curves = batch["curves"].to(device)
            times = batch["times"].to(device)
            lengths = batch["lengths"]
            # two augmentations
            c1 = visit_augment(curves)
            c2 = visit_augment(curves)
            v1, _, _ = model(c1, times, lengths)
            v2, _, _ = model(c2, times, lengths)
            # only use visit-level max pooling for contrast
            emb1 = []
            emb2 = []
            for i, L in enumerate(lengths):
                emb1.append(v1[i, :L].mean(dim=0))
                emb2.append(v2[i, :L].mean(dim=0))
            z1 = torch.stack(emb1, dim=0)
            z2 = torch.stack(emb2, dim=0)
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
            reps = torch.cat([z1, z2], dim=0)  # [2B, d]
            sim = reps @ reps.t() / temperature
            mask = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
            sim.masked_fill_(mask, -1e9)
            targets = torch.cat([torch.arange(z1.size(0), 2 * z1.size(0), device=sim.device),
                                 torch.arange(0, z1.size(0), device=sim.device)], dim=0)
            loss = F.cross_entropy(sim, targets)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += loss.item()
            n += 1
        avg = loss_sum / max(1, n)
        print(f"[SSL] Epoch {ep:03d} loss={avg:.4f}")
    ckpt = outdir / "ssl_pretrain.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved SSL checkpoint to {ckpt}")


# ----------------------------- CLI ----------------------------------------
def split_patients(patients: List[PatientSeq], val_frac: float = 0.2, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(patients))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - val_frac))
    train_idx = idx[:cut]
    val_idx = idx[cut:]
    train = [patients[i] for i in train_idx]
    val = [patients[i] for i in val_idx]
    return train, val


def parse_args():
    ap = argparse.ArgumentParser(description="HiLoS-Cardio training script")
    ap.add_argument("--parquet", type=str, required=True, help="Path to strain_dataset.parquet")
    ap.add_argument("--labels-csv", type=str, required=True, help="CSV with columns short_id,outcome[,event_visit_index]")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for checkpoints")
    ap.add_argument("--stage", type=str, choices=["ssl", "supervised"], default="supervised")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--ssl-epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--no-raw-layers", action="store_true", help="Exclude raw endo/epi channels (keep Δ and dΔ only)")
    ap.add_argument("--lambda-mono", type=float, default=0.5)
    ap.add_argument("--lambda-gls", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    include_raw = not args.no_raw_layers
    patients = build_patient_sequences(
        parquet_path=Path(args.parquet),
        labels_csv=Path(args.labels_csv),
        T=args.T,
        include_raw_layers=include_raw,
    )
    if not patients:
        raise RuntimeError("No patient sequences built. Check data/labels.")

    train_patients, val_patients = split_patients(patients, val_frac=args.val_frac, seed=args.seed)
    train_ds = TransmuralSequenceDataset(train_patients)
    val_ds = TransmuralSequenceDataset(val_patients)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_sequences)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_sequences)

    in_channels = 2 if not include_raw else 4  # Δ, dΔ, [endo, epi]
    model = HiLoSCardio(in_ch=in_channels)
    model.to(device)

    outdir = Path(args.outdir)
    if args.stage == "ssl":
        run_ssl_pretrain(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=args.ssl_epochs,
            lr=args.lr,
            outdir=outdir,
        )
    else:
        run_supervised(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            lambda_mono=args.lambda_mono,
            lambda_gls=args.lambda_gls,
            outdir=outdir,
        )


if __name__ == "__main__":
    main()
