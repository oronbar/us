"""
Patient-conditioned spatio-temporal encoder with compression
and longitudinal training for GLS supervision.

This script implements the model and training schedule described earlier:
  (2) Model: patient-conditioned spatio-temporal encoder with compression
  (3) Training strategy: Stage A (self-supervised) + Stage B (GLS-supervised)

INPUT DATA
---------
- A Parquet file produced by vvi_xml_preprocess.py with rows at the granularity
  of (patient, study, view, cycle, segment), containing the columns:
    short_id, study_datetime, view, dicom_file, cycle_index, segment,
    time, endo, epi, myo, delta, gls

- For modeling we use the segment-level Δ-curves (Endo - Epi). We resample each
  curve to a fixed length T (default 64). Each training sample aggregates the 4
  kept segments for a (short_id, view, study/cycle) unit.

WHAT THIS SCRIPT DOES
---------------------
1) Builds a PyTorch Dataset that groups rows by (patient, view, study, cycle)
   into a single example with 4 segments of Δ-curves + GLS label for the cycle.
2) Stage A pretraining:
   - Masked Time Modeling (MTM) on segment encoders
   - Curve reconstruction via a small decoder (autoencoder regularization)
   - Contrastive (SimCLR-style) on visit embeddings with simple augmentations
3) Stage B fine-tuning:
   - Compute per-patient baseline embedding b_p from earliest visit (average of its cycles)
   - Patient-conditioned GLS regression head (Huber loss) using (v, b_p)

USAGE (PowerShell / bash)
-------------------------
python patient_conditioned_strain_model.py \
  --parquet "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Tags\\VVI\\processed\\strain_dataset.parquet" \
  --outdir  "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Tags\\VVI\\models" \
  --stage pretrain  --epochs 50

# After pretraining
python patient_conditioned_strain_model.py \
  --parquet "C:\\...\\strain_dataset.parquet" \
  --outdir  "C:\\...\\models" \
  --stage finetune --epochs 30 --load "C:\\...\\models\\pretrain_best.pt"

REQUIREMENTS
------------
python -m pip install torch pandas numpy pyarrow scikit-learn

Notes
-----
- We train per (view, cycle). Later you can aggregate per-visit/ per-patient.
- For simplicity, we drop examples that miss any of the 4 required segments.
- You can adjust T, batch size, and loss weights via CLI flags.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

# ----------------------------- Config -------------------------------------
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

VIEW_TO_SEGS = {"4C": SEG_ORDER_4C, "2C": SEG_ORDER_2C}

# ----------------------------- Utils --------------------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resample_curve(y: List[float], T: int) -> np.ndarray:
    """Linearly resample a 1D list/array to length T over normalized index domain."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return np.zeros(T, dtype=float)
    if n == T:
        return y.copy()
    x_old = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, T)
    return np.interp(x_new, x_old, y)


# ----------------------------- Dataset ------------------------------------
@dataclass
class GroupKey:
    short_id: str
    view: str
    study_uid: str
    dicom_file: str
    cycle_index: int

class StrainParquetDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        T: int = 64,
        view_filter: Optional[str] = None,
        train: bool = True,
    ):
        super().__init__()
        self.T = T
        if view_filter is not None:
            df = df[df["view"] == view_filter]
        # Keep only needed columns
        req = [
            "short_id",
            "study_instance_uid",
            "view",
            "dicom_file",
            "cycle_index",
            "segment",
            "delta",
            "gls",
            "study_datetime",
        ]
        for c in req:
            if c not in df.columns:
                raise ValueError(f"Missing column {c} in parquet")
        # group by key and ensure 4 segments exist
        self.groups: List[GroupKey] = []
        self.samples: List[dict] = []
        # ensure delta lists are arrays
        # Sort by datetime for stable grouping
        df = df.copy()
        df.sort_values(by=["short_id", "study_datetime", "view", "dicom_file", "cycle_index"], inplace=True)
        for (sid, view, uid, dfile, ci), sub in df.groupby(
            ["short_id", "view", "study_instance_uid", "dicom_file", "cycle_index"], sort=False
        ):
            seg_order = VIEW_TO_SEGS.get(view, SEG_ORDER_4C)
            present = set(sub["segment"].tolist())
            if not all(seg in present for seg in seg_order):
                continue  # drop incomplete
            # build array [S,T]
            S = len(seg_order)
            X = np.zeros((S, T), dtype=np.float32)
            for si, seg in enumerate(seg_order):
                row = sub[sub["segment"] == seg].iloc[0]
                X[si] = resample_curve(row["delta"], T)
            # GLS label (per cycle) — we pick it from any segment row of the group (identical)
            gls_vals = sub["gls"].dropna().values
            y_gls = float(gls_vals[0]) if len(gls_vals) else np.nan
            self.groups.append(GroupKey(sid, view, uid, dfile, int(ci)))
            self.samples.append({
                "X": X,  # [S,T]
                "gls": y_gls,
                "short_id": sid,
                "view": view,
                "study_uid": uid,
                "dicom_file": dfile,
                "cycle_index": int(ci),
                "study_datetime": pd.to_datetime(sub["study_datetime"].iloc[0]) if "study_datetime" in sub else None,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        X = torch.tensor(s["X"], dtype=torch.float32)  # [S,T]
        y = torch.tensor(s["gls"], dtype=torch.float32)
        return X, y, s


# ----------------------------- Augmentations -------------------------------
class CurveAug:
    """
    Lightweight augmentations to combat data scarcity:
      - per-sample/segment scaling
      - small circular time shifts
      - Gaussian jitter
      - random time masking (temporal dropout)
      - optional segment dropout (simulate missing/noisy segments)
    """
    def __init__(
        self,
        jitter_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        warp_max_shift: int = 3,
        time_mask_frac: float = 0.15,
        seg_dropout: float = 0.1,
    ):
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.warp_max = warp_max_shift
        self.time_mask_frac = time_mask_frac
        self.seg_dropout = seg_dropout

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B,S,T] or [S,T]
        single = False
        if X.dim() == 2:
            X = X.unsqueeze(0)
            single = True
        B, S, T = X.shape

        # per-segment scaling (helps robustness to gain differences)
        scales = torch.empty(B, S, 1, device=X.device).uniform_(*self.scale_range)
        Y = X * scales

        # circular time shift per sample
        if self.warp_max > 0:
            shifts = torch.randint(low=-self.warp_max, high=self.warp_max + 1, size=(B,), device=X.device)
            Y = torch.stack([torch.roll(Y[i], shifts=int(shifts[i].item()), dims=-1) for i in range(B)], dim=0)

        # time masking: zero-out a small temporal window per sample (simulates dropped frames)
        win = max(1, int(self.time_mask_frac * T))
        if win > 0:
            for i in range(B):
                start = torch.randint(low=0, high=max(1, T - win), size=(1,), device=X.device).item()
                Y[i, :, start:start + win] = 0.0

        # optional segment dropout: with prob p, zero one segment per sample
        if self.seg_dropout > 0:
            drop_mask = torch.rand(B, device=X.device) < self.seg_dropout
            if drop_mask.any():
                seg_ids = torch.randint(low=0, high=S, size=(drop_mask.sum(),), device=X.device)
                for idx, seg_id in zip(drop_mask.nonzero().flatten(), seg_ids):
                    Y[int(idx.item()), int(seg_id.item()), :] = 0.0

        # Gaussian jitter
        Y = Y + torch.randn_like(Y) * self.jitter_std

        return Y.squeeze(0) if single else Y


# ----------------------------- Model --------------------------------------
class SegmentEncoder(nn.Module):
    def __init__(self, d=48, T=64, nheads=4, nlayers=2):
        super().__init__()
        self.T = T
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2, dilation=1), nn.ReLU(),
            nn.Conv1d(32, 48, 5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(48, d, 5, padding=8, dilation=4)
        )
        self.pos = nn.Parameter(torch.randn(1, d, T) * 0.01)
        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=nheads, batch_first=True)
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

    def forward(self, x):
        # x: [B,S,T]
        B, S, T = x.shape
        x = x.view(B * S, 1, T)          # -> [B*S, 1, T]
        h = self.conv(x) + self.pos      # -> [B*S, d, T]
        h = h.transpose(1, 2)            # -> [B*S, T, d]  (time is dim=1)
        h = self.tr(h)                   # -> [B*S, T, d]  (batch_first=True)
        z = h.mean(dim=1)                # -> [B*S, d]     (mean over time)
        z = z.view(B, S, -1)             # -> [B, S, d]
        return z, h.view(B, S, T, -1) # token and time states


class VisitAggregator(nn.Module):
    def __init__(self, d=48, D=64, nheads=4, nlayers=2, n_seg_tokens=4):
        super().__init__()
        self.seg_type_emb = nn.Embedding(16, d)
        self.view_emb = nn.Embedding(4, D)  # match visit embedding dimension
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=nheads, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.proj = nn.Linear(d, D)
        self.n_seg_tokens = n_seg_tokens

    def forward(self, z_seg, seg_ids, view_id):
        # z_seg: [B,S,d], seg_ids: [B,S], view_id: [B]
        B, S, d = z_seg.shape
        h = z_seg + self.seg_type_emb(seg_ids)
        h = self.tr(h)                # [B,S,d]
        v = self.proj(h.mean(dim=1))  # [B,D]
        # add view token (bias-like)
        v = v + self.view_emb(view_id)
        return v, h


class Decoder(nn.Module):
    def __init__(self, d=48, T=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, T)
        )

    def forward(self, z_seg):
        # z_seg: [B,S,d]
        B, S, d = z_seg.shape
        pred = self.mlp(z_seg)  # [B,S,T]
        return pred


class MTMHead(nn.Module):
    def __init__(self, d=48):
        super().__init__()
        self.proj = nn.Linear(d, 1)

    def forward(self, h_time):
        # h_time: [B,S,T,d]
        return self.proj(h_time).squeeze(-1)  # [B,S,T]


class GLSHead(nn.Module):
    def __init__(self, D=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * D + 2 * D, 64),  # [v, b, v-b, v*b]
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, v, b):
        x = torch.cat([v, b, v - b, v * b], dim=-1)
        y = self.mlp(x).squeeze(-1)
        return y


class Model(nn.Module):
    def __init__(self, d=48, D=64, T=64, nheads=4):
        super().__init__()
        self.T = T
        self.seg_encoder = SegmentEncoder(d=d, T=T, nheads=nheads)
        self.aggregator = VisitAggregator(d=d, D=D)
        self.decoder = Decoder(d=d, T=T)
        self.mtm = MTMHead(d=d)
        self.gls_head = GLSHead(D=D)

    def forward(self, X, seg_ids, view_id):
        # X: [B,S,T]
        z_seg, h_time = self.seg_encoder(X)
        v, h_seg = self.aggregator(z_seg, seg_ids, view_id)
        recon = self.decoder(z_seg)          # [B,S,T]
        mtm_pred = self.mtm(h_time)          # [B,S,T]
        return v, z_seg, h_time, recon, mtm_pred


# ----------------------------- Helpers ------------------------------------
SEG_ID_MAP = {
    # 4C
    "basal_inferoseptal": 0,
    "mid_inferoseptal": 1,
    "mid_anterolateral": 2,
    "basal_anterolateral": 3,
    # 2C
    "basal_inferior": 4,
    "mid_inferior": 5,
    "mid_anterior": 6,
    "basal_anterior": 7,
}
VIEW_ID_MAP = {"2C": 0, "4C": 1}


def batchify(samples: List[Tuple[torch.Tensor, torch.Tensor, dict]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
    X = torch.stack([s[0] for s in samples], dim=0)  # [B,S,T]
    y = torch.stack([s[1] for s in samples], dim=0)  # [B]
    meta = [s[2] for s in samples]
    seg_ids = []
    view_ids = []
    for m in meta:
        seg_order = VIEW_TO_SEGS.get(m["view"], SEG_ORDER_4C)
        seg_ids.append(torch.tensor([SEG_ID_MAP[s] for s in seg_order], dtype=torch.long))
        view_ids.append(torch.tensor(VIEW_ID_MAP.get(m["view"], 1), dtype=torch.long))
    seg_ids = torch.stack(seg_ids, dim=0)
    view_ids = torch.stack(view_ids, dim=0)
    return X, y, seg_ids, view_ids, meta


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B, D = z1.size()
    reps = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = reps @ reps.t() / temperature
    # mask self-similarity
    mask = torch.eye(2 * B, device=z1.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)
    # positives are (i, i+B) and (i+B, i)
    targets = torch.cat([torch.arange(B, 2 * B, device=z1.device), torch.arange(0, B, device=z1.device)], dim=0)
    loss = F.cross_entropy(sim, targets)
    return loss


# ----------------------------- Baseline utils ------------------------------
@torch.no_grad()
def compute_patient_baselines(model: Model, loader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    """Compute per-patient baseline embedding b_p from *earliest* visit cycles.
    For each short_id, we find their minimum study_datetime and average v over those cycles.
    """
    model.eval()
    # Collect per-patient earliest datetime
    earliest: Dict[str, pd.Timestamp] = {}
    storage: Dict[str, List[torch.Tensor]] = {}
    aug = CurveAug()  # not used here
    for X, y, seg_ids, view_ids, meta in loader:
        # choose by datetime
        dt = [m.get("study_datetime") for m in meta]
        sids = [m["short_id"] for m in meta]
        X = X.to(device)
        seg_ids = seg_ids.to(device)
        view_ids = view_ids.to(device)
        v, _, _, _, _ = model(X, seg_ids, view_ids)
        for i, sid in enumerate(sids):
            dti = dt[i]
            if sid not in earliest or (pd.notna(dti) and dti < earliest[sid]):
                earliest[sid] = dti
    # Second pass to collect vectors from earliest date only
    for X, y, seg_ids, view_ids, meta in loader:
        dt = [m.get("study_datetime") for m in meta]
        sids = [m["short_id"] for m in meta]
        X = X.to(device)
        seg_ids = seg_ids.to(device)
        view_ids = view_ids.to(device)
        v, _, _, _, _ = model(X, seg_ids, view_ids)
        for i, sid in enumerate(sids):
            if dt[i] == earliest.get(sid):
                storage.setdefault(sid, []).append(v[i].detach().cpu())
    baselines: Dict[str, torch.Tensor] = {}
    for sid, vecs in storage.items():
        if len(vecs):
            baselines[sid] = torch.stack(vecs, dim=0).mean(dim=0)
    return baselines


# ----------------------------- Training Loops ------------------------------

def train_stage_a(
    model: Model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    scheduler: str = "cosine",
    w_mtm: float = 1.0,
    w_recon: float = 1.0,
    w_contrast: float = 0.1,
    outpath: Optional[Path] = None,
    writer: Optional[SummaryWriter] = None,
    global_step_start: int = 0,
):
    aug = CurveAug()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = None
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr * 0.1
        )
    best_val = 1e9
    global_step = global_step_start
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = mtm_sum = recon_sum = contrast_sum = 0.0
        n_batches = 0
        for X, y, seg_ids, view_ids, meta in train_loader:
            X = X.to(device)
            seg_ids = seg_ids.to(device)
            view_ids = view_ids.to(device)

            # two views for contrastive
            X1 = aug(X.clone())
            X2 = aug(X.clone())

            v1, z1, h1, recon1, mtm1 = model(X1, seg_ids, view_ids)
            v2, z2, h2, recon2, mtm2 = model(X2, seg_ids, view_ids)

            # MTM: randomly mask positions and predict original
            B, S, T = X.shape
            mask = torch.zeros(B, S, T, device=device).bernoulli_(0.15).bool()
            mtm_loss = F.mse_loss(mtm1[mask], X[mask]) + F.mse_loss(mtm2[mask], X[mask])

            # Reconstruction loss (autoencoder)
            recon_loss = F.mse_loss(recon1, X) + F.mse_loss(recon2, X)

            # Contrastive on visit embeddings
            contrast_loss = nt_xent_loss(v1, v2)

            loss = w_mtm * mtm_loss + w_recon * recon_loss + w_contrast * contrast_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += loss.item()
            mtm_sum += mtm_loss.item()
            recon_sum += recon_loss.item()
            contrast_sum += contrast_loss.item()
            n_batches += 1
            if writer is not None:
                writer.add_scalar("trainA/total", loss.item(), global_step)
                writer.add_scalar("trainA/mtm", mtm_loss.item(), global_step)
                writer.add_scalar("trainA/recon", recon_loss.item(), global_step)
                writer.add_scalar("trainA/contrast", contrast_loss.item(), global_step)
                global_step += 1

        # validation
        model.eval()
        with torch.no_grad():
            vloss = v_mtm = v_recon = 0.0
            nval = 0
            for X, y, seg_ids, view_ids, meta in val_loader:
                X = X.to(device)
                seg_ids = seg_ids.to(device)
                view_ids = view_ids.to(device)
                v, z, h, recon, mtm = model(X, seg_ids, view_ids)
                B, S, T = X.shape
                mask = torch.zeros(B, S, T, device=device).bernoulli_(0.15).bool()
                mtm_loss = F.mse_loss(mtm[mask], X[mask])
                recon_loss = F.mse_loss(recon, X)
                loss = w_mtm * mtm_loss + w_recon * recon_loss
                vloss += loss.item()
                v_mtm += mtm_loss.item()
                v_recon += recon_loss.item()
                nval += 1
        tr = loss_sum / max(1, n_batches)
        vl = vloss / max(1, nval)
        if writer is not None:
            writer.add_scalar("valA/total", vl, ep)
            writer.add_scalar("valA/mtm", v_mtm / max(1, nval), ep)
            writer.add_scalar("valA/recon", v_recon / max(1, nval), ep)
            writer.add_scalar("trainA/lr", opt.param_groups[0]["lr"], ep)
        print(f"[StageA] Epoch {ep:03d} train={tr:.4f} val={vl:.4f}")
        if outpath is not None and vl < best_val:
            best_val = vl
            outpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), outpath)
            print(f"  -> Saved best StageA to {outpath}")
        if sched is not None:
            sched.step()


def train_stage_b(
    model: Model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 5e-4,
    outpath: Optional[Path] = None,
    writer: Optional[SummaryWriter] = None,
    global_step_start: int = 0,
):
    # Freeze encoder by default? We'll allow fine-tuning all
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = 1e9

    # Precompute baselines from the current model (on both train+val to get b_p for all IDs)
    union_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False, collate_fn=batchify)
    baselines = compute_patient_baselines(model, union_loader, device)

    def gls_epoch(loader, train: bool, step_offset: int = 0):
        if train:
            model.train()
        else:
            model.eval()
        loss_sum = 0.0
        n = 0
        for X, y, seg_ids, view_ids, meta in loader:
            X = X.to(device)
            y = y.to(device)
            seg_ids = seg_ids.to(device)
            view_ids = view_ids.to(device)
            v, z, h, recon, mtm = model(X, seg_ids, view_ids)
            # bring baselines
            b_list = []
            for m in meta:
                sid = m["short_id"]
                if sid not in baselines:
                    # fall back to zero vector if baseline not found
                    b_list.append(torch.zeros_like(v[0]))
                else:
                    b_list.append(baselines[sid].to(device))
            b = torch.stack(b_list, dim=0)
            y_hat = model.gls_head(v, b)
            # Huber is robust to noise
            loss = F.smooth_l1_loss(y_hat, y)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if writer is not None:
                    writer.add_scalar("trainB/gls_loss", loss.item(), step_offset + n)
            loss_sum += loss.item(); n += 1
        return loss_sum / max(1, n)

    global_step = global_step_start
    for ep in range(1, epochs + 1):
        tr = gls_epoch(train_loader, train=True, step_offset=global_step)
        global_step += len(train_loader)
        vl = gls_epoch(val_loader, train=False)
        if writer is not None:
            writer.add_scalar("valB/gls_loss", vl, ep)
        print(f"[StageB] Epoch {ep:03d} GLS train={tr:.4f} val={vl:.4f}")
        if outpath is not None and vl < best_val:
            best_val = vl
            outpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), outpath)
            print(f"  -> Saved best StageB to {outpath}")


# ----------------------------- CLI ----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--stage", type=str, choices=["pretrain", "finetune"], required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate for pretrain")
    ap.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "cosine"],
        default="cosine",
        help="LR scheduler for pretrain",
    )
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--load", type=str, default=None, help="Path to .pt checkpoint to load")
    ap.add_argument("--logdir", type=str, default=None, help="Optional TensorBoard log directory")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_parquet(args.parquet)

    # Split by patient to avoid leakage
    patients = df["short_id"].dropna().unique().tolist()
    train_ids, val_ids = train_test_split(patients, test_size=0.2, random_state=42)
    df_train = df[df["short_id"].isin(train_ids)].copy()
    df_val = df[df["short_id"].isin(val_ids)].copy()

    train_ds = StrainParquetDataset(df_train, T=args.T)
    val_ds = StrainParquetDataset(df_val, T=args.T)

    collate = batchify
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    model = Model(T=args.T)
    if args.load:
        print(f"Loading checkpoint from {args.load}")
        model.load_state_dict(torch.load(args.load, map_location="cpu"))

    model.to(device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=args.logdir) if args.logdir else None

    if args.stage == "pretrain":
        outpath = outdir / "pretrain_best.pt"
        train_stage_a(
            model,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            scheduler=args.scheduler,
            outpath=outpath,
            writer=writer,
        )
    else:
        outpath = outdir / "finetune_best.pt"
        train_stage_b(
            model,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            outpath=outpath,
            writer=writer,
        )


if __name__ == "__main__":
    main()
