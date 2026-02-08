"""
Train a temporal model on per-frame embeddings to predict cine-level GLS.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ichilov_pipeline2_utils import (
    add_gls_from_report,
    resolve_patient_column,
    resolve_time_column,
)

try:
    from sklearn.model_selection import GroupShuffleSplit
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "scikit-learn is required. Install with: .venv\\Scripts\\python -m pip install scikit-learn"
    ) from exc

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_train_temporal_gls")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_embedding(val: object) -> Optional[np.ndarray]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, np.ndarray):
        return val.astype(np.float32)
    if isinstance(val, list):
        return np.asarray(val, dtype=np.float32)
    if isinstance(val, str):
        try:
            parsed = eval(val, {"__builtins__": {}})
            return np.asarray(parsed, dtype=np.float32)
        except Exception:
            return None
    return None


def _load_embeddings(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)


def _cine_key_column(df: pd.DataFrame) -> str:
    for col in ("source_dicom", "cropped_dicom", "dicom", "dicom_path"):
        if col in df.columns:
            return col
    raise ValueError("No cine identifier column found (expected source_dicom/cropped_dicom).")


@dataclass
class CineItem:
    embeddings: np.ndarray
    times: np.ndarray
    frame_idx: np.ndarray
    gls: float
    patient_id: str


class CineDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int, use_phase: bool) -> None:
        self.seq_len = int(seq_len)
        self.use_phase = bool(use_phase)
        self.items: List[CineItem] = []

        emb_col = "embedding"
        if emb_col not in df.columns:
            raise ValueError("Input dataframe missing 'embedding' column.")

        cine_col = _cine_key_column(df)
        patient_col = resolve_patient_column(df)
        if patient_col is None:
            logger.warning("No patient column found; using cine id as group key.")
            patient_col = cine_col

        if "frame_index" not in df.columns:
            df = df.copy()
            df["frame_index"] = df.groupby(cine_col).cumcount()
        if "frame_time" not in df.columns:
            df = df.copy()
            df["frame_time"] = df["frame_index"].astype(float)

        for _, grp in df.groupby(cine_col):
            emb_list = []
            times = []
            frames = []
            for _, row in grp.iterrows():
                emb = _parse_embedding(row[emb_col])
                if emb is None:
                    continue
                emb_list.append(emb)
                times.append(float(row.get("frame_time", 0.0)))
                frames.append(int(row.get("frame_index", len(frames))))
            if not emb_list:
                continue
            gls_vals = grp.get("gls")
            gls = None
            if gls_vals is not None:
                gls = pd.to_numeric(gls_vals, errors="coerce").dropna().values
                gls = float(gls[0]) if len(gls) else None
            if gls is None or np.isnan(gls):
                continue
            emb_arr = np.stack(emb_list, axis=0)
            times_arr = np.asarray(times, dtype=np.float32)
            frames_arr = np.asarray(frames, dtype=np.int64)
            patient_id = str(grp[patient_col].iloc[0]) if patient_col in grp.columns else "unknown"
            self.items.append(CineItem(embeddings=emb_arr, times=times_arr, frame_idx=frames_arr, gls=gls, patient_id=patient_id))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        emb = item.embeddings
        times = item.times
        frame_idx = item.frame_idx
        n = emb.shape[0]
        if n >= self.seq_len:
            sel = np.linspace(0, n - 1, self.seq_len)
            sel = np.clip(np.round(sel).astype(int), 0, n - 1)
            emb = emb[sel]
            times = times[sel]
            frame_idx = frame_idx[sel]
            mask = np.ones(self.seq_len, dtype=np.bool_)
        else:
            pad = self.seq_len - n
            emb = np.concatenate([emb, np.zeros((pad, emb.shape[1]), dtype=emb.dtype)], axis=0)
            times = np.concatenate([times, np.zeros(pad, dtype=times.dtype)], axis=0)
            frame_idx = np.concatenate([frame_idx, np.zeros(pad, dtype=frame_idx.dtype)], axis=0)
            mask = np.zeros(self.seq_len, dtype=np.bool_)
            mask[:n] = True

        if times.max() > 0:
            times_norm = times / times.max()
        else:
            times_norm = times
        delta_t = np.zeros_like(times_norm)
        delta_t[1:] = np.diff(times_norm)

        phase = None
        if self.use_phase:
            ed = None
            es = None
            phase = np.zeros_like(times_norm)
            if ed is not None and es is not None and es > ed:
                phase = (frame_idx - ed) / float(es - ed)
                phase = np.clip(phase, 0.0, 1.0)

        return (
            torch.tensor(emb, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(times_norm, dtype=torch.float32),
            torch.tensor(delta_t, dtype=torch.float32),
            torch.tensor(phase, dtype=torch.float32) if phase is not None else None,
            torch.tensor(item.gls, dtype=torch.float32),
        )


class Time2Vec(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(dim - 1))
        self.b = nn.Parameter(torch.zeros(dim - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B,T]
        v0 = self.w0 * t + self.b0
        v1 = torch.sin(t.unsqueeze(-1) * self.w + self.b)
        return torch.cat([v0.unsqueeze(-1), v1], dim=-1)


class TemporalModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        model: str = "transformer",
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_delta_t: bool = True,
        use_phase: bool = False,
        time_embedding: str = "relative",
    ) -> None:
        super().__init__()
        self.model_type = model
        self.use_delta_t = use_delta_t
        self.use_phase = use_phase
        self.time_embedding = time_embedding

        self.input_proj = nn.Linear(input_dim, model_dim) if input_dim != model_dim else nn.Identity()
        self.delta_proj = nn.Linear(1, model_dim) if use_delta_t else None
        self.phase_proj = nn.Linear(1, model_dim) if use_phase else None

        if time_embedding == "absolute":
            self.pos_embed = nn.Embedding(512, model_dim)
            self.time2vec = None
        elif time_embedding == "time2vec":
            self.pos_embed = None
            self.time2vec = Time2Vec(model_dim)
        else:
            self.pos_embed = None
            self.time2vec = None

        if model == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif model == "gru":
            self.encoder = nn.GRU(
                input_size=model_dim,
                hidden_size=model_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        elif model == "tcn":
            layers = []
            for _ in range(num_layers):
                layers.append(nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            self.encoder = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown model type: {model}")

        self.reg_head = nn.Linear(model_dim, 1)
        self.order_head = nn.Linear(model_dim, 2)
        self.recon_head = nn.Linear(model_dim, input_dim)

    def _time_embed(self, time_vals: torch.Tensor, seq_len: int) -> torch.Tensor:
        if self.time_embedding == "absolute":
            idx = torch.arange(seq_len, device=time_vals.device)
            return self.pos_embed(idx)[None, :, :]
        if self.time_embedding == "time2vec":
            return self.time2vec(time_vals)
        # relative: sinusoidal from time_vals
        dim = self.reg_head.in_features
        freqs = torch.arange(0, dim, 2, device=time_vals.device).float() / dim
        ang = time_vals.unsqueeze(-1) * (10000 ** (-freqs))
        emb = torch.zeros(time_vals.shape[0], time_vals.shape[1], dim, device=time_vals.device)
        emb[:, :, 0::2] = torch.sin(ang)
        emb[:, :, 1::2] = torch.cos(ang)
        return emb

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_vals: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
        phase: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,T,D]
        h = self.input_proj(x)
        seq_len = h.shape[1]
        h = h + self._time_embed(time_vals, seq_len)
        if self.use_delta_t and delta_t is not None:
            h = h + self.delta_proj(delta_t.unsqueeze(-1))
        if self.use_phase and phase is not None:
            h = h + self.phase_proj(phase.unsqueeze(-1))

        if self.model_type == "transformer":
            h = self.encoder(h, src_key_padding_mask=~mask)
        elif self.model_type == "gru":
            lengths = mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
            out_packed, _ = self.encoder(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=seq_len)
        else:  # tcn
            h = self.encoder(h.transpose(1, 2)).transpose(1, 2)

        mask_f = mask.unsqueeze(-1).float()
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return h, pooled


def _regression_loss(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "mae":
        return F.l1_loss(pred, target)
    if kind == "huber":
        return F.smooth_l1_loss(pred, target)
    return F.mse_loss(pred, target)


def _info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)

def _train_one_epoch(
    model: TemporalModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_kind: str,
    aux_order: bool,
    aux_masked: bool,
    aux_contrastive: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        if batch is None:
            continue
        x, mask, time_vals, delta_t, phase, y = batch
        x = x.to(device)
        mask = mask.to(device)
        time_vals = time_vals.to(device)
        delta_t = delta_t.to(device)
        y = y.to(device)
        phase = phase.to(device) if phase is not None else None

        x_in = x
        mask_for_recon = None
        if aux_masked:
            mask_for_recon = torch.zeros_like(mask)
            for i in range(mask.shape[0]):
                valid = mask[i].nonzero(as_tuple=False).squeeze(1)
                if valid.numel() < 2:
                    continue
                m = max(1, int(0.15 * valid.numel()))
                sel = valid[torch.randperm(valid.numel(), device=device)[:m]]
                mask_for_recon[i, sel] = True
            x_in = x.clone()
            x_in[mask_for_recon] = 0.0

        seq_out, pooled = model(x_in, mask, time_vals, delta_t, phase)
        pred = model.reg_head(pooled).squeeze(-1)
        loss = _regression_loss(pred, y, loss_kind)

        if aux_order:
            with torch.no_grad():
                rev = torch.flip(x_in, dims=[1])
            order_labels = torch.zeros(x_in.size(0), dtype=torch.long, device=device)
            flip_mask = torch.rand(x_in.size(0), device=device) < 0.5
            x_order = x_in.clone()
            x_order[flip_mask] = rev[flip_mask]
            order_labels[~flip_mask] = 1
            seq_o, pooled_o = model(x_order, mask, time_vals, delta_t, phase)
            logits = model.order_head(pooled_o)
            loss = loss + 0.1 * F.cross_entropy(logits, order_labels)

        if aux_masked and mask_for_recon is not None:
            recon = model.recon_head(seq_out)
            if mask_for_recon.any():
                loss = loss + 0.1 * F.mse_loss(recon[mask_for_recon], x[mask_for_recon])

        if aux_contrastive:
            noise = torch.randn_like(x) * 0.01
            seq1, pool1 = model(x + noise, mask, time_vals, delta_t, phase)
            seq2, pool2 = model(x - noise, mask, time_vals, delta_t, phase)
            loss = loss + 0.1 * _info_nce(pool1, pool2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        total_count += x.size(0)

    return total_loss / max(total_count, 1)


def _evaluate(
    model: TemporalModel,
    loader: DataLoader,
    device: torch.device,
    loss_kind: str,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            if batch is None:
                continue
            x, mask, time_vals, delta_t, phase, y = batch
            x = x.to(device)
            mask = mask.to(device)
            time_vals = time_vals.to(device)
            delta_t = delta_t.to(device)
            y = y.to(device)
            phase = phase.to(device) if phase is not None else None

            seq_out, pooled = model(x, mask, time_vals, delta_t, phase)
            pred = model.reg_head(pooled).squeeze(-1)
            loss = _regression_loss(pred, y, loss_kind)
            total_loss += float(loss.item()) * x.size(0)
            total_count += x.size(0)

    return total_loss / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train temporal GLS model from frame embeddings.")
    parser.add_argument("--input-embeddings", type=Path, required=True, help="Input parquet/csv of frame embeddings.")
    parser.add_argument("--report-xlsx", type=Path, default=None, help="Optional report XLSX for GLS labels.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Optional log dir (unused).")
    parser.add_argument("--views", type=str, default="", help="Optional view filter.")
    parser.add_argument("--model", type=str, choices=["transformer", "tcn", "gru"], default="transformer")
    parser.add_argument("--seq-len", type=int, default=32, help="Frames per cine sequence.")
    parser.add_argument("--use-delta-t", action="store_true", default=True)
    parser.add_argument("--no-use-delta-t", dest="use_delta_t", action="store_false")
    parser.add_argument("--use-phase", action="store_true", default=False)
    parser.add_argument("--time-embedding", type=str, choices=["relative", "absolute", "time2vec"], default="relative")
    parser.add_argument("--loss", type=str, choices=["mse", "mae", "huber"], default="huber")
    parser.add_argument("--aux-order-verification", action="store_true", default=False)
    parser.add_argument("--aux-masked-timestep", action="store_true", default=False)
    parser.add_argument("--aux-contrastive", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    _set_seed(args.seed)
    df = _load_embeddings(args.input_embeddings)
    if args.report_xlsx and args.report_xlsx.exists():
        df = add_gls_from_report(df, args.report_xlsx)

    if "view" in df.columns and args.views:
        view_set = set(v.strip() for v in args.views.replace(";", ",").split(",") if v.strip())
        df = df[df["view"].isin(view_set)]

    if "gls" not in df.columns:
        raise ValueError("GLS labels not found. Provide --report-xlsx or include gls column.")

    dataset = CineDataset(df, seq_len=args.seq_len, use_phase=args.use_phase)
    if len(dataset) == 0:
        raise RuntimeError("No valid cine samples found.")

    patient_col = resolve_patient_column(df)
    if patient_col is None:
        logger.warning("No patient column found; using random split.")
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_idx, val_idx = indices[:split], indices[split:]
    else:
        groups = [item.patient_id for item in dataset.items]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        train_idx, val_idx = next(splitter.split(np.arange(len(dataset)), groups=groups))

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx) if len(val_idx) else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_ds else None

    sample_item = dataset[0][0]
    input_dim = sample_item.shape[1]
    model = TemporalModel(
        input_dim=input_dim,
        model_dim=args.model_dim,
        model=args.model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_delta_t=args.use_delta_t,
        use_phase=args.use_phase,
        time_embedding=args.time_embedding,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")
    run_name = args.run_name or f"temporal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / f"{run_name}_best.pt"
    history: List[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.loss,
            args.aux_order_verification,
            args.aux_masked_timestep,
            args.aux_contrastive,
        )
        val_loss = _evaluate(model, val_loader, device, args.loss) if val_loader else train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("Epoch %d/%d | train=%.4f val=%.4f", epoch, args.epochs, train_loss, val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "input_dim": input_dim,
                        "model_dim": args.model_dim,
                        "model": args.model,
                        "num_layers": args.num_layers,
                        "num_heads": args.num_heads,
                        "dropout": args.dropout,
                        "use_delta_t": args.use_delta_t,
                        "use_phase": args.use_phase,
                        "time_embedding": args.time_embedding,
                        "seq_len": args.seq_len,
                    },
                },
                best_path,
            )

    (args.output_dir / f"{run_name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / f"{run_name}_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    logger.info("Best checkpoint: %s", best_path)


if __name__ == "__main__":
    main()
