"""
Train longitudinal model on visit embeddings to predict GLS trajectories or deterioration.
"""
from __future__ import annotations

import argparse
import json
import logging
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

from ichilov_pipeline2_utils import resolve_patient_column, resolve_time_column

try:
    from sklearn.model_selection import GroupShuffleSplit
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "scikit-learn is required. Install with: .venv\\Scripts\\python -m pip install scikit-learn"
    ) from exc

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_train_longitudinal")


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
            return np.asarray(eval(val, {"__builtins__": {}}), dtype=np.float32)
        except Exception:
            return None
    return None


@dataclass
class PatientItem:
    embeddings: np.ndarray  # [V,D]
    times: np.ndarray  # [V]
    gls: np.ndarray  # [V]
    patient_id: str


class PatientDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int) -> None:
        self.items: List[PatientItem] = []
        self.max_len = int(max_len)

        emb_col = "embedding"
        if emb_col not in df.columns:
            raise ValueError("Input dataframe missing 'embedding' column.")
        if "gls" not in df.columns:
            raise ValueError("Input dataframe missing 'gls' column.")

        patient_col = resolve_patient_column(df)
        if patient_col is None:
            raise ValueError("No patient column found.")
        time_col = resolve_time_column(df)
        if time_col is None:
            raise ValueError("No visit/time column found.")

        df = df.copy()
        if not pd.api.types.is_numeric_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

        for pid, grp in df.groupby(patient_col):
            grp = grp.sort_values(by=time_col)
            emb_list = []
            gls_list = []
            times = []
            for _, row in grp.iterrows():
                emb = _parse_embedding(row[emb_col])
                if emb is None:
                    continue
                gls_val = row.get("gls")
                if pd.isna(gls_val):
                    continue
                emb_list.append(emb)
                gls_list.append(float(gls_val))
                times.append(row[time_col])
            if len(emb_list) < 2:
                continue
            emb_arr = np.stack(emb_list, axis=0)
            gls_arr = np.asarray(gls_list, dtype=np.float32)
            times_arr = np.asarray(times)
            if np.issubdtype(times_arr.dtype, np.datetime64):
                base = times_arr[0]
                deltas = (times_arr - base) / np.timedelta64(1, "M")
                times_arr = deltas.astype(np.float32)
            else:
                times_arr = times_arr.astype(np.float32)
            self.items.append(PatientItem(embeddings=emb_arr, times=times_arr, gls=gls_arr, patient_id=str(pid)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        emb = item.embeddings
        times = item.times
        gls = item.gls
        n = emb.shape[0]
        if n > self.max_len:
            emb = emb[: self.max_len]
            times = times[: self.max_len]
            gls = gls[: self.max_len]
            n = self.max_len
        pad = self.max_len - n
        if pad > 0:
            emb = np.concatenate([emb, np.zeros((pad, emb.shape[1]), dtype=emb.dtype)], axis=0)
            times = np.concatenate([times, np.zeros(pad, dtype=times.dtype)], axis=0)
            gls = np.concatenate([gls, np.zeros(pad, dtype=gls.dtype)], axis=0)
        mask = np.zeros(self.max_len, dtype=np.bool_)
        mask[:n] = True
        delta_t = np.zeros_like(times)
        delta_t[1:n] = np.diff(times[:n])
        return (
            torch.tensor(emb, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(times, dtype=torch.float32),
            torch.tensor(delta_t, dtype=torch.float32),
            torch.tensor(gls, dtype=torch.float32),
        )


class Time2Vec(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(dim - 1))
        self.b = nn.Parameter(torch.zeros(dim - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        v0 = self.w0 * t + self.b0
        v1 = torch.sin(t.unsqueeze(-1) * self.w + self.b)
        return torch.cat([v0.unsqueeze(-1), v1], dim=-1)


class LongitudinalModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        model: str = "gru",
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_delta_t: bool = True,
        time_embedding: str = "relative",
        task: str = "delta_gls",
    ) -> None:
        super().__init__()
        self.model_type = model
        self.use_delta_t = use_delta_t
        self.time_embedding = time_embedding
        self.task = task

        self.input_proj = nn.Linear(input_dim, model_dim) if input_dim != model_dim else nn.Identity()
        self.delta_proj = nn.Linear(1, model_dim) if use_delta_t else None

        if time_embedding == "absolute":
            self.pos_embed = nn.Embedding(512, model_dim)
            self.time2vec = None
        elif time_embedding == "time2vec":
            self.pos_embed = None
            self.time2vec = Time2Vec(model_dim)
        else:
            self.pos_embed = None
            self.time2vec = None

        if model == "gru":
            self.encoder = nn.GRU(model_dim, model_dim, num_layers=num_layers, batch_first=True)
        elif model == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown model type: {model}")

        out_dim = 1
        self.head = nn.Linear(model_dim, out_dim)

    def _time_embed(self, time_vals: torch.Tensor, seq_len: int) -> torch.Tensor:
        if self.time_embedding == "absolute":
            idx = torch.arange(seq_len, device=time_vals.device)
            return self.pos_embed(idx)[None, :, :]
        if self.time_embedding == "time2vec":
            return self.time2vec(time_vals)
        dim = self.head.in_features
        freqs = torch.arange(0, dim, 2, device=time_vals.device).float() / dim
        ang = time_vals.unsqueeze(-1) * (10000 ** (-freqs))
        emb = torch.zeros(time_vals.shape[0], time_vals.shape[1], dim, device=time_vals.device)
        emb[:, :, 0::2] = torch.sin(ang)
        emb[:, :, 1::2] = torch.cos(ang)
        return emb

    def forward(self, x: torch.Tensor, mask: torch.Tensor, time_vals: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        seq_len = h.shape[1]
        h = h + self._time_embed(time_vals, seq_len)
        if self.use_delta_t:
            h = h + self.delta_proj(delta_t.unsqueeze(-1))

        if self.model_type == "gru":
            lengths = mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
            out_packed, _ = self.encoder(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=seq_len)
        else:
            h = self.encoder(h, src_key_padding_mask=~mask)
        return self.head(h).squeeze(-1)


def _build_targets(
    gls: torch.Tensor,
    mask: torch.Tensor,
    times: torch.Tensor,
    task: str,
    horizon_months: float,
    delta_thresh: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # gls: [B,T]
    B, T = gls.shape
    target = torch.zeros((B, T), device=gls.device)
    target_mask = torch.zeros((B, T), dtype=torch.bool, device=gls.device)
    for i in range(T - 1):
        valid = mask[:, i] & mask[:, i + 1]
        if horizon_months > 0:
            time_delta = times[:, i + 1] - times[:, i]
            valid = valid & (time_delta <= horizon_months)
        if task == "next_gls":
            target[valid, i] = gls[valid, i + 1]
            target_mask[valid, i] = True
        elif task == "delta_gls":
            target[valid, i] = gls[valid, i + 1] - gls[valid, i]
            target_mask[valid, i] = True
        else:  # deterioration
            delta = gls[:, i + 1] - gls[:, i]
            target[valid, i] = (delta[valid] >= delta_thresh).float()
            target_mask[valid, i] = True
    return target, target_mask


def _regression_loss(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "mae":
        return F.l1_loss(pred, target)
    if kind == "huber":
        return F.smooth_l1_loss(pred, target)
    return F.mse_loss(pred, target)


def _train_one_epoch(
    model: LongitudinalModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    loss_kind: str,
    horizon_months: float,
    delta_thresh: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        if batch is None:
            continue
        x, mask, times, delta_t, gls = batch
        x = x.to(device)
        mask = mask.to(device)
        times = times.to(device)
        delta_t = delta_t.to(device)
        gls = gls.to(device)

        pred = model(x, mask, times, delta_t)
        target, target_mask = _build_targets(gls, mask, times, task, horizon_months, delta_thresh)
        if task == "deterioration":
            loss = F.binary_cross_entropy_with_logits(pred[target_mask], target[target_mask]) if target_mask.any() else torch.tensor(0.0, device=device)
        else:
            loss = _regression_loss(pred[target_mask], target[target_mask], loss_kind) if target_mask.any() else torch.tensor(0.0, device=device)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        total_count += x.size(0)

    return total_loss / max(total_count, 1)


def _evaluate(
    model: LongitudinalModel,
    loader: DataLoader,
    device: torch.device,
    task: str,
    loss_kind: str,
    horizon_months: float,
    delta_thresh: float,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            if batch is None:
                continue
            x, mask, times, delta_t, gls = batch
            x = x.to(device)
            mask = mask.to(device)
            times = times.to(device)
            delta_t = delta_t.to(device)
            gls = gls.to(device)

            pred = model(x, mask, times, delta_t)
            target, target_mask = _build_targets(gls, mask, times, task, horizon_months, delta_thresh)
            if task == "deterioration":
                loss = F.binary_cross_entropy_with_logits(pred[target_mask], target[target_mask]) if target_mask.any() else torch.tensor(0.0, device=device)
            else:
                loss = _regression_loss(pred[target_mask], target[target_mask], loss_kind) if target_mask.any() else torch.tensor(0.0, device=device)

            total_loss += float(loss.item()) * x.size(0)
            total_count += x.size(0)

    return total_loss / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train longitudinal model on visit embeddings.")
    parser.add_argument("--input-embeddings", type=Path, required=True, help="Visit embeddings parquet/csv.")
    parser.add_argument("--report-xlsx", type=Path, default=None, help="Unused (expects gls in embeddings).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--views", type=str, default="")
    parser.add_argument("--model", type=str, choices=["gru", "transformer"], default="gru")
    parser.add_argument("--use-delta-t", action="store_true", default=True)
    parser.add_argument("--no-use-delta-t", dest="use_delta_t", action="store_false")
    parser.add_argument("--time-embedding", type=str, choices=["relative", "absolute", "time2vec"], default="relative")
    parser.add_argument("--task", type=str, choices=["next_gls", "delta_gls", "deterioration"], default="delta_gls")
    parser.add_argument("--horizon-months", type=float, default=6.0)
    parser.add_argument("--loss", type=str, choices=["mse", "mae", "huber"], default="huber")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=16)
    parser.add_argument("--deterioration-delta", type=float, default=2.0)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    _set_seed(args.seed)

    df = pd.read_parquet(args.input_embeddings) if args.input_embeddings.suffix.lower() == ".parquet" else pd.read_csv(args.input_embeddings)
    if "view" in df.columns and args.views:
        view_set = set(v.strip() for v in args.views.replace(";", ",").split(",") if v.strip())
        df = df[df["view"].isin(view_set)]

    dataset = PatientDataset(df, max_len=args.max_len)
    if len(dataset) == 0:
        raise RuntimeError("No valid patient sequences found.")

    groups = [item.patient_id for item in dataset.items]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(np.arange(len(dataset)), groups=groups))

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx) if len(val_idx) else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_ds else None

    input_dim = dataset.items[0].embeddings.shape[1]
    model = LongitudinalModel(
        input_dim=input_dim,
        model_dim=args.model_dim,
        model=args.model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_delta_t=args.use_delta_t,
        time_embedding=args.time_embedding,
        task=args.task,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")
    run_name = args.run_name or f"longitudinal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / f"{run_name}_best.pt"
    history: List[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.task,
            args.loss,
            args.horizon_months,
            args.deterioration_delta,
        )
        val_loss = _evaluate(
            model,
            val_loader,
            device,
            args.task,
            args.loss,
            args.horizon_months,
            args.deterioration_delta,
        ) if val_loader else train_loss
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
                        "time_embedding": args.time_embedding,
                        "task": args.task,
                        "max_len": args.max_len,
                    },
                },
                best_path,
            )

    (args.output_dir / f"{run_name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / f"{run_name}_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    logger.info("Best checkpoint: %s", best_path)


if __name__ == "__main__":
    main()
