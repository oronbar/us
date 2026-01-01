"""
Train models (CNN/Transformer/MLP + baselines) for GLS prediction from embeddings.

Input:
  - Parquet/CSV embeddings dataframe (from ichilov_encode_dicoms.py)
  - Optional Excel report to supply GLS targets

Output:
  - Model weights + metrics in output directory

Usage (PowerShell):
  .venv\\Scripts\\python ichilov_train_gls.py ^
    --input-embeddings "$env:USERPROFILE\\OneDrive - Technion\\DS\\Ichilov_GLS_embeddings.parquet" ^
    --report-xlsx "$env:USERPROFILE\\OneDrive - Technion\\DS\\Report_Ichilov_GLS_oron.xlsx" ^
    --output-dir "$env:USERPROFILE\\OneDrive - Technion\\DS\\Ichilov_GLS_models"

Notes:
  - Use --target-mode delta for longitudinal delta GLS targets (requires a visit time column).
  - Use --objective ranking for pairwise within-patient ranking loss.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for saving plots in headless runs.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_train_gls")

VIEW_KEYS = ("A2C", "A3C", "A4C")
TIME_COL_CANDIDATES = (
    "study_datetime",
    "study_date",
    "visit_date",
    "exam_date",
    "acquisition_date",
    "acquisition_time",
    "scan_date",
    "visit_time",
    "date",
    "datetime",
    "visit_index",
    "visit_num",
    "visit_number",
    "timepoint",
    "time_point",
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _find_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = False) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    for cand in candidates:
        key = cand.lower()
        for col in df.columns:
            if key in str(col).lower():
                return col
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def _parse_embedding(val: object) -> Optional[np.ndarray]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, np.ndarray):
        return val.astype(np.float32)
    if isinstance(val, list):
        return np.asarray(val, dtype=np.float32)
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return np.asarray(parsed, dtype=np.float32)
        except Exception:
            return None
    return None


def _normalize_path(s: str) -> str:
    return str(Path(s)).lower().replace("\\", "/")


def _load_embeddings(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)


def _add_gls_from_report(df: pd.DataFrame, report_xlsx: Path) -> pd.DataFrame:
    rep = pd.read_excel(report_xlsx, engine="openpyxl")
    rep.columns = [str(c).strip() for c in rep.columns]

    mapping_full: Dict[str, float] = {}
    mapping_base: Dict[str, float] = {}

    for view in VIEW_KEYS:
        gls_col = _find_column(rep, [f"{view}_GLS"], required=False)
        src_col = _find_column(rep, [f"{view}_GLS_SOURCE_DICOM"], required=False)
        if not gls_col or not src_col:
            continue
        for _, row in rep.iterrows():
            src = row.get(src_col)
            gls = row.get(gls_col)
            if pd.isna(src) or pd.isna(gls):
                continue
            try:
                gls_val = float(gls)
            except Exception:
                continue
            src_str = str(src)
            mapping_full[_normalize_path(src_str)] = gls_val
            mapping_base[Path(src_str).name.lower()] = gls_val

    if "source_dicom" not in df.columns:
        raise ValueError("Embeddings dataframe missing 'source_dicom' column needed for GLS join.")

    gls_vals: List[Optional[float]] = []
    for _, row in df.iterrows():
        src = row.get("source_dicom")
        if pd.isna(src):
            gls_vals.append(None)
            continue
        src_str = str(src)
        key_full = _normalize_path(src_str)
        if key_full in mapping_full:
            gls_vals.append(mapping_full[key_full])
            continue
        base = Path(src_str).name.lower()
        gls_vals.append(mapping_base.get(base))

    df = df.copy()
    df["gls"] = gls_vals
    return df


def _resolve_time_column(df: pd.DataFrame, requested: str) -> Optional[str]:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Time column '{requested}' not found in embeddings df.")
        return requested
    return _find_column(df, TIME_COL_CANDIDATES, required=False)


def _coerce_time_values(series: pd.Series) -> Optional[pd.Series]:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().any():
        return parsed
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric
    return None


def _resolve_group_cols(df: pd.DataFrame, patient_col: str, group_by_view: bool) -> List[str]:
    group_cols = [patient_col]
    if group_by_view and "view" in df.columns:
        group_cols.append("view")
    return group_cols


def _apply_target_transform(
    df: pd.DataFrame,
    target_mode: str,
    group_cols: Sequence[str],
    time_col: Optional[str],
) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    if target_mode == "absolute":
        df["target"] = df["gls"]
        return df, "GLS"
    if target_mode == "patient_centered":
        df["target"] = df["gls"] - df.groupby(group_cols)["gls"].transform("mean")
        return df, "Centered GLS"
    if target_mode == "delta":
        if time_col is None:
            raise ValueError("Delta target requires a visit time column. Provide --time-col.")
        if "_visit_time" not in df.columns:
            time_vals = _coerce_time_values(df[time_col])
            if time_vals is None or time_vals.isna().all():
                raise ValueError(f"Could not parse visit time column '{time_col}'.")
            df["_visit_time"] = time_vals
        if df["_visit_time"].isna().any():
            before = len(df)
            df = df[df["_visit_time"].notna()].copy()
            dropped = before - len(df)
            if dropped:
                logger.warning("Dropped %d rows with missing visit time for delta target.", dropped)
        min_time = df.groupby(group_cols)["_visit_time"].transform("min")
        baseline = (
            df[df["_visit_time"] == min_time]
            .groupby(group_cols)["gls"]
            .mean()
            .rename("_gls_baseline")
        )
        df = df.merge(baseline, on=group_cols, how="left")
        df["target"] = df["gls"] - df["_gls_baseline"]
        missing = df["_gls_baseline"].isna().sum()
        if missing:
            logger.warning("Dropped %d rows without baseline GLS for delta target.", missing)
            df = df[df["_gls_baseline"].notna()].copy()
        return df, "Delta GLS"
    raise ValueError(f"Unknown target mode: {target_mode}")


class EmbeddingDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float().view(-1, 1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class PairwiseRankingDataset(Dataset):
    def __init__(self, x: np.ndarray, pairs: np.ndarray, labels: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.pairs = pairs.astype(np.int64)
        self.labels = torch.from_numpy(labels).float().view(-1, 1)

    def __len__(self) -> int:
        return self.pairs.shape[0]

    def __getitem__(self, idx: int):
        i, j = self.pairs[idx]
        return self.x[i], self.x[j], self.labels[idx]


class CNNRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.net(x)
        return self.head(x)


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = nn.Parameter(torch.zeros(1, input_dim, d_model))
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)
        x = x.unsqueeze(-1)  # (B, L, 1)
        x = self.input_proj(x) + self.pos
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)

class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualMLPBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, depth: int = 3, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.in_proj = nn.Linear(input_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, dropout) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.drop(self.act(self.in_proj(x)))
        for block in self.blocks:
            x = block(x)
        return x


class MLPRegressor(nn.Module):
    """Strong baseline for fixed embeddings (no fake locality)."""

    def __init__(self, input_dim: int, hidden: int = 256, depth: int = 3, dropout: float = 0.2):
        super().__init__()
        self.backbone = ResidualMLPBackbone(input_dim, hidden=hidden, depth=depth, dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)


class MLPHeteroscedastic(nn.Module):
    """
    Predict mean + log-variance for noisy GLS.
    Outputs: (mu, log_var) in normalized target space.
    """
    def __init__(self, input_dim: int, hidden: int = 256, depth: int = 3, dropout: float = 0.2):
        super().__init__()
        self.backbone = ResidualMLPBackbone(input_dim, hidden=hidden, depth=depth, dropout=dropout)
        self.mu = nn.Linear(hidden, 1)
        self.log_var = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.mu(h), self.log_var(h)


def _pairwise_loss(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    margin: float,
) -> torch.Tensor:
    diff = scores_a - scores_b
    if loss_type == "hinge":
        return torch.clamp(margin - target * diff, min=0.0).mean()
    return F.softplus(-target * diff).mean()


def _predict_scores(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: List[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze(1)
            preds.extend(out.cpu().numpy().tolist())
    return np.asarray(preds, dtype=float)


def _fit_linear_calibration(scores: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if len(scores) < 2 or np.std(scores) < 1e-8:
        return 0.0, float(np.nanmean(targets)) if len(targets) else 0.0
    A = np.vstack([scores, np.ones_like(scores)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, targets, rcond=None)
    scale, bias = coeffs
    return float(scale), float(bias)


def _spearman_corr(preds: np.ndarray, trues: np.ndarray) -> float:
    preds = np.asarray(preds, dtype=float)
    trues = np.asarray(trues, dtype=float)
    if len(preds) < 2 or np.std(preds) < 1e-8 or np.std(trues) < 1e-8:
        return float("nan")
    ranks_pred = pd.Series(preds).rank().to_numpy()
    ranks_true = pd.Series(trues).rank().to_numpy()
    return float(np.corrcoef(ranks_pred, ranks_true)[0, 1])


def _build_pairwise_pairs(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    order_col: str,
    label_source: str,
    direction: str,
    max_pairs_per_group: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    pairs: List[Tuple[int, int]] = []
    labels: List[int] = []
    for _, grp in df.groupby(list(group_cols)):
        grp = grp.dropna(subset=[order_col])
        if len(grp) < 2:
            continue
        grp = grp.sort_values(order_col)
        idx = grp.index.to_numpy()
        local_pairs: List[Tuple[int, int]] = []
        local_labels: List[int] = []
        if label_source == "time":
            if direction == "nondecreasing":
                for i in range(len(idx) - 1):
                    for j in range(i + 1, len(idx)):
                        local_pairs.append((idx[j], idx[i]))
                        local_labels.append(1)
            else:
                for i in range(len(idx) - 1):
                    for j in range(i + 1, len(idx)):
                        local_pairs.append((idx[i], idx[j]))
                        local_labels.append(1)
        else:
            y = grp["target"].to_numpy()
            for i in range(len(idx) - 1):
                for j in range(i + 1, len(idx)):
                    yi = y[i]
                    yj = y[j]
                    if not np.isfinite(yi) or not np.isfinite(yj) or yi == yj:
                        continue
                    if yi > yj:
                        local_pairs.append((idx[i], idx[j]))
                        local_labels.append(1)
                    else:
                        local_pairs.append((idx[j], idx[i]))
                        local_labels.append(1)
        if max_pairs_per_group > 0 and len(local_pairs) > max_pairs_per_group:
            choices = rng.choice(len(local_pairs), size=max_pairs_per_group, replace=False)
            local_pairs = [local_pairs[c] for c in choices]
            local_labels = [local_labels[c] for c in choices]
        pairs.extend(local_pairs)
        labels.extend(local_labels)
    if not pairs:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)
    return np.asarray(pairs, dtype=int), np.asarray(labels, dtype=float)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: float,
    y_std: float,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    preds: List[float] = []
    trues: List[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze(1)
            preds.extend(out.cpu().numpy().tolist())
            trues.extend(y.squeeze(1).cpu().numpy().tolist())
    preds_arr = np.asarray(preds, dtype=float) * y_std + y_mean
    trues_arr = np.asarray(trues, dtype=float) * y_std + y_mean
    mae = float(np.mean(np.abs(preds_arr - trues_arr)))
    rmse = float(np.sqrt(np.mean((preds_arr - trues_arr) ** 2)))
    mse = float(np.mean((preds_arr - trues_arr) ** 2))
    return mae, rmse, mse, preds_arr, trues_arr


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    out_path: Path,
    lr_factor: float,
    lr_patience: int,
    lr_min: float,
    y_mean: float,
    y_std: float,
    loss_type: str,
    log_dir: Optional[Path] = None,
    plot_every: int = 10,
    run_args: Optional[Dict[str, object]] = None,
    objective: str = "regression",
    pairwise_train_loader: Optional[DataLoader] = None,
    pairwise_val_loader: Optional[DataLoader] = None,
    ranking_loss: str = "hinge",
    ranking_margin: float = 0.1,
    train_eval_loader: Optional[DataLoader] = None,
    val_eval_loader: Optional[DataLoader] = None,
    y_train_raw: Optional[np.ndarray] = None,
    y_val_raw: Optional[np.ndarray] = None,
    target_label: str = "GLS",
) -> Dict[str, float]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=lr_factor, patience=lr_patience, min_lr=lr_min)
    if objective == "ranking":
        if pairwise_train_loader is None or pairwise_val_loader is None:
            raise ValueError("Pairwise loaders are required for ranking objective.")
        if train_eval_loader is None or val_eval_loader is None or y_train_raw is None or y_val_raw is None:
            raise ValueError("Eval loaders and raw targets are required for ranking objective.")
    if loss_type == "mae":
        loss_fn = nn.L1Loss()
    elif loss_type == "huber":
        loss_fn = nn.SmoothL1Loss(beta=1.0)
    else:
        loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    def _normalize_run_args(values: Dict[str, object]) -> Dict[str, object]:
        normalized: Dict[str, object] = {}
        for key, val in values.items():
            if isinstance(val, Path):
                normalized[key] = str(val)
            elif isinstance(val, (list, tuple)):
                normalized[key] = [str(v) if isinstance(v, Path) else v for v in val]
            else:
                normalized[key] = val
        return normalized

    def _sanitize_token(token: str) -> str:
        cleaned = []
        for ch in str(token):
            cleaned.append(ch if (ch.isalnum() or ch in ("-", "_")) else "-")
        out = "".join(cleaned).strip("-_")
        return out or "arg"

    norm_args = _normalize_run_args(run_args) if run_args is not None else None
    suffix = ""
    if norm_args is not None:
        last_args = None
        model_name = str(norm_args.get("trained_model")) if "trained_model" in norm_args else None
        candidates = []
        for path in out_path.parent.glob("pred_plots_*"):
            if not path.is_dir():
                continue
            args_file = path / "run_args.json"
            if not args_file.exists():
                continue
            try:
                with args_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if model_name is not None and str(data.get("trained_model")) != model_name:
                continue
            candidates.append((args_file.stat().st_mtime, data))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            last_args = candidates[0][1]
        if last_args is not None:
            diff_keys = [k for k in sorted(norm_args.keys()) if last_args.get(k) != norm_args.get(k)]
            if diff_keys:
                suffix = "__" + "_".join(_sanitize_token(k) for k in diff_keys)

    plot_dir = out_path.parent / f"pred_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if norm_args is not None:
        args_path = plot_dir / "run_args.json"
        with args_path.open("w", encoding="utf-8") as f:
            json.dump(norm_args, f, indent=2, sort_keys=True)
        logger.info(f"Saved run args: {args_path}")
    writer = SummaryWriter(str(log_dir)) if log_dir is not None else None
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        if objective == "ranking":
            for x_a, x_b, target in pairwise_train_loader:
                x_a = x_a.to(device)
                x_b = x_b.to(device)
                target = target.to(device).view(-1)
                opt.zero_grad()
                out_a = model(x_a)
                out_b = model(x_b)
                if isinstance(out_a, tuple):
                    out_a = out_a[0]
                if isinstance(out_b, tuple):
                    out_b = out_b[0]
                out_a = out_a.squeeze(1)
                out_b = out_b.squeeze(1)
                loss = _pairwise_loss(out_a, out_b, target, ranking_loss, ranking_margin)
                loss.backward()
                opt.step()
                running += float(loss.item()) * x_a.size(0)
            train_loss = running / len(pairwise_train_loader.dataset)
        else:
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                out = model(x)
                if isinstance(out, tuple) and len(out) == 2:
                    mu, log_var = out
                    # Gaussian NLL in normalized y space:
                    # 0.5 * (log_var + (y-mu)^2 / exp(log_var))
                    loss = 0.5 * (log_var + (y - mu) ** 2 / torch.exp(log_var))
                    loss = loss.mean()
                else:
                    loss = loss_fn(out, y)
                loss.backward()
                opt.step()
                running += float(loss.item()) * x.size(0)
            train_loss = running / len(train_loader.dataset)

        # Validation loss using the training loss function.
        model.eval()
        if objective == "ranking":
            val_running = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for x_a, x_b, target in pairwise_val_loader:
                    x_a = x_a.to(device)
                    x_b = x_b.to(device)
                    target = target.to(device).view(-1)
                    out_a = model(x_a)
                    out_b = model(x_b)
                    if isinstance(out_a, tuple):
                        out_a = out_a[0]
                    if isinstance(out_b, tuple):
                        out_b = out_b[0]
                    out_a = out_a.squeeze(1)
                    out_b = out_b.squeeze(1)
                    loss = _pairwise_loss(out_a, out_b, target, ranking_loss, ranking_margin)
                    val_running += float(loss.item()) * x_a.size(0)
                    diff = out_a - out_b
                    correct += int((diff * target > 0).sum().item())
                    total += int(target.numel())
            val_loss = val_running / len(pairwise_val_loader.dataset)
            val_pairwise_acc = correct / total if total else float("nan")
            val_mse = val_loss
        else:
            val_running = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    out = model(x)
                    if isinstance(out, tuple) and len(out) == 2:
                        mu, log_var = out
                        loss = 0.5 * (log_var + (y - mu) ** 2 / torch.exp(log_var))
                        loss = loss.mean()
                    else:
                        loss = loss_fn(out, y)
                    val_running += float(loss.item()) * x.size(0)
            val_loss = val_running / len(val_loader.dataset)
            val_pairwise_acc = float("nan")
            val_mae, val_rmse, val_mse, preds_arr, trues_arr = _evaluate(
                model, val_loader, device, y_mean, y_std
            )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_mse)
        current_lr = opt.param_groups[0]["lr"]
        if objective == "ranking":
            logger.info(
                f"Epoch {epoch:03d} | lr={current_lr:.2e} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_pairwise_acc={val_pairwise_acc:.4f}"
            )
            if writer:
                writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
                writer.add_scalar("metrics/val_pairwise_acc", val_pairwise_acc, epoch)
                writer.add_scalar("lr", current_lr, epoch)
                writer.flush()
        else:
            logger.info(
                f"Epoch {epoch:03d} | lr={current_lr:.2e} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_mae={val_mae:.4f} | val_rmse={val_rmse:.4f}"
            )
            if writer:
                writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
                writer.add_scalar("metrics/val_mae", val_mae, epoch)
                writer.add_scalar("metrics/val_rmse", val_rmse, epoch)
                writer.add_scalar("metrics/val_mse", val_mse, epoch)
                writer.add_scalar("lr", current_lr, epoch)
                writer.flush()
            if plot_every > 0 and epoch % plot_every == 0:
                plot_dir.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots()
                ax.plot(trues_arr, label="true")
                ax.plot(preds_arr, label="pred")
                ax.set_xlabel("Sample")
                ax.set_ylabel(target_label)
                ax.set_title(f"{out_path.stem} epoch {epoch} predictions")
                ax.legend()
                fig.tight_layout()
                plot_path = plot_dir / f"{out_path.stem}_epoch{epoch:03d}.png"
                fig.savefig(plot_path)
                plt.close(fig)
                logger.info(f"Saved prediction plot: {plot_path}")
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, out_path)

    eval_model = model
    if best_state is not None:
        eval_model.load_state_dict(best_state)
    if objective == "ranking":
        val_running = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_a, x_b, target in pairwise_val_loader:
                x_a = x_a.to(device)
                x_b = x_b.to(device)
                target = target.to(device).view(-1)
                out_a = eval_model(x_a)
                out_b = eval_model(x_b)
                if isinstance(out_a, tuple):
                    out_a = out_a[0]
                if isinstance(out_b, tuple):
                    out_b = out_b[0]
                out_a = out_a.squeeze(1)
                out_b = out_b.squeeze(1)
                loss = _pairwise_loss(out_a, out_b, target, ranking_loss, ranking_margin)
                val_running += float(loss.item()) * x_a.size(0)
                diff = out_a - out_b
                correct += int((diff * target > 0).sum().item())
                total += int(target.numel())
        val_pairwise_loss = val_running / len(pairwise_val_loader.dataset)
        val_pairwise_acc = correct / total if total else float("nan")
        train_scores = _predict_scores(eval_model, train_eval_loader, device)
        val_scores = _predict_scores(eval_model, val_eval_loader, device)
        calib_scale, calib_bias = _fit_linear_calibration(train_scores, y_train_raw)
        preds_arr = val_scores * calib_scale + calib_bias
        trues_arr = np.asarray(y_val_raw, dtype=float)
        val_mae = float(np.mean(np.abs(preds_arr - trues_arr))) if len(trues_arr) else float("nan")
        val_rmse = float(np.sqrt(np.mean((preds_arr - trues_arr) ** 2))) if len(trues_arr) else float("nan")
        val_mse = float(np.mean((preds_arr - trues_arr) ** 2)) if len(trues_arr) else float("nan")
        if len(preds_arr) > 1 and np.std(preds_arr) > 1e-8 and np.std(trues_arr) > 1e-8:
            val_corr = float(np.corrcoef(preds_arr, trues_arr)[0, 1])
        else:
            val_corr = float("nan")
        val_spearman = _spearman_corr(val_scores, trues_arr)
    else:
        val_pairwise_loss = float("nan")
        val_pairwise_acc = float("nan")
        calib_scale = float("nan")
        calib_bias = float("nan")
        val_mae, val_rmse, val_mse, preds_arr, trues_arr = _evaluate(
            eval_model, val_loader, device, y_mean, y_std
        )
        if len(preds_arr) > 1 and np.std(preds_arr) > 1e-8 and np.std(trues_arr) > 1e-8:
            val_corr = float(np.corrcoef(preds_arr, trues_arr)[0, 1])
        else:
            val_corr = float("nan")
        val_spearman = _spearman_corr(preds_arr, trues_arr)

    mean_vals = (preds_arr + trues_arr) / 2.0
    diff_vals = preds_arr - trues_arr
    diff_mean = float(np.mean(diff_vals)) if len(diff_vals) else float("nan")
    diff_std = float(np.std(diff_vals)) if len(diff_vals) else float("nan")
    loa_upper = diff_mean + 1.96 * diff_std if np.isfinite(diff_mean) and np.isfinite(diff_std) else float("nan")
    loa_lower = diff_mean - 1.96 * diff_std if np.isfinite(diff_mean) and np.isfinite(diff_std) else float("nan")

    if len(diff_vals):
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        ax.scatter(mean_vals, diff_vals, alpha=0.6, s=12)
        ax.axhline(diff_mean, color="red", linestyle="--", linewidth=1.5, label=f"mean={diff_mean:.3f}")
        if np.isfinite(loa_upper) and np.isfinite(loa_lower):
            ax.axhline(loa_upper, color="gray", linestyle="--", linewidth=1.0, label=f"+1.96 SD={loa_upper:.3f}")
            ax.axhline(loa_lower, color="gray", linestyle="--", linewidth=1.0, label=f"-1.96 SD={loa_lower:.3f}")
        ax.set_xlabel("Mean of true and pred")
        ax.set_ylabel("Pred - True")
        ax.set_title(f"{out_path.stem} Bland-Altman")
        ax.legend()
        fig.tight_layout()
        ba_path = plot_dir / f"{out_path.stem}_bland_altman.png"
        fig.savefig(ba_path)
        plt.close(fig)
        logger.info(f"Saved Bland-Altman plot: {ba_path}")

        metrics_path = plot_dir / f"{out_path.stem}_val_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                    "val_mse": val_mse,
                    "val_corr": val_corr,
                    "val_spearman": val_spearman,
                    "val_pairwise_loss": val_pairwise_loss,
                    "val_pairwise_acc": val_pairwise_acc,
                    "calibration_scale": calib_scale,
                    "calibration_bias": calib_bias,
                    "objective": objective,
                    "bland_altman_mean": diff_mean,
                    "bland_altman_loa_lower": loa_lower,
                    "bland_altman_loa_upper": loa_upper,
                },
                f,
                indent=2,
            )
        logger.info(f"Saved validation metrics: {metrics_path}")

    if train_losses and val_losses:
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        epochs = np.arange(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label="train")
        ax.plot(epochs, val_losses, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{out_path.stem} train/val loss")
        ax.legend()
        fig.tight_layout()
        loss_plot_path = plot_dir / f"{out_path.stem}_loss_curve.png"
        fig.savefig(loss_plot_path)
        plt.close(fig)
        logger.info(f"Saved loss curve plot: {loss_plot_path}")

    if writer:
        writer.close()

    return {
        "val_mse": val_mse,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_corr": val_corr,
        "val_spearman": val_spearman,
        "val_pairwise_loss": val_pairwise_loss,
        "val_pairwise_acc": val_pairwise_acc,
        "calibration_scale": calib_scale,
        "calibration_bias": calib_bias,
        "bland_altman_mean": diff_mean,
        "bland_altman_loa_lower": loa_lower,
        "bland_altman_loa_upper": loa_upper,
    }


def _regression_metrics(preds: np.ndarray, trues: np.ndarray) -> Dict[str, float]:
    preds = np.asarray(preds, dtype=float)
    trues = np.asarray(trues, dtype=float)
    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    if len(preds) > 1 and np.std(preds) > 1e-8 and np.std(trues) > 1e-8:
        corr = float(np.corrcoef(preds, trues)[0, 1])
    else:
        corr = float("nan")
    return {"mae": mae, "rmse": rmse, "corr": corr}


def _run_linear_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge

    models = {
        "ridge": Ridge(alpha=10.0),
        "elasticnet": ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000),
        "huber": HuberRegressor(),
    }
    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        results[name] = _regression_metrics(preds, y_val)
    return results


def _run_tree_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    try:
        import xgboost as xgb

        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
        )
        xgb_model.fit(x_train, y_train)
        preds = xgb_model.predict(x_val)
        results["xgboost"] = _regression_metrics(preds, y_val)
    except Exception as exc:
        logger.warning("XGBoost baseline skipped: %s", exc)

    try:
        import lightgbm as lgb

        lgb_model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=64,
            random_state=seed,
        )
        lgb_model.fit(x_train, y_train)
        preds = lgb_model.predict(x_val)
        results["lightgbm"] = _regression_metrics(preds, y_val)
    except Exception as exc:
        logger.warning("LightGBM baseline skipped: %s", exc)

    try:
        from catboost import CatBoostRegressor

        cat_model = CatBoostRegressor(
            depth=6,
            learning_rate=0.05,
            iterations=300,
            random_seed=seed,
            verbose=False,
        )
        cat_model.fit(x_train, y_train)
        preds = cat_model.predict(x_val)
        results["catboost"] = _regression_metrics(preds, y_val)
    except Exception as exc:
        logger.warning("CatBoost baseline skipped: %s", exc)

    return results


def main() -> None:
    user_home = Path.home()
    if user_home.name == "oronbar.RF":
        user_home = Path("F:\\")
    parser = argparse.ArgumentParser(description="Train GLS prediction from embeddings.")
    parser.add_argument(
        "--input-embeddings",
        type=Path,
        default=user_home / "OneDrive - Technion" / "DS" / "Ichilov_GLS_embeddings_full.parquet",
        help="Embedding dataframe (parquet/csv) from ichilov_encode_dicoms.py",
    )
    parser.add_argument(
        "--report-xlsx",
        type=Path,
        required=False,
        default=user_home / "OneDrive - Technion" / "DS" / "Report_Ichilov_GLS_oron.xlsx",
        help="Optional report Excel to supply GLS targets if missing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=user_home / "OneDrive - Technion" / "models" / "Ichilov_GLS_models",
        help="Output directory for model weights and metrics",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Optional TensorBoard log directory (default: OUTPUT_DIR/tensorboard/<timestamp>)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "transformer", "mlp", "mlp_unc", "both", "all"],
        default="mlp",
        help="Model type to train (both=cnn+transformer, all=cnn+transformer+mlp+mlp_unc)",
    )
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-factor", type=float, default=0.5, help="LR reduce factor on plateau")
    parser.add_argument("--lr-patience", type=int, default=5, help="Epoch patience before LR drop")
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Lower bound for learning rate")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--loss",
        type=str,
        choices=["mse", "mae", "huber"],
        default="huber",
        help="Regression loss to use (regression objective only).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Optional embedding dimension to select (e.g., 768 for MAE max-pooled, 1536 for STF fusion).",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="",
        help="Optional target column name (defaults to 'gls' if present)",
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        choices=["absolute", "delta", "patient_centered"],
        default="absolute",
        help="Target to learn: absolute GLS, delta vs earliest visit, or patient-centered GLS.",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="",
        help="Optional visit time column for delta/ranking (auto-detect if empty).",
    )
    parser.add_argument(
        "--group-by-view",
        dest="group_by_view",
        action="store_true",
        default=True,
        help="Group baselines/centering and ranking pairs within view when available.",
    )
    parser.add_argument(
        "--no-group-by-view",
        dest="group_by_view",
        action="store_false",
        help="Do not group baselines/centering or ranking pairs by view.",
    )
    parser.add_argument(
        "--target-min",
        type=float,
        default=None,
        help="Optional minimum filter on transformed target.",
    )
    parser.add_argument(
        "--target-max",
        type=float,
        default=None,
        help="Optional maximum filter on transformed target.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=["regression", "ranking"],
        default="regression",
        help="Training objective: regression or within-patient ranking.",
    )
    parser.add_argument(
        "--pairwise-label",
        type=str,
        choices=["time", "target"],
        default="time",
        help="For ranking: label pairs by visit time monotonicity or target ordering.",
    )
    parser.add_argument(
        "--ranking-direction",
        type=str,
        choices=["nondecreasing", "nonincreasing"],
        default="nondecreasing",
        help="For pairwise-label=time: later >= earlier (nondecreasing) or later <= earlier (nonincreasing).",
    )
    parser.add_argument(
        "--ranking-loss",
        type=str,
        choices=["hinge", "logistic"],
        default="hinge",
        help="Pairwise ranking loss.",
    )
    parser.add_argument(
        "--ranking-margin",
        type=float,
        default=0.1,
        help="Margin for hinge ranking loss.",
    )
    parser.add_argument(
        "--max-pairs-per-group",
        type=int,
        default=0,
        help="Limit number of pairs per patient/group for ranking (0 = all).",
    )
    parser.add_argument("--mlp-hidden", type=int, default=256, help="Hidden size for MLP models.")
    parser.add_argument("--mlp-depth", type=int, default=3, help="Number of residual blocks in the MLP.")
    parser.add_argument("--mlp-dropout", type=float, default=0.2, help="Dropout rate for MLP models.")
    parser.add_argument(
        "--run-baselines",
        action="store_true",
        default=True,
        help="Run linear baselines (ridge/elasticnet/huber).",
    )
    parser.add_argument(
        "--run-tree-baselines",
        action="store_true",
        default=True,
        help="Run optional tree baselines (xgboost/lightgbm/catboost).",
    )
    parser.add_argument(
        "--gls-min",
        type=float,
        default=-23,
        help="Optional minimum raw GLS filter (inclusive).",
    )
    parser.add_argument(
        "--gls-max",
        type=float,
        default=-15,
        help="Optional maximum raw GLS filter (inclusive).",
    )
    parser.add_argument(
        "--standardize",
        dest="standardize",
        action="store_true",
        default=True,
        help="Disable standardization of embeddings/targets (use raw values).",
    )
    args = parser.parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = args.log_dir or args.output_dir / "tensorboard" / run_id
    _set_seed(args.seed)

    df = _load_embeddings(args.input_embeddings)
    df.columns = [str(c).strip() for c in df.columns]

    if args.target_col:
        if args.target_col not in df.columns:
            raise ValueError(f"Target column '{args.target_col}' not found in embeddings df.")
        df["gls"] = df[args.target_col]
    elif "gls" not in df.columns and "GLS" in df.columns:
        df["gls"] = df["GLS"]

    if "gls" not in df.columns:
        if args.report_xlsx:
            df = _add_gls_from_report(df, args.report_xlsx)
        else:
            raise ValueError("No GLS target found. Provide --report-xlsx or --target-col.")

    if args.embedding_dim is not None:
        if "embedding_dim" in df.columns:
            df = df[df["embedding_dim"] == args.embedding_dim].copy()
            logger.info(f"Filtered embeddings to embedding_dim={args.embedding_dim}, remaining rows: {len(df)}")
        else:
            logger.warning("--embedding-dim was provided but 'embedding_dim' column is missing; proceeding without filtering.")

    emb_col = _find_column(df, ["embedding"], required=True)
    df["embedding_vec"] = df[emb_col].apply(_parse_embedding)
    df = df[df["embedding_vec"].notna()].copy()
    df["gls"] = pd.to_numeric(df["gls"], errors="coerce")
    df = df[df["gls"].notna()].copy()
    if args.gls_min is not None:
        df = df[df["gls"] >= args.gls_min].copy()
    if args.gls_max is not None:
        df = df[df["gls"] <= args.gls_max].copy()
    if args.gls_min is not None or args.gls_max is not None:
        logger.info(
            "Filtered GLS range to [%s, %s]; remaining rows: %d",
            "min" if args.gls_min is None else f"{args.gls_min:g}",
            "max" if args.gls_max is None else f"{args.gls_max:g}",
            len(df),
        )

    if df.empty:
        raise ValueError("No valid rows with embeddings and GLS targets.")

    emb_lengths = df["embedding_vec"].apply(lambda x: x.shape[0]).unique()
    if len(emb_lengths) != 1:
        if args.embedding_dim is None and "embedding_dim" in df.columns:
            # Auto-select the most frequent dimension as a fallback for mixed MAE/STF files.
            dim_counts = df["embedding_dim"].value_counts()
            target_dim = int(dim_counts.idxmax())
            df = df[df["embedding_dim"] == target_dim].copy()
            emb_lengths = df["embedding_vec"].apply(lambda x: x.shape[0]).unique()
            logger.info(
                f"Detected mixed embedding dims; auto-selected embedding_dim={target_dim}. "
                f"Remaining rows: {len(df)}"
            )
        else:
            raise ValueError(f"Inconsistent embedding dimensions: {emb_lengths}")
    input_dim = int(emb_lengths[0])

    patient_col = _find_column(df, ["patient_id", "patient_num"], required=False)
    if patient_col is None:
        if "patient_id" in df.columns:
            patient_col = "patient_id"
        elif "patient_num" in df.columns:
            patient_col = "patient_num"
    if patient_col is None:
        raise ValueError("No patient identifier column found for group split.")

    group_cols = _resolve_group_cols(df, patient_col, args.group_by_view)
    needs_time = args.target_mode == "delta" or args.objective == "ranking"
    time_col = None
    if needs_time:
        time_col = _resolve_time_column(df, args.time_col)
        if time_col is None:
            raise ValueError("No visit time column found; pass --time-col.")
        time_vals = _coerce_time_values(df[time_col])
        if time_vals is None or time_vals.isna().all():
            raise ValueError(f"Could not parse visit time column '{time_col}'.")
        df = df.copy()
        df["_visit_time"] = time_vals
        before = len(df)
        df = df[df["_visit_time"].notna()].copy()
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d rows with missing visit time.", dropped)

    df, target_label = _apply_target_transform(df, args.target_mode, group_cols, time_col)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df[df["target"].notna()].copy()
    if args.target_min is not None:
        df = df[df["target"] >= args.target_min].copy()
    if args.target_max is not None:
        df = df[df["target"] <= args.target_max].copy()
    if args.target_min is not None or args.target_max is not None:
        logger.info(
            "Filtered %s range to [%s, %s]; remaining rows: %d",
            target_label,
            "min" if args.target_min is None else f"{args.target_min:g}",
            "max" if args.target_max is None else f"{args.target_max:g}",
            len(df),
        )
    if df.empty:
        raise ValueError("No valid rows with embeddings and target values.")

    groups = df[patient_col].astype(str).fillna("unknown")
    x = np.stack(df["embedding_vec"].to_list()).astype(np.float32)
    y = df["target"].astype(np.float32).values

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(x, y, groups=groups))
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    y_train_raw = y_train.copy()
    y_val_raw = y_val.copy()
    logger.info(
        "%s train: mean=%.4f std=%.4f | val: mean=%.4f std=%.4f",
        target_label,
        float(np.mean(y_train_raw)),
        float(np.std(y_train_raw)),
        float(np.mean(y_val_raw)),
        float(np.std(y_val_raw)),
    )

    if args.standardize:
        # Standardize embeddings and targets using training statistics.
        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0)
        x_std = np.maximum(x_std, 1e-6)
        x_train = ((x_train - x_mean) / x_std).astype(np.float32)
        x_val = ((x_val - x_mean) / x_std).astype(np.float32)

        if args.objective == "regression":
            y_mean = float(y_train.mean())
            y_std = float(y_train.std())
            y_std = max(y_std, 1e-6)
            y_train = ((y_train - y_mean) / y_std).astype(np.float32)
            y_val = ((y_val - y_mean) / y_std).astype(np.float32)
        else:
            y_train = y_train.astype(np.float32)
            y_val = y_val.astype(np.float32)
            y_mean = 0.0
            y_std = 1.0
    else:
        x_train = x_train.astype(np.float32)
        x_val = x_val.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        y_mean = 0.0
        y_std = 1.0

    train_ds = EmbeddingDataset(x_train, y_train)
    val_ds = EmbeddingDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    pairwise_train_loader = None
    pairwise_val_loader = None
    train_eval_loader = None
    val_eval_loader = None
    if args.objective == "ranking":
        if "_visit_time" not in df.columns:
            raise ValueError("Ranking objective requires visit time; check --time-col.")
        df_train = df.iloc[train_idx].copy().reset_index(drop=True)
        df_val = df.iloc[val_idx].copy().reset_index(drop=True)
        rng = np.random.RandomState(args.seed)
        train_pairs, train_labels = _build_pairwise_pairs(
            df_train,
            group_cols=group_cols,
            order_col="_visit_time",
            label_source=args.pairwise_label,
            direction=args.ranking_direction,
            max_pairs_per_group=args.max_pairs_per_group,
            rng=rng,
        )
        val_pairs, val_labels = _build_pairwise_pairs(
            df_val,
            group_cols=group_cols,
            order_col="_visit_time",
            label_source=args.pairwise_label,
            direction=args.ranking_direction,
            max_pairs_per_group=args.max_pairs_per_group,
            rng=rng,
        )
        if len(train_pairs) == 0 or len(val_pairs) == 0:
            raise ValueError("Not enough visit pairs for ranking objective.")
        pairwise_train_loader = DataLoader(
            PairwiseRankingDataset(x_train, train_pairs, train_labels),
            batch_size=args.batch_size,
            shuffle=True,
        )
        pairwise_val_loader = DataLoader(
            PairwiseRankingDataset(x_val, val_pairs, val_labels),
            batch_size=args.batch_size,
            shuffle=False,
        )
        train_eval_loader = DataLoader(
            EmbeddingDataset(x_train, y_train_raw),
            batch_size=args.batch_size,
            shuffle=False,
        )
        val_eval_loader = DataLoader(
            EmbeddingDataset(x_val, y_val_raw),
            batch_size=args.batch_size,
            shuffle=False,
        )
        logger.info("Ranking pairs: train=%d val=%d", len(train_pairs), len(val_pairs))

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device not in ("auto", "cpu", "cuda"):
        device = torch.device(args.device)

    results = {}
    base_args = dict(vars(args))
    if args.model in ("cnn", "both", "all"):
        logger.info("Training CNN regressor...")
        model = CNNRegressor(input_dim)
        out_path = args.output_dir / "cnn_best.pt"
        cnn_log_dir = base_log_dir / "cnn" if args.model in ("both", "all") else base_log_dir
        run_args = dict(base_args)
        run_args["trained_model"] = "cnn"
        metrics = _train_model(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            out_path=out_path,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            lr_min=args.lr_min,
            y_mean=y_mean,
            y_std=y_std,
            loss_type=args.loss,
            log_dir=cnn_log_dir,
            plot_every=10,
            run_args=run_args,
            objective=args.objective,
            pairwise_train_loader=pairwise_train_loader,
            pairwise_val_loader=pairwise_val_loader,
            ranking_loss=args.ranking_loss,
            ranking_margin=args.ranking_margin,
            train_eval_loader=train_eval_loader,
            val_eval_loader=val_eval_loader,
            y_train_raw=y_train_raw,
            y_val_raw=y_val_raw,
            target_label=target_label,
        )
        results["cnn"] = metrics

    if args.model in ("transformer", "both", "all"):
        logger.info("Training Transformer regressor...")
        model = TransformerRegressor(input_dim)
        out_path = args.output_dir / "transformer_best.pt"
        transformer_log_dir = base_log_dir / "transformer" if args.model in ("both", "all") else base_log_dir
        run_args = dict(base_args)
        run_args["trained_model"] = "transformer"
        metrics = _train_model(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            out_path=out_path,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            lr_min=args.lr_min,
            y_mean=y_mean,
            y_std=y_std,
            loss_type=args.loss,
            log_dir=transformer_log_dir,
            plot_every=10,
            run_args=run_args,
            objective=args.objective,
            pairwise_train_loader=pairwise_train_loader,
            pairwise_val_loader=pairwise_val_loader,
            ranking_loss=args.ranking_loss,
            ranking_margin=args.ranking_margin,
            train_eval_loader=train_eval_loader,
            val_eval_loader=val_eval_loader,
            y_train_raw=y_train_raw,
            y_val_raw=y_val_raw,
            target_label=target_label,
        )
        results["transformer"] = metrics

    if args.model in ("mlp", "all"):
        logger.info("Training MLP regressor...")
        model = MLPRegressor(
            input_dim,
            hidden=args.mlp_hidden,
            depth=args.mlp_depth,
            dropout=args.mlp_dropout,
        )
        out_path = args.output_dir / "mlp_best.pt"
        mlp_log_dir = base_log_dir / "mlp" if args.model == "all" else base_log_dir
        run_args = dict(base_args)
        run_args["trained_model"] = "mlp"
        metrics = _train_model(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            out_path=out_path,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            lr_min=args.lr_min,
            y_mean=y_mean,
            y_std=y_std,
            loss_type=args.loss,
            log_dir=mlp_log_dir,
            plot_every=10,
            run_args=run_args,
            objective=args.objective,
            pairwise_train_loader=pairwise_train_loader,
            pairwise_val_loader=pairwise_val_loader,
            ranking_loss=args.ranking_loss,
            ranking_margin=args.ranking_margin,
            train_eval_loader=train_eval_loader,
            val_eval_loader=val_eval_loader,
            y_train_raw=y_train_raw,
            y_val_raw=y_val_raw,
            target_label=target_label,
        )
        results["mlp"] = metrics

    if args.model in ("mlp_unc", "all"):
        logger.info("Training MLP heteroscedastic regressor...")
        model = MLPHeteroscedastic(
            input_dim,
            hidden=args.mlp_hidden,
            depth=args.mlp_depth,
            dropout=args.mlp_dropout,
        )
        out_path = args.output_dir / "mlp_unc_best.pt"
        mlp_unc_log_dir = base_log_dir / "mlp_unc" if args.model == "all" else base_log_dir
        run_args = dict(base_args)
        run_args["trained_model"] = "mlp_unc"
        metrics = _train_model(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            out_path=out_path,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            lr_min=args.lr_min,
            y_mean=y_mean,
            y_std=y_std,
            loss_type=args.loss,
            log_dir=mlp_unc_log_dir,
            plot_every=10,
            run_args=run_args,
            objective=args.objective,
            pairwise_train_loader=pairwise_train_loader,
            pairwise_val_loader=pairwise_val_loader,
            ranking_loss=args.ranking_loss,
            ranking_margin=args.ranking_margin,
            train_eval_loader=train_eval_loader,
            val_eval_loader=val_eval_loader,
            y_train_raw=y_train_raw,
            y_val_raw=y_val_raw,
            target_label=target_label,
        )
        results["mlp_unc"] = metrics

    run_baselines = args.run_baselines or args.model == "all"
    run_tree_baselines = args.run_tree_baselines or args.model == "all"
    if run_baselines:
        logger.info("Running linear baselines...")
        results["baselines"] = _run_linear_baselines(
            x_train,
            y_train_raw,
            x_val,
            y_val_raw,
        )
    if run_tree_baselines:
        logger.info("Running tree baselines...")
        results["tree_baselines"] = _run_tree_baselines(
            x_train,
            y_train_raw,
            x_val,
            y_val_raw,
            seed=args.seed,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
