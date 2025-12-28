"""
Train lightweight models (CNN or Transformer) for GLS prediction from embeddings.

Input:
  - Parquet/CSV embeddings dataframe (from ichilov_encode_dicoms.py)
  - Optional Excel report to supply GLS targets

Output:
  - Model weights + metrics in output directory

Usage (PowerShell):
  .venv\\Scripts\\python ichilov_train_gls.py ^
    --input-embeddings "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Ichilov_GLS_embeddings.parquet" ^
    --report-xlsx "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Report_Ichilov_GLS_oron.xlsx" ^
    --output-dir "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Ichilov_GLS_models"
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_train_gls")

VIEW_KEYS = ("A2C", "A3C", "A4C")


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


class EmbeddingDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float().view(-1, 1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class CNNRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
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


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    preds: List[float] = []
    trues: List[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x).squeeze(1)
            preds.extend(out.cpu().numpy().tolist())
            trues.extend(y.squeeze(1).cpu().numpy().tolist())
    preds_arr = np.asarray(preds, dtype=float)
    trues_arr = np.asarray(trues, dtype=float)
    mae = float(np.mean(np.abs(preds_arr - trues_arr)))
    rmse = float(np.sqrt(np.mean((preds_arr - trues_arr) ** 2)))
    mse = float(np.mean((preds_arr - trues_arr) ** 2))
    return mae, rmse, mse


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    out_path: Path,
) -> Dict[str, float]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * x.size(0)

        val_mae, val_rmse, val_mse = _evaluate(model, val_loader, device)
        train_loss = running / len(train_loader.dataset)
        logger.info(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_mae={val_mae:.4f} | val_rmse={val_rmse:.4f}"
        )
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, out_path)

    return {"val_mse": best_val}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GLS prediction from embeddings.")
    parser.add_argument(
        "--input-embeddings",
        type=Path,
        default=r"C:\Users\oronbarazani\OneDrive - Technion\DS\Ichilov_GLS_embeddings.parquet",
        help="Embedding dataframe (parquet/csv) from ichilov_encode_dicoms.py",
    )
    parser.add_argument(
        "--report-xlsx",
        type=Path,
        required=False,
        default=r"C:\Users\oronbarazani\OneDrive - Technion\DS\Report_Ichilov_GLS_oron.xlsx",
        help="Optional report Excel to supply GLS targets if missing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=r"C:\Users\oronbarazani\OneDrive - Technion\models\Ichilov_GLS_models",
        help="Output directory for model weights and metrics",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "transformer", "both"],
        default="cnn",
        help="Model type to train",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--target-col",
        type=str,
        default="",
        help="Optional target column name (defaults to 'gls' if present)",
    )
    args = parser.parse_args()

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

    emb_col = _find_column(df, ["embedding"], required=True)
    df["embedding_vec"] = df[emb_col].apply(_parse_embedding)
    df = df[df["embedding_vec"].notna()].copy()
    df["gls"] = pd.to_numeric(df["gls"], errors="coerce")
    df = df[df["gls"].notna()].copy()

    if df.empty:
        raise ValueError("No valid rows with embeddings and GLS targets.")

    emb_lengths = df["embedding_vec"].apply(lambda x: x.shape[0]).unique()
    if len(emb_lengths) != 1:
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

    groups = df[patient_col].astype(str).fillna("unknown")
    x = np.stack(df["embedding_vec"].to_list()).astype(np.float32)
    y = df["gls"].astype(np.float32).values

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(x, y, groups=groups))
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    train_ds = EmbeddingDataset(x_train, y_train)
    val_ds = EmbeddingDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device not in ("auto", "cpu", "cuda"):
        device = torch.device(args.device)

    results = {}
    if args.model in ("cnn", "both"):
        logger.info("Training CNN regressor...")
        model = CNNRegressor(input_dim)
        out_path = args.output_dir / "cnn_best.pt"
        metrics = _train_model(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            out_path=out_path,
        )
        results["cnn"] = metrics

    if args.model in ("transformer", "both"):
        logger.info("Training Transformer regressor...")
        model = TransformerRegressor(input_dim)
        out_path = args.output_dir / "transformer_best.pt"
        metrics = _train_model(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            out_path=out_path,
        )
        results["transformer"] = metrics

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
