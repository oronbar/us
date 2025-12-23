"""
Train baseline models to classify good vs bad strain curves and to regress
time since first visit.

Classification labels (per patient, by visit order):
  - visit_index 0..good_visits-1 -> good (label 0)
  - visit_index good_visits..good_visits+bad_visits-1 -> bad (label 1)
Only patients with >= good_visits + bad_visits visits are used for classification.

Regression target:
  - days since first visit (study_datetime) for each sample.

Each sample is one segment curve (per row). Curves are resampled to length T
and flattened to a feature vector.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

RESULT_TABLE_COLUMNS = [
    "task",
    "curve",
    "model_type",
    "view_filter",
    "T",
    "n_train",
    "n_val",
    "n_patients_train",
    "n_patients_val",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "mae",
    "rmse",
    "r2",
    "val_loss",
    "early_stop_epoch",
    "early_stop_patience",
    "early_stop_delta",
    "tensorboard_logdir",
    "confusion_matrix",
    "label_counts",
    "model_path",
    "confusion_matrix_plot",
    "regression_plot",
    "error",
]


def expand_curve_choice(curve_choice: str) -> List[str]:
    if curve_choice == "both":
        return ["delta", "endo", "endo_epi"]
    return [curve_choice]


def resolve_curve_cols(curve_key: str) -> List[str]:
    if curve_key == "endo_epi":
        return ["endo", "epi"]
    return [curve_key]


def set_seed(seed: int, use_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if use_torch:
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def resample_curve(y: np.ndarray, T: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return np.zeros(T, dtype=float)
    if n == T:
        return y.copy()
    x_old = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, T)
    return np.interp(x_new, x_old, y)


def coerce_curve_array(val: object) -> np.ndarray:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.array([], dtype=float)
    if isinstance(val, np.ndarray):
        return val.astype(float, copy=False)
    if isinstance(val, list):
        return np.asarray(val, dtype=float)
    if isinstance(val, str):
        try:
            import ast
            arr = ast.literal_eval(val)
            return np.asarray(arr, dtype=float)
        except Exception:
            return np.array([], dtype=float)
    try:
        return np.asarray(val, dtype=float)
    except Exception:
        return np.array([], dtype=float)


def pick_patient_col(df: pd.DataFrame) -> str:
    for c in ("patient_key", "short_id", "patient_id"):
        if c in df.columns:
            return c
    raise ValueError("No patient identifier column found (expected patient_key/short_id/patient_id).")


def add_visit_index(df: pd.DataFrame, patient_col: str) -> pd.DataFrame:
    out = df.copy()
    out["study_datetime"] = pd.to_datetime(out["study_datetime"], errors="coerce")
    visits = (
        out[[patient_col, "study_datetime"]]
        .dropna(subset=["study_datetime"])
        .drop_duplicates()
        .sort_values([patient_col, "study_datetime"])
    )
    visits["visit_index"] = visits.groupby(patient_col).cumcount()
    out = out.merge(visits, on=[patient_col, "study_datetime"], how="inner")
    return out


def prepare_classification_df(
    df: pd.DataFrame,
    patient_col: str,
    good_visits: int,
    bad_visits: int,
) -> pd.DataFrame:
    out = add_visit_index(df, patient_col)
    min_visits = good_visits + bad_visits
    visit_counts = out.groupby(patient_col)["visit_index"].max().add(1)
    keep_ids = visit_counts[visit_counts >= min_visits].index
    out = out[out[patient_col].isin(keep_ids)].copy()
    out = out[out["visit_index"] < min_visits].copy()
    out["label"] = (out["visit_index"] >= good_visits).astype(int)
    return out


def prepare_regression_df(df: pd.DataFrame, patient_col: str) -> pd.DataFrame:
    out = df.copy()
    out["study_datetime"] = pd.to_datetime(out["study_datetime"], errors="coerce")
    out = out.dropna(subset=["study_datetime"]).copy()
    baseline = out.groupby(patient_col)["study_datetime"].transform("min")
    out["days_since_first"] = (out["study_datetime"] - baseline).dt.total_seconds() / 86400.0
    return out


def build_samples(
    df: pd.DataFrame,
    curve_col: str,
    label_col: str,
    patient_col: str,
    T: int,
    view_filter: Optional[str] = None,
    return_3d: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    curve_cols = resolve_curve_cols(curve_col)
    missing_curve_cols = [c for c in curve_cols if c not in df.columns]
    if missing_curve_cols:
        raise ValueError(f"Missing curve column(s): {missing_curve_cols}")
    required = [patient_col, "view", "segment", label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    features: List[np.ndarray] = []
    targets: List[float] = []
    patients: List[str] = []

    for _, row in df.iterrows():
        view = str(row["view"]).strip().upper()
        if view_filter and view != view_filter:
            continue
        seg_order = VIEW_TO_SEGS.get(view)
        if not seg_order:
            continue
        seg = str(row["segment"]).strip()
        if seg not in seg_order:
            continue
        y = row[label_col]
        if pd.isna(y):
            continue
        curve_features: List[np.ndarray] = []
        ok = True
        for col in curve_cols:
            arr = coerce_curve_array(row[col])
            if arr.size < 2:
                ok = False
                break
            res = resample_curve(arr, T)
            res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
            curve_features.append(res.astype(np.float32, copy=False))
        if not ok:
            continue
        if return_3d:
            features.append(np.stack(curve_features, axis=0))
        else:
            features.append(np.concatenate(curve_features, axis=0))
        targets.append(float(y))
        patients.append(str(row[patient_col]))

    if not features:
        n_features = T * len(curve_cols)
        if return_3d:
            return np.zeros((0, len(curve_cols), T), dtype=np.float32), np.array([], dtype=float), np.array([], dtype=object)
        return np.zeros((0, n_features), dtype=np.float32), np.array([], dtype=float), np.array([], dtype=object)

    X = np.stack(features, axis=0)
    y = np.asarray(targets, dtype=float)
    p = np.asarray(patients, dtype=object)
    return X, y, p


def split_by_patient(
    patients: np.ndarray,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(patients)
    train_ids, val_ids = train_test_split(uniq, test_size=test_size, random_state=seed, shuffle=True)
    train_mask = np.isin(patients, train_ids)
    val_mask = np.isin(patients, val_ids)
    return train_mask, val_mask


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], outpath: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, outpath: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, s=14, alpha=0.7, edgecolors="none")
    if y_true.size and y_pred.size:
        minv = float(min(np.min(y_true), np.min(y_pred)))
        maxv = float(max(np.max(y_true), np.max(y_pred)))
        if np.isfinite(minv) and np.isfinite(maxv) and minv != maxv:
            ax.plot([minv, maxv], [minv, maxv], "--", color="gray", linewidth=1)
    ax.set_xlabel("True days since first")
    ax.set_ylabel("Predicted days since first")
    ax.set_title(title)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def standardize_train_val(
    X_train: np.ndarray, X_val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    X_train_s = (X_train - mean) / std
    X_val_s = (X_val - mean) / std
    return X_train_s, X_val_s, mean, std


def build_torch_model(model_type: str, in_channels: int, T: int):
    import torch
    import torch.nn as nn

    class SmallCNN(nn.Module):
        def __init__(self, in_ch: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            h = self.net(x)
            h = h.squeeze(-1)
            return self.head(h).squeeze(-1)

    class SmallTransformer(nn.Module):
        def __init__(self, in_ch: int, seq_len: int):
            super().__init__()
            d_model = 32
            nhead = 4
            nlayers = 2
            self.input_proj = nn.Linear(in_ch, d_model)
            self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x):
            # x: [B, C, T] -> [B, T, C]
            h = x.permute(0, 2, 1)
            h = self.input_proj(h) + self.pos[:, : h.size(1), :]
            h = self.encoder(h)
            h = h.mean(dim=1)
            return self.head(h).squeeze(-1)

    if model_type == "cnn":
        return SmallCNN(in_channels)
    if model_type == "transformer":
        return SmallTransformer(in_channels, T)
    raise ValueError(f"Unknown model_type: {model_type}")


def run_classification(
    X: np.ndarray,
    y: np.ndarray,
    patients: np.ndarray,
    outdir: Path,
    tag: str,
    test_size: float,
    seed: int,
    model_type: str = "linear",
) -> Dict[str, object]:
    if X.shape[0] == 0:
        raise ValueError("No samples available for classification.")
    train_mask, val_mask = split_by_patient(patients, test_size=test_size, seed=seed)
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    if len(np.unique(y_train)) < 2:
        raise ValueError("Training set has a single class; cannot fit classifier.")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_val)[:, 1]
        except Exception:
            y_prob = None

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    metrics: Dict[str, object] = {
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_patients_train": int(np.unique(patients[train_mask]).size),
        "n_patients_val": int(np.unique(patients[val_mask]).size),
        "model_type": model_type,
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }
    if y_prob is not None and len(np.unique(y_val)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_val, y_prob))
    else:
        metrics["roc_auc"] = None

    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / f"classifier_{tag}.joblib"
    dump(model, model_path)
    metrics["model_path"] = str(model_path)
    plot_path = outdir / f"confusion_matrix_{tag}.png"
    plot_confusion_matrix(cm, ["good", "bad"], plot_path, f"Confusion Matrix ({tag})")
    metrics["confusion_matrix_plot"] = str(plot_path)
    return metrics


def run_regression(
    X: np.ndarray,
    y: np.ndarray,
    patients: np.ndarray,
    outdir: Path,
    tag: str,
    test_size: float,
    seed: int,
    model_type: str = "linear",
) -> Dict[str, object]:
    if X.shape[0] == 0:
        raise ValueError("No samples available for regression.")
    train_mask, val_mask = split_by_patient(patients, test_size=test_size, seed=seed)
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    metrics: Dict[str, object] = {
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_patients_train": int(np.unique(patients[train_mask]).size),
        "n_patients_val": int(np.unique(patients[val_mask]).size),
        "model_type": model_type,
        "mae": float(mean_absolute_error(y_val, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_val, y_pred)),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / f"regressor_{tag}.joblib"
    dump(model, model_path)
    metrics["model_path"] = str(model_path)
    plot_path = outdir / f"regression_{tag}.png"
    plot_regression_scatter(y_val, y_pred, plot_path, f"Regression ({tag})")
    metrics["regression_plot"] = str(plot_path)
    return metrics


def run_classification_torch(
    X: np.ndarray,
    y: np.ndarray,
    patients: np.ndarray,
    outdir: Path,
    tag: str,
    test_size: float,
    seed: int,
    model_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    logdir: Optional[Path] = None,
    early_stop_patience: int = 5,
    early_stop_delta: float = 0.0,
) -> Dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.tensorboard import SummaryWriter

    if X.shape[0] == 0:
        raise ValueError("No samples available for classification.")
    train_mask, val_mask = split_by_patient(patients, test_size=test_size, seed=seed)
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    if len(np.unique(y_train)) < 2:
        raise ValueError("Training set has a single class; cannot fit classifier.")

    X_train, X_val, mean, std = standardize_train_val(X_train, X_val)
    X_train = X_train.astype(np.float32, copy=False)
    X_val = X_val.astype(np.float32, copy=False)
    y_train = y_train.astype(np.float32, copy=False)
    y_val = y_val.astype(np.float32, copy=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_torch_model(model_type, in_channels=X_train.shape[1], T=X_train.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")

    def eval_loader(loader):
        model.eval()
        losses = []
        all_logits = []
        all_y = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                losses.append(loss.item() * yb.numel())
                all_logits.append(logits.detach().cpu().numpy())
                all_y.append(yb.detach().cpu().numpy())
        y_true = np.concatenate(all_y) if all_y else np.array([], dtype=float)
        logits = np.concatenate(all_logits) if all_logits else np.array([], dtype=float)
        loss = float(np.sum(losses) / max(1, y_true.size))
        return loss, y_true, logits

    writer = None
    log_path = None
    if logdir is not None:
        log_path = Path(logdir) / tag
        writer = SummaryWriter(log_dir=str(log_path))

    prev_train = None
    prev_val = None
    diverge_count = 0
    stop_epoch = None
    for ep in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * yb.numel()
            train_n += yb.numel()
        train_loss = float(train_loss_sum / max(1, train_n))
        val_loss, _, _ = eval_loader(val_loader)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if prev_train is not None and prev_val is not None:
            train_down = train_loss < prev_train - early_stop_delta
            val_up = val_loss > prev_val + early_stop_delta
            if train_down and val_up:
                diverge_count += 1
            else:
                diverge_count = 0
            if diverge_count >= early_stop_patience:
                stop_epoch = ep
                break
        if writer is not None:
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, ep)
        prev_train = train_loss
        prev_val = val_loss

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, y_true, logits = eval_loader(val_loader)
    y_prob = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics: Dict[str, object] = {
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_patients_train": int(np.unique(patients[train_mask]).size),
        "n_patients_val": int(np.unique(patients[val_mask]).size),
        "model_type": model_type,
        "val_loss": float(val_loss),
        "early_stop_epoch": stop_epoch,
        "early_stop_patience": int(early_stop_patience),
        "early_stop_delta": float(early_stop_delta),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }
    if log_path is not None:
        metrics["tensorboard_logdir"] = str(log_path)
    if y_true.size > 0 and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None

    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / f"classifier_{tag}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean.astype(np.float32, copy=False),
            "std": std.astype(np.float32, copy=False),
            "model_type": model_type,
            "in_channels": int(X_train.shape[1]),
            "T": int(X_train.shape[2]),
        },
        model_path,
    )
    metrics["model_path"] = str(model_path)
    plot_path = outdir / f"confusion_matrix_{tag}.png"
    plot_confusion_matrix(cm, ["good", "bad"], plot_path, f"Confusion Matrix ({tag})")
    metrics["confusion_matrix_plot"] = str(plot_path)
    if writer is not None:
        writer.flush()
        writer.close()
    return metrics


def run_regression_torch(
    X: np.ndarray,
    y: np.ndarray,
    patients: np.ndarray,
    outdir: Path,
    tag: str,
    test_size: float,
    seed: int,
    model_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    logdir: Optional[Path] = None,
    early_stop_patience: int = 5,
    early_stop_delta: float = 0.0,
) -> Dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.tensorboard import SummaryWriter

    if X.shape[0] == 0:
        raise ValueError("No samples available for regression.")
    train_mask, val_mask = split_by_patient(patients, test_size=test_size, seed=seed)
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    X_train, X_val, mean, std = standardize_train_val(X_train, X_val)
    X_train = X_train.astype(np.float32, copy=False)
    X_val = X_val.astype(np.float32, copy=False)
    y_train = y_train.astype(np.float32, copy=False)
    y_val = y_val.astype(np.float32, copy=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_torch_model(model_type, in_channels=X_train.shape[1], T=X_train.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")

    def eval_loader(loader):
        model.eval()
        losses = []
        preds = []
        ys = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                losses.append(loss.item() * yb.numel())
                preds.append(pred.detach().cpu().numpy())
                ys.append(yb.detach().cpu().numpy())
        y_true = np.concatenate(ys) if ys else np.array([], dtype=float)
        y_pred = np.concatenate(preds) if preds else np.array([], dtype=float)
        loss = float(np.sum(losses) / max(1, y_true.size))
        return loss, y_true, y_pred

    writer = None
    log_path = None
    if logdir is not None:
        log_path = Path(logdir) / tag
        writer = SummaryWriter(log_dir=str(log_path))

    prev_train = None
    prev_val = None
    diverge_count = 0
    stop_epoch = None
    for ep in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * yb.numel()
            train_n += yb.numel()
        train_loss = float(train_loss_sum / max(1, train_n))
        val_loss, _, _ = eval_loader(val_loader)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if prev_train is not None and prev_val is not None:
            train_down = train_loss < prev_train - early_stop_delta
            val_up = val_loss > prev_val + early_stop_delta
            if train_down and val_up:
                diverge_count += 1
            else:
                diverge_count = 0
            if diverge_count >= early_stop_patience:
                stop_epoch = ep
                break
        if writer is not None:
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, ep)
        prev_train = train_loss
        prev_val = val_loss

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, y_true, y_pred = eval_loader(val_loader)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))) if y_true.size else float("nan")
    metrics: Dict[str, object] = {
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_patients_train": int(np.unique(patients[train_mask]).size),
        "n_patients_val": int(np.unique(patients[val_mask]).size),
        "model_type": model_type,
        "val_loss": float(val_loss),
        "early_stop_epoch": stop_epoch,
        "early_stop_patience": int(early_stop_patience),
        "early_stop_delta": float(early_stop_delta),
        "mae": float(mean_absolute_error(y_true, y_pred)) if y_true.size else float("nan"),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)) if y_true.size else float("nan"),
    }
    if log_path is not None:
        metrics["tensorboard_logdir"] = str(log_path)

    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / f"regressor_{tag}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean.astype(np.float32, copy=False),
            "std": std.astype(np.float32, copy=False),
            "model_type": model_type,
            "in_channels": int(X_train.shape[1]),
            "T": int(X_train.shape[2]),
        },
        model_path,
    )
    metrics["model_path"] = str(model_path)
    plot_path = outdir / f"regression_{tag}.png"
    plot_regression_scatter(y_true, y_pred, plot_path, f"Regression ({tag})")
    metrics["regression_plot"] = str(plot_path)
    if writer is not None:
        writer.flush()
        writer.close()
    return metrics


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def to_builtin(val: object) -> object:
    if isinstance(val, (np.floating, np.integer)):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def sort_nested_results(section: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(section, dict):
        return section
    ordered: Dict[str, object] = {}
    for key in sorted(section.keys()):
        val = section[key]
        if isinstance(val, dict):
            ordered[key] = {k: val[k] for k in sorted(val.keys())}
        else:
            ordered[key] = val
    return ordered


def collect_results_rows(results: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    view_filter = results.get("view_filter")
    T = results.get("T")
    for task in ("classification", "regression"):
        section = results.get(task)
        if not isinstance(section, dict):
            continue
        for curve, curve_val in section.items():
            if isinstance(curve_val, dict) and "error" in curve_val:
                rows.append(
                    {
                        "task": task,
                        "curve": curve,
                        "model_type": None,
                        "view_filter": view_filter,
                        "T": T,
                        "error": curve_val.get("error"),
                    }
                )
                continue
            if isinstance(curve_val, dict):
                for model_type, metrics in curve_val.items():
                    if isinstance(metrics, dict):
                        row = {
                            "task": task,
                            "curve": curve,
                            "model_type": model_type,
                            "view_filter": view_filter,
                            "T": T,
                        }
                        for k, v in metrics.items():
                            if k in ("confusion_matrix", "label_counts"):
                                row[k] = json.dumps(v)
                            else:
                                row[k] = to_builtin(v)
                        rows.append(row)
            else:
                rows.append(
                    {
                        "task": task,
                        "curve": curve,
                        "model_type": None,
                        "view_filter": view_filter,
                        "T": T,
                        "error": "unexpected format",
                    }
                )
    return rows


def write_results_table(rows: List[Dict[str, object]], outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in RESULT_TABLE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[RESULT_TABLE_COLUMNS]
    df.to_csv(outpath, index=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train visit-based strain curve models.")
    ap.add_argument(
        "--parquet",
        type=Path,
        default=Path(r"F:\DS\strain_dataset_combined.parquet"),
        help="Path to combined parquet file.",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("models"),
        help="Output directory for models and metrics.",
    )
    ap.add_argument(
        "--curve",
        type=str,
        choices=["delta", "endo", "endo_epi", "both"],
        default="both",
        help="Curve column to use (endo_epi concatenates endo+epi).",
    )
    ap.add_argument(
        "--task",
        type=str,
        choices=["classify", "regress", "both"],
        default="both",
        help="Task to run.",
    )
    ap.add_argument(
        "--model",
        type=str,
        choices=["linear", "cnn", "transformer", "all"],
        default="all",
        help="Model type to train.",
    )
    ap.add_argument(
        "--view",
        type=str,
        choices=["2C", "4C", "both"],
        default="both",
        help="View filter.",
    )
    ap.add_argument("--T", type=int, default=64, help="Resample length for each curve.")
    ap.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction by patient.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20, help="Epochs for CNN/Transformer training.")
    ap.add_argument("--batch", type=int, default=64, help="Batch size for CNN/Transformer training.")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate for CNN/Transformer training.")
    ap.add_argument(
        "--logdir",
        type=Path,
        default=None,
        help="Optional TensorBoard log directory. Defaults to <outdir>/<run_stamp>/tensorboard.",
    )
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Stop when train decreases and val increases for this many consecutive epochs.",
    )
    ap.add_argument(
        "--early-stop-delta",
        type=float,
        default=0.0,
        help="Minimum change required to count as divergence.",
    )
    ap.add_argument("--good-visits", type=int, default=2, help="Number of earliest visits labeled good.")
    ap.add_argument("--bad-visits", type=int, default=2, help="Number of subsequent visits labeled bad.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    use_torch = args.model in ("cnn", "transformer", "all")
    set_seed(args.seed, use_torch=use_torch)

    if not args.parquet.is_file():
        raise FileNotFoundError(f"Parquet not found: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    patient_col = pick_patient_col(df)

    view_filter = None if args.view == "both" else args.view
    curve_cols: Iterable[str] = expand_curve_choice(args.curve)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.outdir / run_stamp
    logdir = args.logdir if args.logdir is not None else run_dir / "tensorboard"

    model_types = ["linear", "cnn", "transformer"] if args.model == "all" else [args.model]

    results: Dict[str, object] = {
        "parquet": str(args.parquet),
        "patient_col": patient_col,
        "view_filter": view_filter or "both",
        "T": args.T,
        "seed": args.seed,
        "curve_choice": args.curve,
        "run_dir": str(run_dir),
        "model_choice": args.model,
        "model_types": model_types,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "tensorboard_logdir": str(logdir),
        "early_stop_patience": args.early_stop_patience,
        "early_stop_delta": args.early_stop_delta,
    }

    if args.task in ("classify", "both"):
        clf_df = prepare_classification_df(
            df, patient_col=patient_col, good_visits=args.good_visits, bad_visits=args.bad_visits
        )
        results["classification"] = {}
        for curve_col in curve_cols:
            X3, y, p = build_samples(
                clf_df,
                curve_col=curve_col,
                label_col="label",
                patient_col=patient_col,
                T=args.T,
                view_filter=view_filter,
                return_3d=True,
            )
            if X3.shape[0] == 0:
                results["classification"][curve_col] = {"error": "no samples"}
                continue
            X_flat = X3.reshape(X3.shape[0], -1)
            label_counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
            results["classification"][curve_col] = {}
            for model_type in model_types:
                tag = f"{curve_col}_T{args.T}_view{view_filter or 'both'}_model{model_type}_taskclass"
                try:
                    if model_type == "linear":
                        metrics = run_classification(
                            X_flat, y, p, outdir=run_dir, tag=tag,
                            test_size=args.test_size, seed=args.seed, model_type=model_type
                        )
                    else:
                        metrics = run_classification_torch(
                            X3, y, p, outdir=run_dir, tag=tag,
                            test_size=args.test_size, seed=args.seed, model_type=model_type,
                            epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                            logdir=logdir,
                            early_stop_patience=args.early_stop_patience,
                            early_stop_delta=args.early_stop_delta,
                        )
                    metrics["label_counts"] = label_counts
                    results["classification"][curve_col][model_type] = metrics
                except Exception as exc:
                    results["classification"][curve_col][model_type] = {
                        "error": str(exc),
                        "label_counts": label_counts,
                    }

    if args.task in ("regress", "both"):
        reg_df = prepare_regression_df(df, patient_col=patient_col)
        results["regression"] = {}
        for curve_col in curve_cols:
            X3, y, p = build_samples(
                reg_df,
                curve_col=curve_col,
                label_col="days_since_first",
                patient_col=patient_col,
                T=args.T,
                view_filter=view_filter,
                return_3d=True,
            )
            if X3.shape[0] == 0:
                results["regression"][curve_col] = {"error": "no samples"}
                continue
            X_flat = X3.reshape(X3.shape[0], -1)
            results["regression"][curve_col] = {}
            for model_type in model_types:
                tag = f"{curve_col}_T{args.T}_view{view_filter or 'both'}_model{model_type}_taskreg"
                try:
                    if model_type == "linear":
                        metrics = run_regression(
                            X_flat, y, p, outdir=run_dir, tag=tag,
                            test_size=args.test_size, seed=args.seed, model_type=model_type
                        )
                    else:
                        metrics = run_regression_torch(
                            X3, y, p, outdir=run_dir, tag=tag,
                            test_size=args.test_size, seed=args.seed, model_type=model_type,
                            epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                            logdir=logdir,
                            early_stop_patience=args.early_stop_patience,
                            early_stop_delta=args.early_stop_delta,
                        )
                    results["regression"][curve_col][model_type] = metrics
                except Exception as exc:
                    results["regression"][curve_col][model_type] = {"error": str(exc)}

    if "classification" in results:
        results["classification"] = sort_nested_results(results["classification"])
    if "regression" in results:
        results["regression"] = sort_nested_results(results["regression"])

    rows = collect_results_rows(results)
    results["results_table"] = rows
    table_path = run_dir / "metrics_visit_models.csv"
    write_results_table(rows, table_path)
    results["results_table_csv"] = str(table_path)

    out_metrics = run_dir / "metrics_visit_models.json"
    write_json(out_metrics, results)
    print(f"Saved metrics to {out_metrics}")


if __name__ == "__main__":
    main()
