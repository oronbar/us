"""
Train baseline models to classify good vs bad strain curves and to regress
time since first visit.

Classification labels (per patient):
  - first_four: visit_index 0..good_visits-1 -> good; remaining up to (good+bad) -> bad
  - first_last: earliest visit -> good, latest visit -> bad
  - gls_drop: if GLS trend slope < slope_threshold and GLS magnitude drops below threshold at a visit,
    that visit and onward are bad; earlier visits are good

Regression target:
  - days since first visit (study_datetime) for each sample.

Each sample can be a single segment curve (per row) or a full view with all
4 segments. Curves are resampled to length T and flattened to a feature vector.
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
    "classify_mode",
    "gls_threshold",
    "gls_slope_threshold",
    "sample_level",
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
    "tsne_plot",
    "tsne_samples",
    "tsne_perplexity",
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
    keep_ids = visit_counts[visit_counts >= (good_visits + 1)].index
    out = out[out[patient_col].isin(keep_ids)].copy()
    out["max_visits"] = out[patient_col].map(visit_counts).clip(upper=min_visits)
    out = out[out["visit_index"] < out["max_visits"]].copy()
    out["label"] = (out["visit_index"] >= good_visits).astype(int)
    return out


def prepare_classification_df_first_last(df: pd.DataFrame, patient_col: str) -> pd.DataFrame:
    out = df.copy()
    out["study_datetime"] = pd.to_datetime(out["study_datetime"], errors="coerce")
    out = out.dropna(subset=["study_datetime"]).copy()
    visit_counts = out.groupby(patient_col)["study_datetime"].nunique()
    keep_ids = visit_counts[visit_counts >= 2].index
    out = out[out[patient_col].isin(keep_ids)].copy()
    out["min_dt"] = out.groupby(patient_col)["study_datetime"].transform("min")
    out["max_dt"] = out.groupby(patient_col)["study_datetime"].transform("max")
    out = out[(out["study_datetime"] == out["min_dt"]) | (out["study_datetime"] == out["max_dt"])].copy()
    out["label"] = (out["study_datetime"] == out["max_dt"]).astype(int)
    return out


def prepare_classification_df_gls_drop(
    df: pd.DataFrame,
    patient_col: str,
    gls_threshold: float = 15.0,
    slope_threshold: float = -0.7,
) -> pd.DataFrame:
    if "gls" not in df.columns:
        raise ValueError("Missing required column 'gls' for GLS drop classification.")
    out = df.copy()
    out["study_datetime"] = pd.to_datetime(out["study_datetime"], errors="coerce")
    out = out.dropna(subset=["study_datetime"]).copy()
    visit_gls = (
        out.groupby([patient_col, "study_datetime"])["gls"]
        .apply(lambda s: np.nanmean(np.abs(s)))
        .reset_index(name="gls_abs")
    )
    visit_gls = visit_gls.dropna(subset=["gls_abs"]).copy()
    label_rows = []
    for pid, grp in visit_gls.groupby(patient_col):
        grp = grp.sort_values("study_datetime")
        vals = grp["gls_abs"].to_numpy()
        if len(vals) < 2:
            continue
        x = np.arange(len(vals), dtype=float)
        try:
            slope = np.polyfit(x, vals, 1)[0]
        except Exception:
            slope = np.nan
        if np.isnan(slope) or slope >= slope_threshold:
            labels = np.zeros(len(vals), dtype=int)
        else:
            below = np.where(vals < gls_threshold)[0]
            if below.size == 0:
                labels = np.zeros(len(vals), dtype=int)
            else:
                first_bad = int(below[0])
                labels = (x >= first_bad).astype(int)
        tmp = grp[[patient_col, "study_datetime"]].copy()
        tmp["label"] = labels
        label_rows.append(tmp)
    if not label_rows:
        return out.iloc[0:0].copy()
    labels_df = pd.concat(label_rows, ignore_index=True)
    out = out.merge(labels_df, on=[patient_col, "study_datetime"], how="inner")
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
    return_meta: bool = False,
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
    visit_indices: List[int] = []

    visit_index_lookup = None
    if return_meta and "visit_index" not in df.columns and "study_datetime" in df.columns:
        tmp = df[[patient_col, "study_datetime"]].dropna().drop_duplicates()
        tmp["study_datetime"] = pd.to_datetime(tmp["study_datetime"], errors="coerce")
        tmp = tmp.dropna(subset=["study_datetime"]).sort_values([patient_col, "study_datetime"])
        tmp["visit_index"] = tmp.groupby(patient_col).cumcount()
        visit_index_lookup = tmp.set_index([patient_col, "study_datetime"])["visit_index"]

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
        pid = row[patient_col]
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
        patients.append(str(pid))
        if return_meta:
            visit_idx = 0
            if "visit_index" in df.columns and not pd.isna(row.get("visit_index")):
                visit_idx = int(row.get("visit_index"))
            elif visit_index_lookup is not None:
                dt = row.get("study_datetime")
                if not pd.isna(dt):
                    dt = pd.to_datetime(dt)
                    visit_idx = int(visit_index_lookup.get((pid, dt), 0))
            visit_indices.append(visit_idx)

    if not features:
        n_features = T * len(curve_cols)
        if return_3d:
            return np.zeros((0, len(curve_cols), T), dtype=np.float32), np.array([], dtype=float), np.array([], dtype=object)
        return np.zeros((0, n_features), dtype=np.float32), np.array([], dtype=float), np.array([], dtype=object)

    X = np.stack(features, axis=0)
    y = np.asarray(targets, dtype=float)
    p = np.asarray(patients, dtype=object)
    if return_meta:
        return X, y, p, np.asarray(visit_indices, dtype=int)
    return X, y, p


def _group_cols_for_view(df: pd.DataFrame, patient_col: str) -> List[str]:
    cols = [patient_col, "study_datetime", "view"]
    for c in ("study_instance_uid", "dicom_file", "cycle_index"):
        if c in df.columns:
            cols.append(c)
    return cols


def build_samples_view(
    df: pd.DataFrame,
    curve_col: str,
    label_col: str,
    patient_col: str,
    T: int,
    view_filter: Optional[str] = None,
    return_3d: bool = False,
    return_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    curve_cols = resolve_curve_cols(curve_col)
    missing_curve_cols = [c for c in curve_cols if c not in df.columns]
    if missing_curve_cols:
        raise ValueError(f"Missing curve column(s): {missing_curve_cols}")
    required = [patient_col, "study_datetime", "view", "segment", label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    features: List[np.ndarray] = []
    targets: List[float] = []
    patients: List[str] = []
    visit_indices: List[int] = []

    visit_index_lookup = None
    if return_meta and "visit_index" not in df.columns:
        if "study_datetime" in df.columns:
            tmp = df[[patient_col, "study_datetime"]].dropna().drop_duplicates()
            tmp["study_datetime"] = pd.to_datetime(tmp["study_datetime"], errors="coerce")
            tmp = tmp.dropna(subset=["study_datetime"]).sort_values([patient_col, "study_datetime"])
            tmp["visit_index"] = tmp.groupby(patient_col).cumcount()
            visit_index_lookup = tmp.set_index([patient_col, "study_datetime"])["visit_index"]

    group_cols = _group_cols_for_view(df, patient_col)
    for _, sub in df.groupby(group_cols, sort=False):
        view = str(sub["view"].iloc[0]).strip().upper()
        if view_filter and view != view_filter:
            continue
        seg_order = VIEW_TO_SEGS.get(view)
        if not seg_order:
            continue
        present = set(sub["segment"].astype(str).tolist())
        if not all(seg in present for seg in seg_order):
            continue
        y = sub[label_col].iloc[0]
        if pd.isna(y):
            continue
        pid = sub[patient_col].iloc[0]
        curve_features: List[np.ndarray] = []
        ok = True
        for seg in seg_order:
            row = sub[sub["segment"] == seg].iloc[0]
            for col in curve_cols:
                arr = coerce_curve_array(row[col])
                if arr.size < 2:
                    ok = False
                    break
                res = resample_curve(arr, T)
                res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
                curve_features.append(res.astype(np.float32, copy=False))
            if not ok:
                break
        if not ok:
            continue
        if return_3d:
            features.append(np.stack(curve_features, axis=0))
        else:
            features.append(np.concatenate(curve_features, axis=0))
        targets.append(float(y))
        patients.append(str(pid))
        if return_meta:
            visit_idx = 0
            if "visit_index" in sub.columns and not pd.isna(sub["visit_index"].iloc[0]):
                visit_idx = int(sub["visit_index"].iloc[0])
            elif visit_index_lookup is not None:
                dt = sub["study_datetime"].iloc[0]
                if not pd.isna(dt):
                    dt = pd.to_datetime(dt)
                    visit_idx = int(visit_index_lookup.get((pid, dt), 0))
            visit_indices.append(visit_idx)

    if not features:
        n_features = T * len(curve_cols) * 4
        if return_3d:
            return (
                np.zeros((0, len(curve_cols) * 4, T), dtype=np.float32),
                np.array([], dtype=float),
                np.array([], dtype=object),
            )
        return np.zeros((0, n_features), dtype=np.float32), np.array([], dtype=float), np.array([], dtype=object)

    X = np.stack(features, axis=0)
    y = np.asarray(targets, dtype=float)
    p = np.asarray(patients, dtype=object)
    if return_meta:
        return X, y, p, np.asarray(visit_indices, dtype=int)
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

    class SmallMLP(nn.Module):
        def __init__(self, in_ch: int, seq_len: int):
            super().__init__()
            hidden = 128
            in_dim = in_ch * seq_len
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.head = nn.Linear(hidden, 1)

        def forward_features(self, x):
            h = x.reshape(x.size(0), -1)
            return self.backbone(h)

        def forward(self, x):
            h = self.forward_features(x)
            return self.head(h).squeeze(-1)

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

        def forward_features(self, x):
            h = self.net(x)
            return h.squeeze(-1)

        def forward(self, x):
            h = self.forward_features(x)
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

        def forward_features(self, x):
            # x: [B, C, T] -> [B, T, C]
            h = x.permute(0, 2, 1)
            h = self.input_proj(h) + self.pos[:, : h.size(1), :]
            h = self.encoder(h)
            return h.mean(dim=1)

        def forward(self, x):
            h = self.forward_features(x)
            return self.head(h).squeeze(-1)

    if model_type == "mlp":
        return SmallMLP(in_channels, T)
    if model_type == "cnn":
        return SmallCNN(in_channels)
    if model_type == "transformer":
        return SmallTransformer(in_channels, T)
    raise ValueError(f"Unknown model_type: {model_type}")


def select_tsne_subset(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_samples <= 0 or X.shape[0] <= max_samples:
        idx = np.arange(X.shape[0])
        return X, y, idx
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_samples, replace=False)
    return X[idx], y[idx], idx


def compute_tsne_2d(X: np.ndarray, perplexity: float, seed: int) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if X.shape[0] < 3:
        return None, None
    from sklearn.manifold import TSNE

    max_per = max(1.0, (X.shape[0] - 1) / 3.0)
    per = min(perplexity, max_per)
    per = min(per, X.shape[0] - 1)
    if per < 1.0:
        per = 1.0
    tsne = TSNE(
        n_components=2,
        perplexity=per,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X), float(per)


def plot_tsne(
    z: np.ndarray,
    patient_ids: Optional[np.ndarray],
    visit_idx: Optional[np.ndarray],
    outpath: Path,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = "tab:blue"
    if patient_ids is not None and len(patient_ids):
        pids = np.asarray(patient_ids)
        uniq = pd.unique(pids)
        n = len(uniq)
        cmap = plt.get_cmap("tab20" if n <= 20 else "hsv")
        denom = max(1, n - 1)
        color_map = {pid: cmap(i / denom) for i, pid in enumerate(uniq)}
        colors = [color_map[pid] for pid in pids]

    sizes = 14
    if visit_idx is not None and len(visit_idx):
        v = np.asarray(visit_idx, dtype=float)
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        if vmax > vmin:
            sizes = 12.0 + (v - vmin) / (vmax - vmin) * 40.0
        else:
            sizes = np.full_like(v, 20.0, dtype=float)

    ax.scatter(z[:, 0], z[:, 1], c=colors, s=sizes, alpha=0.8, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def extract_embeddings_torch(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    feats: List[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            h = model.forward_features(xb)
            feats.append(h.detach().cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 0), dtype=np.float32)


def run_classification(
    X: np.ndarray,
    y: np.ndarray,
    patients: np.ndarray,
    visit_index: Optional[np.ndarray],
    outdir: Path,
    tag: str,
    test_size: float,
    seed: int,
    model_type: str = "linear",
    tsne_enabled: bool = False,
    tsne_max_samples: int = 1000,
    tsne_perplexity: float = 30.0,
    tsne_seed: int = 42,
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
    tsne_plot = None
    tsne_samples = None
    tsne_per_used = None
    if tsne_enabled:
        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        p_all = np.concatenate([patients[train_mask], patients[val_mask]])
        v_all = None
        if visit_index is not None:
            v_all = np.concatenate([visit_index[train_mask], visit_index[val_mask]])
        scaler = StandardScaler().fit(X_train)
        X_all = scaler.transform(X_all)
        X_sub, y_sub, idx = select_tsne_subset(X_all, y_all, tsne_max_samples, tsne_seed)
        p_sub = p_all[idx] if p_all is not None else None
        v_sub = v_all[idx] if v_all is not None else None
        z, per_used = compute_tsne_2d(X_sub, tsne_perplexity, tsne_seed)
        if z is not None:
            tsne_plot = outdir / f"tsne_{tag}.png"
            plot_tsne(z, p_sub, v_sub, tsne_plot, f"t-SNE ({tag})")
            tsne_samples = int(X_sub.shape[0])
            tsne_per_used = per_used

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
        "tsne_plot": str(tsne_plot) if tsne_plot is not None else None,
        "tsne_samples": tsne_samples,
        "tsne_perplexity": tsne_per_used,
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
    visit_index: Optional[np.ndarray],
    outdir: Path,
    tag: str,
    test_size: float,
    seed: int,
    model_type: str = "linear",
    tsne_enabled: bool = False,
    tsne_max_samples: int = 1000,
    tsne_perplexity: float = 30.0,
    tsne_seed: int = 42,
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
    tsne_plot = None
    tsne_samples = None
    tsne_per_used = None
    if tsne_enabled:
        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        p_all = np.concatenate([patients[train_mask], patients[val_mask]])
        v_all = None
        if visit_index is not None:
            v_all = np.concatenate([visit_index[train_mask], visit_index[val_mask]])
        scaler = StandardScaler().fit(X_train)
        X_all = scaler.transform(X_all)
        X_sub, y_sub, idx = select_tsne_subset(X_all, y_all, tsne_max_samples, tsne_seed)
        p_sub = p_all[idx] if p_all is not None else None
        v_sub = v_all[idx] if v_all is not None else None
        z, per_used = compute_tsne_2d(X_sub, tsne_perplexity, tsne_seed)
        if z is not None:
            tsne_plot = outdir / f"tsne_{tag}.png"
            plot_tsne(z, p_sub, v_sub, tsne_plot, f"t-SNE ({tag})")
            tsne_samples = int(X_sub.shape[0])
            tsne_per_used = per_used

    metrics: Dict[str, object] = {
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_patients_train": int(np.unique(patients[train_mask]).size),
        "n_patients_val": int(np.unique(patients[val_mask]).size),
        "model_type": model_type,
        "mae": float(mean_absolute_error(y_val, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_val, y_pred)),
        "tsne_plot": str(tsne_plot) if tsne_plot is not None else None,
        "tsne_samples": tsne_samples,
        "tsne_perplexity": tsne_per_used,
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
    visit_index: Optional[np.ndarray],
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
    tsne_enabled: bool = False,
    tsne_max_samples: int = 1000,
    tsne_perplexity: float = 30.0,
    tsne_seed: int = 42,
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

    tsne_plot = None
    tsne_samples = None
    tsne_per_used = None
    if tsne_enabled:
        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        p_all = np.concatenate([patients[train_mask], patients[val_mask]])
        v_all = None
        if visit_index is not None:
            v_all = np.concatenate([visit_index[train_mask], visit_index[val_mask]])
        X_sub, y_sub, idx = select_tsne_subset(X_all, y_all, tsne_max_samples, tsne_seed)
        p_sub = p_all[idx] if p_all is not None else None
        v_sub = v_all[idx] if v_all is not None else None
        emb = extract_embeddings_torch(model, X_sub, batch_size)
        z, per_used = compute_tsne_2d(emb, tsne_perplexity, tsne_seed)
        if z is not None:
            tsne_plot = outdir / f"tsne_{tag}.png"
            plot_tsne(z, p_sub, v_sub, tsne_plot, f"t-SNE ({tag})")
            tsne_samples = int(X_sub.shape[0])
            tsne_per_used = per_used

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
        "tsne_plot": str(tsne_plot) if tsne_plot is not None else None,
        "tsne_samples": tsne_samples,
        "tsne_perplexity": tsne_per_used,
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
    visit_index: Optional[np.ndarray],
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
    tsne_enabled: bool = False,
    tsne_max_samples: int = 1000,
    tsne_perplexity: float = 30.0,
    tsne_seed: int = 42,
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
    tsne_plot = None
    tsne_samples = None
    tsne_per_used = None
    if tsne_enabled:
        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        p_all = np.concatenate([patients[train_mask], patients[val_mask]])
        v_all = None
        if visit_index is not None:
            v_all = np.concatenate([visit_index[train_mask], visit_index[val_mask]])
        X_sub, y_sub, idx = select_tsne_subset(X_all, y_all, tsne_max_samples, tsne_seed)
        p_sub = p_all[idx] if p_all is not None else None
        v_sub = v_all[idx] if v_all is not None else None
        emb = extract_embeddings_torch(model, X_sub, batch_size)
        z, per_used = compute_tsne_2d(emb, tsne_perplexity, tsne_seed)
        if z is not None:
            tsne_plot = outdir / f"tsne_{tag}.png"
            plot_tsne(z, p_sub, v_sub, tsne_plot, f"t-SNE ({tag})")
            tsne_samples = int(X_sub.shape[0])
            tsne_per_used = per_used
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
        "tsne_plot": str(tsne_plot) if tsne_plot is not None else None,
        "tsne_samples": tsne_samples,
        "tsne_perplexity": tsne_per_used,
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
    classify_mode = results.get("classify_mode")
    gls_threshold = results.get("gls_threshold")
    gls_slope_threshold = results.get("gls_slope_threshold")
    sample_level = results.get("sample_level")
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
                        "classify_mode": classify_mode,
                        "gls_threshold": gls_threshold,
                        "gls_slope_threshold": gls_slope_threshold,
                        "sample_level": sample_level,
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
                            "classify_mode": classify_mode,
                            "gls_threshold": gls_threshold,
                            "gls_slope_threshold": gls_slope_threshold,
                            "sample_level": sample_level,
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
                        "classify_mode": classify_mode,
                        "gls_threshold": gls_threshold,
                        "gls_slope_threshold": gls_slope_threshold,
                        "sample_level": sample_level,
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
    user_home = Path.home()
    ap.add_argument(
        "--parquet",
        type=Path,
        default=user_home / "OneDrive - Technion" / "DS" / "strain_dataset_combined.parquet",
        help="Path to combined parquet file.",
    )
    ap.add_argument(
        "--experiment-name",
        "--experiment_name",
        type=str,
        default="VVI_strain_curves_classification",
        help="Subfolder name appended to --outdir.",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=user_home / "OneDrive - Technion" / "Experiments",
        help="Base output directory for models and metrics; experiment name is appended.",
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
        default="classify",
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
    ap.add_argument(
        "--sample-level",
        type=str,
        choices=["curve", "view"],
        default="curve",
        help="Sample granularity: curve uses per-segment curves; view uses all 4 segments together.",
    )
    ap.add_argument("--T", type=int, default=64, help="Resample length for each curve.")
    ap.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction by patient.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20, help="Epochs for CNN/Transformer training.")
    ap.add_argument("--batch", type=int, default=64, help="Batch size for CNN/Transformer training.")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate for CNN/Transformer training.")
    ap.add_argument(
        "--classify-mode",
        type=str,
        choices=["first_four", "first_last", "gls_drop"],
        default="first_four",
        help="Classification labeling: first_four uses first/second vs later; first_last uses earliest vs latest; gls_drop uses GLS trend+threshold.",
    )
    ap.add_argument(
        "--gls-threshold",
        type=float,
        default=15.0,
        help="GLS magnitude threshold used for gls_drop classification.",
    )
    ap.add_argument(
        "--gls-slope-threshold",
        type=float,
        default=-0.7,
        help="GLS slope threshold used for gls_drop classification.",
    )
    ap.add_argument(
        "--tsne",
        default=True,
        action="store_true",
        help="Enable t-SNE plots using the learned latent representations.",
    )
    ap.add_argument(
        "--tsne-max-samples",
        type=int,
        default=1000,
        help="Maximum number of samples to use for t-SNE.",
    )
    ap.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (will be capped to valid range).",
    )
    ap.add_argument(
        "--tsne-seed",
        type=int,
        default=42,
        help="Random seed for t-SNE subsampling and initialization.",
    )
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
    args = ap.parse_args()
    if args.experiment_name:
        args.outdir = args.outdir / args.experiment_name
    return args


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
        "classify_mode": args.classify_mode,
        "gls_threshold": args.gls_threshold,
        "gls_slope_threshold": args.gls_slope_threshold,
        "sample_level": args.sample_level,
        "tsne_enabled": bool(args.tsne),
        "tsne_max_samples": args.tsne_max_samples,
        "tsne_perplexity": args.tsne_perplexity,
        "tsne_seed": args.tsne_seed,
    }

    if args.task in ("classify", "both"):
        if args.classify_mode == "first_last":
            clf_df = prepare_classification_df_first_last(df, patient_col=patient_col)
        elif args.classify_mode == "gls_drop":
            clf_df = prepare_classification_df_gls_drop(
                df,
                patient_col=patient_col,
                gls_threshold=args.gls_threshold,
                slope_threshold=args.gls_slope_threshold,
            )
        else:
            clf_df = prepare_classification_df(
                df, patient_col=patient_col, good_visits=args.good_visits, bad_visits=args.bad_visits
            )
        results["classification"] = {}
        for curve_col in curve_cols:
            visit_idx = None
            if args.sample_level == "view":
                if args.tsne:
                    X3, y, p, visit_idx = build_samples_view(
                        clf_df,
                        curve_col=curve_col,
                        label_col="label",
                        patient_col=patient_col,
                        T=args.T,
                        view_filter=view_filter,
                        return_3d=True,
                        return_meta=True,
                    )
                else:
                    X3, y, p = build_samples_view(
                        clf_df,
                        curve_col=curve_col,
                        label_col="label",
                        patient_col=patient_col,
                        T=args.T,
                        view_filter=view_filter,
                        return_3d=True,
                    )
            else:
                if args.tsne:
                    X3, y, p, visit_idx = build_samples(
                        clf_df,
                        curve_col=curve_col,
                        label_col="label",
                        patient_col=patient_col,
                        T=args.T,
                        view_filter=view_filter,
                        return_3d=True,
                        return_meta=True,
                    )
                else:
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
                tag = (
                    f"{curve_col}_T{args.T}_view{view_filter or 'both'}_sample{args.sample_level}"
                    f"_model{model_type}_taskclass"
                )
                try:
                    if model_type == "linear":
                        metrics = run_classification(
                            X_flat, y, p, visit_idx, outdir=run_dir, tag=tag,
                            test_size=args.test_size, seed=args.seed, model_type=model_type,
                            tsne_enabled=args.tsne, tsne_max_samples=args.tsne_max_samples,
                            tsne_perplexity=args.tsne_perplexity, tsne_seed=args.tsne_seed
                        )
                    else:
                        metrics = run_classification_torch(
                            X3, y, p, visit_idx, outdir=run_dir, tag=tag,
                            test_size=args.test_size, seed=args.seed, model_type=model_type,
                            epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                            logdir=logdir,
                            early_stop_patience=args.early_stop_patience,
                            early_stop_delta=args.early_stop_delta,
                            tsne_enabled=args.tsne, tsne_max_samples=args.tsne_max_samples,
                            tsne_perplexity=args.tsne_perplexity, tsne_seed=args.tsne_seed,
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
            visit_idx = None
            if args.sample_level == "view":
                if args.tsne:
                    X3, y, p, visit_idx = build_samples_view(
                        reg_df,
                        curve_col=curve_col,
                        label_col="days_since_first",
                        patient_col=patient_col,
                        T=args.T,
                        view_filter=view_filter,
                        return_3d=True,
                        return_meta=True,
                    )
                else:
                    X3, y, p = build_samples_view(
                        reg_df,
                        curve_col=curve_col,
                        label_col="days_since_first",
                        patient_col=patient_col,
                        T=args.T,
                        view_filter=view_filter,
                        return_3d=True,
                    )
            else:
                if args.tsne:
                    X3, y, p, visit_idx = build_samples(
                        reg_df,
                        curve_col=curve_col,
                        label_col="days_since_first",
                        patient_col=patient_col,
                        T=args.T,
                        view_filter=view_filter,
                        return_3d=True,
                        return_meta=True,
                    )
                else:
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
                tag = (
                    f"{curve_col}_T{args.T}_view{view_filter or 'both'}_sample{args.sample_level}"
                    f"_model{model_type}_taskreg"
                )
                try:
                    if model_type == "linear":
                        metrics = run_regression(
                            X_flat, y, p, visit_idx, outdir=run_dir, tag=tag,
                            test_size=args.test_size, seed=args.seed, model_type=model_type,
                            tsne_enabled=args.tsne, tsne_max_samples=args.tsne_max_samples,
                            tsne_perplexity=args.tsne_perplexity, tsne_seed=args.tsne_seed
                        )
                    else:
                        metrics = run_regression_torch(
                            X3, y, p, visit_idx, outdir=run_dir, tag=tag,
                            test_size=args.test_size, seed=args.seed, model_type=model_type,
                            epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                            logdir=logdir,
                            early_stop_patience=args.early_stop_patience,
                            early_stop_delta=args.early_stop_delta,
                            tsne_enabled=args.tsne, tsne_max_samples=args.tsne_max_samples,
                            tsne_perplexity=args.tsne_perplexity, tsne_seed=args.tsne_seed,
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
    table_path = run_dir / f"metrics_visit_models_{run_stamp}.csv"
    write_results_table(rows, table_path)
    results["results_table_csv"] = str(table_path)

    out_metrics = run_dir / f"metrics_visit_models_{run_stamp}.json"
    write_json(out_metrics, results)
    print(f"Saved metrics to {out_metrics}")


if __name__ == "__main__":
    main()
