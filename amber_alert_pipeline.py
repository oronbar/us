from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


VIEWS = ["A2C", "A3C", "A4C"]
LAYERS = ["endo", "mid"]
PRIMARY_THRESHOLD = 0.15
SENSITIVITY_THRESHOLDS = [0.125, 0.15, 0.175]
HORIZON_DAYS = 180.0
SEED = 20260719
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    default_out = Path(
        r"C:\Users\oronbarazani\Documents\Codex\2026-07-12\create-a-scheduled-task-called-weekday\outputs"
    )
    p.add_argument("--curves", type=Path, default=default_out / "Ichilov_july_dataset.parquet")
    p.add_argument("--visits", type=Path, default=default_out / "Ichilov_july_visits.parquet")
    p.add_argument("--output-dir", type=Path, default=default_out)
    p.add_argument("--epochs", type=int, default=240)
    p.add_argument("--patience", type=int, default=35)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--cnn-repeats", type=int, default=3)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def normalize_curve(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32)
    scale = float(np.max(np.abs(curve)))
    return curve / scale if scale > 1e-6 else curve


def load_curve_map(path: Path) -> dict[str, np.ndarray]:
    cols = ["visit_id", "curve_family", "layer", "view", "resampled_values"]
    frame = pd.read_parquet(path, columns=cols)
    frame = frame[
        (frame["curve_family"] == "global_strain")
        & frame["layer"].isin(LAYERS)
        & frame["view"].isin(VIEWS)
    ].copy()
    aggregated: dict[tuple[str, str, str], np.ndarray] = {}
    for key, group in frame.groupby(["visit_id", "view", "layer"], sort=False):
        arrays = np.stack(group["resampled_values"].map(lambda x: np.asarray(x, dtype=np.float32)))
        aggregated[key] = arrays.mean(axis=0).astype(np.float32)
    result: dict[str, np.ndarray] = {}
    for visit_id in frame["visit_id"].unique():
        if not all((visit_id, view, layer) in aggregated for view in VIEWS for layer in LAYERS):
            continue
        view_tensors = []
        for view in VIEWS:
            endo = normalize_curve(aggregated[(visit_id, view, "endo")])
            mid = normalize_curve(aggregated[(visit_id, view, "mid")])
            view_tensors.append(np.stack([endo, mid, endo - mid], axis=0))
        result[str(visit_id)] = np.stack(view_tensors, axis=0).astype(np.float32)
    return result


CORE_SCALAR_FEATURES = [
    "current_mid_gls",
    "current_endo_gls",
    "baseline_mid_gls",
    "baseline_endo_gls",
    "current_mid_relative_decline",
    "current_endo_relative_decline",
    "current_ef",
    "previous_mid_gls",
    "previous_endo_gls",
    "previous_ef",
    "mid_slope_per_100d",
    "endo_slope_per_100d",
    "ef_slope_per_100d",
    "time_since_previous_days",
    "visit_order",
    "current_endo_mid_gap",
    "baseline_endo_mid_gap",
    "has_history",
]

ENGINEERED_SOURCE_FEATURES = [
    "endo_peak_abs_std",
    "endo_time_to_peak_norm_std",
    "endo_strain_burden_std",
    "endo_curve_dispersion_rms",
    "endo_shape_dispersion_rms",
    "endo_shape_incoherence",
    "endo_post_systolic_fraction",
    "endo_within_view_peak_robust_sd_mean",
    "endo_spatial_peak_graph_roughness",
    "endo_spatial_timing_graph_roughness",
    "mid_peak_abs_std",
    "mid_time_to_peak_norm_std",
    "mid_strain_burden_std",
    "mid_curve_dispersion_rms",
    "mid_shape_dispersion_rms",
    "mid_shape_incoherence",
    "mid_post_systolic_fraction",
    "mid_within_view_peak_robust_sd_mean",
    "mid_spatial_peak_graph_roughness",
    "mid_spatial_timing_graph_roughness",
]


def safe_float(value: object) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except Exception:
        return np.nan


def select_patient_endpoint(
    ordered: pd.DataFrame,
    layer: str,
    threshold: float,
) -> tuple[int | None, bool, str, int, int]:
    """Return selected index, event flag, status, dropped visits, and raw cutoff order."""
    working = ordered.sort_values("target_visit_order").copy()
    raw_last_order = int(working["target_visit_order"].max())
    # User-defined rescue rule: remove trailing visits whose prediction interval
    # exceeds 180 days, then re-evaluate the shortened timeline.
    while len(working) and not bool(working.iloc[-1]["within_180_days"]):
        working = working.iloc[:-1]
    if not len(working):
        return None, False, "excluded_no_visit_within_180_days", raw_last_order - 1, raw_last_order
    dropped_trailing = raw_last_order - int(working.iloc[-1]["target_visit_order"])

    event_rows = working[working[f"target_{layer}_relative_decline"] > threshold]
    if len(event_rows):
        first_index = int(event_rows.index[0])
        target_order = int(working.loc[first_index, "target_visit_order"])
        if target_order < 3:
            return None, True, "excluded_first_crossing_before_third_visit", dropped_trailing, target_order
        if not bool(working.loc[first_index, "within_180_days"]):
            return None, True, "excluded_first_crossing_interval_gt180", dropped_trailing, target_order
        return first_index, True, "first_crossing", dropped_trailing, target_order

    candidates = working[working["target_visit_order"] >= 3].sort_values(
        "target_visit_order", ascending=False
    )
    for index in candidates.index:
        if bool(working.loc[index, "within_180_days"]):
            index = int(index)
            dropped = raw_last_order - int(working.loc[index, "target_visit_order"])
            status = "last_followup" if dropped == 0 else "recovered_by_dropping_late_visits"
            return index, False, status, dropped, int(working.loc[index, "target_visit_order"])
    return None, False, "excluded_no_qualifying_three_visit_endpoint", dropped_trailing, int(working.iloc[-1]["target_visit_order"])


def build_pairs(visits_path: Path, curve_map: dict[str, np.ndarray]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    visits = pd.read_parquet(visits_path).sort_values(["patient_id", "visit_order"]).copy()
    visits["mid_gls"] = visits["gls_mid_peak_avg"].abs()
    visits["endo_gls"] = visits["gls_endo_peak_avg"].abs()
    records: list[dict[str, object]] = []
    current_curves: list[np.ndarray] = []
    previous_curves: list[np.ndarray] = []
    for patient_id, group in visits.groupby("patient_id", sort=False):
        group = group.sort_values("visit_order").reset_index(drop=True)
        baseline = group.iloc[0]
        baseline_mid = safe_float(baseline["mid_gls"])
        baseline_endo = safe_float(baseline["endo_gls"])
        for index in range(len(group) - 1):
            current = group.iloc[index]
            target = group.iloc[index + 1]
            previous = group.iloc[index - 1] if index > 0 else current
            current_id = str(current["visit_id"])
            previous_id = str(previous["visit_id"])
            if current_id not in curve_map or previous_id not in curve_map:
                continue
            target_gap = safe_float(target["days_since_baseline"] - current["days_since_baseline"])
            previous_gap = safe_float(current["days_since_baseline"] - previous["days_since_baseline"])
            has_history = float(index > 0 and previous_gap > 0)
            current_mid = safe_float(current["mid_gls"])
            current_endo = safe_float(current["endo_gls"])
            previous_mid = safe_float(previous["mid_gls"])
            previous_endo = safe_float(previous["endo_gls"])
            current_ef = safe_float(current["ef_biplane"])
            previous_ef = safe_float(previous["ef_biplane"])
            target_mid = safe_float(target["mid_gls"])
            target_endo = safe_float(target["endo_gls"])
            target_ef = safe_float(target["ef_biplane"])
            record: dict[str, object] = {
                "sample_id": f"{patient_id}|V{int(current['visit_order']):02d}->V{int(target['visit_order']):02d}",
                "patient_id": str(patient_id),
                "current_visit_id": current_id,
                "target_visit_id": str(target["visit_id"]),
                "current_visit_order": int(current["visit_order"]),
                "target_visit_order": int(target["visit_order"]),
                "current_datetime": current["study_datetime"],
                "target_datetime": target["study_datetime"],
                "followup_days": target_gap,
                "within_180_days": bool(0 < target_gap <= HORIZON_DAYS),
                "is_final_transition": bool(index == len(group) - 2),
                "baseline_mid_gls": baseline_mid,
                "baseline_endo_gls": baseline_endo,
                "current_mid_gls": current_mid,
                "current_endo_gls": current_endo,
                "target_mid_gls": target_mid,
                "target_endo_gls": target_endo,
                "current_mid_relative_decline": 1.0 - current_mid / baseline_mid,
                "current_endo_relative_decline": 1.0 - current_endo / baseline_endo,
                "target_mid_relative_decline": 1.0 - target_mid / baseline_mid,
                "target_endo_relative_decline": 1.0 - target_endo / baseline_endo,
                "current_ef": current_ef,
                "previous_mid_gls": previous_mid,
                "previous_endo_gls": previous_endo,
                "previous_ef": previous_ef,
                "target_ef": target_ef,
                "mid_slope_per_100d": ((current_mid - previous_mid) / previous_gap * 100.0) if has_history else 0.0,
                "endo_slope_per_100d": ((current_endo - previous_endo) / previous_gap * 100.0) if has_history else 0.0,
                "ef_slope_per_100d": ((current_ef - previous_ef) / previous_gap * 100.0) if has_history and np.isfinite(current_ef) and np.isfinite(previous_ef) else np.nan,
                "time_since_previous_days": previous_gap if has_history else 0.0,
                "visit_order": int(current["visit_order"]),
                "current_endo_mid_gap": current_endo - current_mid,
                "baseline_endo_mid_gap": baseline_endo - baseline_mid,
                "has_history": has_history,
            }
            for feature in ENGINEERED_SOURCE_FEATURES:
                record[feature] = safe_float(current.get(feature, np.nan))
            for layer in ["mid", "endo"]:
                cur = record[f"current_{layer}_relative_decline"]
                nxt = record[f"target_{layer}_relative_decline"]
                for threshold in SENSITIVITY_THRESHOLDS:
                    suffix = str(threshold).replace(".", "p")
                    record[f"{layer}_eligible_{suffix}"] = bool(cur <= threshold)
                    record[f"{layer}_incident_{suffix}"] = bool(cur <= threshold and nxt > threshold)
                    record[f"{layer}_already_{suffix}"] = bool(cur > threshold)
            records.append(record)
            current_curves.append(curve_map[current_id])
            previous_curves.append(curve_map[previous_id] if has_history else curve_map[current_id])
    pairs = pd.DataFrame(records)
    # Landmark each patient at the first threshold crossing. If no crossing occurs,
    # the selected endpoint is the last available transition. Rows after a first
    # event are excluded from model development for that layer.
    for layer in ["mid", "endo"]:
        pairs[f"{layer}_pre_first_event"] = False
        pairs[f"{layer}_selected_transition"] = False
        pairs[f"{layer}_selected_event"] = False
        pairs[f"{layer}_selection_status"] = ""
        pairs[f"{layer}_dropped_late_visits"] = 0
        for _, patient_group in pairs.groupby("patient_id", sort=False):
            ordered = patient_group.sort_values("target_visit_order")
            selected_index, is_event, status, dropped, cutoff_order = select_patient_endpoint(
                ordered, layer, PRIMARY_THRESHOLD
            )
            before_indices = ordered.index[ordered["target_visit_order"] <= cutoff_order]
            pairs.loc[before_indices, f"{layer}_pre_first_event"] = True
            pairs.loc[ordered.index, f"{layer}_selection_status"] = status
            pairs.loc[ordered.index, f"{layer}_dropped_late_visits"] = dropped
            if selected_index is not None:
                pairs.loc[selected_index, f"{layer}_selected_transition"] = True
                pairs.loc[selected_index, f"{layer}_selected_event"] = is_event
    return pairs, np.stack(current_curves), np.stack(previous_curves)


def label_audit(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer in ["mid", "endo"]:
        for threshold in SENSITIVITY_THRESHOLDS:
            selected_indices = []
            selected_events = []
            statuses = []
            dropped_counts = []
            for _, patient_group in pairs.groupby("patient_id", sort=False):
                ordered = patient_group.sort_values("target_visit_order")
                index, event, status, dropped, _ = select_patient_endpoint(ordered, layer, threshold)
                if index is not None:
                    selected_indices.append(index)
                    selected_events.append(event)
                    statuses.append(status)
                    dropped_counts.append(dropped)
            selected = pairs.loc[selected_indices].copy()
            selected["selected_event_for_threshold"] = selected_events
            selected["selection_status_for_threshold"] = statuses
            selected["dropped_late_visits_for_threshold"] = dropped_counts
            rows.append({
                "subset": "first_event_population",
                "layer": layer,
                "relative_decline_threshold": threshold,
                "n_transitions": len(selected),
                "n_patients": selected["patient_id"].nunique(),
                "eligible_transitions": len(selected),
                "incident_events": int(selected["selected_event_for_threshold"].sum()),
                "incident_rate_among_eligible": float(selected["selected_event_for_threshold"].mean()) if len(selected) else np.nan,
                "already_deteriorated": 0,
                "target_positive": int(selected["selected_event_for_threshold"].sum()),
                "recovered_by_dropping_late_visits": int((selected["dropped_late_visits_for_threshold"] > 0).sum()),
                "excluded_patients": int(pairs["patient_id"].nunique() - selected["patient_id"].nunique()),
            })
    return pd.DataFrame(rows)


def choose_ridge_alpha(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    unique = np.unique(groups)
    splits = min(4, len(unique))
    if splits < 2:
        return 10.0
    cv = GroupKFold(n_splits=splits)
    scores = []
    for alpha in alphas:
        fold_scores = []
        for tr, va in cv.split(x, y, groups):
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ])
            pipe.fit(x[tr], y[tr])
            fold_scores.append(mean_absolute_error(y[va], pipe.predict(x[va])))
        scores.append(np.mean(fold_scores))
    return float(alphas[int(np.argmin(scores))])


def fit_ridge_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target_column: str,
) -> tuple[np.ndarray, float, float]:
    x_train = train[features].to_numpy(float)
    x_test = test[features].to_numpy(float)
    y_train = train[target_column].to_numpy(float)
    groups = train["patient_id"].to_numpy()
    alpha = choose_ridge_alpha(x_train, y_train, groups)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    pipe.fit(x_train, y_train)
    prediction = pipe.predict(x_test)
    residual_scale = max(0.25, float(np.sqrt(np.mean((y_train - pipe.predict(x_train)) ** 2))))
    return prediction, residual_scale, alpha


class ViewEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(3, 12, kernel_size=7, padding=3)
        self.norm1 = nn.GroupNorm(3, 12)
        self.conv2 = nn.Conv1d(12, 16, kernel_size=5, padding=2)
        self.norm2 = nn.GroupNorm(4, 16)
        self.conv3 = nn.Conv1d(16, 16, kernel_size=5, padding=4, dilation=2)
        self.norm3 = nn.GroupNorm(4, 16)
        self.skip = nn.Conv1d(12, 16, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = F.gelu(self.norm1(self.conv1(x)))
        hidden = F.gelu(self.norm2(self.conv2(first)))
        hidden = F.gelu(self.norm3(self.conv3(hidden)) + self.skip(first))
        return torch.cat([hidden.mean(dim=-1), hidden.amax(dim=-1)], dim=1)


class AmberCurveModel(nn.Module):
    def __init__(self, n_scalars: int) -> None:
        super().__init__()
        self.encoder = ViewEncoder()
        self.fusion = nn.Sequential(
            nn.Linear(3 * 32 * 2 + n_scalars, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.gls_mu = nn.Linear(32, 1)
        self.gls_scale = nn.Linear(32, 1)
        self.ef_mu = nn.Linear(32, 1)

    def encode_visit(self, x: torch.Tensor) -> torch.Tensor:
        batch, views, channels, length = x.shape
        encoded = self.encoder(x.reshape(batch * views, channels, length))
        return encoded.reshape(batch, views * encoded.shape[-1])

    def forward(
        self,
        current_curve: torch.Tensor,
        previous_curve: torch.Tensor,
        scalars: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = self.encode_visit(current_curve)
        previous = self.encode_visit(previous_curve)
        trend = (current - previous) * history_mask[:, None]
        hidden = self.fusion(torch.cat([current, trend, scalars], dim=1))
        return (
            self.gls_mu(hidden).squeeze(1),
            F.softplus(self.gls_scale(hidden).squeeze(1)) + 0.05,
            self.ef_mu(hidden).squeeze(1),
        )


def prepare_scalars(train: pd.DataFrame, other: pd.DataFrame, features: list[str]):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train_x = imputer.fit_transform(train[features].to_numpy(float))
    train_x = scaler.fit_transform(train_x)
    other_x = scaler.transform(imputer.transform(other[features].to_numpy(float)))
    return train_x.astype(np.float32), other_x.astype(np.float32), imputer, scaler


def tensor_dataset(
    indices: np.ndarray,
    curve_current: np.ndarray,
    curve_previous: np.ndarray,
    scalars: np.ndarray,
    frame: pd.DataFrame,
    y_mean: float,
    y_std: float,
    ef_mean: float,
    ef_std: float,
    target_column: str,
) -> TensorDataset:
    target_ef = frame.iloc[indices]["target_ef"].to_numpy(float)
    ef_mask = np.isfinite(target_ef).astype(np.float32)
    ef_values = np.nan_to_num((target_ef - ef_mean) / ef_std, nan=0.0).astype(np.float32)
    return TensorDataset(
        torch.from_numpy(curve_current[indices]),
        torch.from_numpy(curve_previous[indices]),
        torch.from_numpy(scalars[indices]),
        torch.from_numpy(frame.iloc[indices]["has_history"].to_numpy(np.float32)),
        torch.from_numpy(((frame.iloc[indices][target_column].to_numpy(float) - y_mean) / y_std).astype(np.float32)),
        torch.from_numpy(ef_values),
        torch.from_numpy(ef_mask),
    )


def train_one_model(
    frame: pd.DataFrame,
    curve_current: np.ndarray,
    curve_previous: np.ndarray,
    outer_train_idx: np.ndarray,
    outer_test_idx: np.ndarray,
    scalar_features: list[str],
    epochs: int,
    patience: int,
    seed: int,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    set_seed(seed)
    train_frame = frame.iloc[outer_train_idx]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=seed)
    sub_rel, val_rel = next(splitter.split(train_frame, groups=train_frame["patient_id"]))
    sub_idx = outer_train_idx[sub_rel]
    val_idx = outer_train_idx[val_rel]

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    all_raw = frame[scalar_features].to_numpy(float)
    sub_x = scaler.fit_transform(imputer.fit_transform(all_raw[sub_idx]))
    all_x = scaler.transform(imputer.transform(all_raw)).astype(np.float32)
    y_train = frame.iloc[sub_idx][target_column].to_numpy(float)
    y_mean = float(np.mean(y_train))
    y_std = max(0.5, float(np.std(y_train, ddof=0)))
    ef_train = frame.iloc[sub_idx]["target_ef"].to_numpy(float)
    ef_mean = float(np.nanmean(ef_train))
    ef_std = max(1.0, float(np.nanstd(ef_train)))

    model = AmberCurveModel(len(scalar_features)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    train_ds = tensor_dataset(sub_idx, curve_current, curve_previous, all_x, frame, y_mean, y_std, ef_mean, ef_std, target_column)
    train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True)
    val_ds = tensor_dataset(val_idx, curve_current, curve_previous, all_x, frame, y_mean, y_std, ef_mean, ef_std, target_column)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    best_state = None
    best_mae = np.inf
    best_epoch = 0
    stale = 0
    df = 4.0
    for epoch in range(epochs):
        model.train()
        for cur, prev, scalar, hist, y, ef, ef_mask in train_loader:
            cur, prev, scalar, hist = cur.to(DEVICE), prev.to(DEVICE), scalar.to(DEVICE), hist.to(DEVICE)
            y, ef, ef_mask = y.to(DEVICE), ef.to(DEVICE), ef_mask.to(DEVICE)
            optimizer.zero_grad()
            mu, scale_z, ef_mu = model(cur, prev, scalar, hist)
            gls_loss = -torch.distributions.StudentT(df=df, loc=mu, scale=scale_z).log_prob(y).mean()
            ef_loss_values = F.smooth_l1_loss(ef_mu, ef, reduction="none") * ef_mask
            ef_loss = ef_loss_values.sum() / ef_mask.sum().clamp_min(1.0)
            loss = gls_loss + 0.12 * ef_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            cur, prev, scalar, hist, y, _, _ = next(iter(val_loader))
            cur, prev, scalar, hist, y = cur.to(DEVICE), prev.to(DEVICE), scalar.to(DEVICE), hist.to(DEVICE), y.to(DEVICE)
            mu, _, _ = model(cur, prev, scalar, hist)
            val_mae = float(torch.mean(torch.abs((mu - y) * y_std)).item())
        if val_mae < best_mae - 1e-4:
            best_mae = val_mae
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        idx = outer_test_idx
        mu_z, scale_z, _ = model(
            torch.from_numpy(curve_current[idx]).to(DEVICE),
            torch.from_numpy(curve_previous[idx]).to(DEVICE),
            torch.from_numpy(all_x[idx]).to(DEVICE),
            torch.from_numpy(frame.iloc[idx]["has_history"].to_numpy(np.float32)).to(DEVICE),
        )
        prediction = mu_z.cpu().numpy() * y_std + y_mean
        scale = scale_z.cpu().numpy() * y_std
        # Calibrate scale using the patient-held validation subset.
        v_mu_z, v_scale_z, _ = model(
            torch.from_numpy(curve_current[val_idx]).to(DEVICE),
            torch.from_numpy(curve_previous[val_idx]).to(DEVICE),
            torch.from_numpy(all_x[val_idx]).to(DEVICE),
            torch.from_numpy(frame.iloc[val_idx]["has_history"].to_numpy(np.float32)).to(DEVICE),
        )
        v_prediction = v_mu_z.cpu().numpy() * y_std + y_mean
        v_scale = v_scale_z.cpu().numpy() * y_std
        v_truth = frame.iloc[val_idx][target_column].to_numpy(float)
        multiplier = float(np.sqrt(np.mean((v_truth - v_prediction) ** 2) / max(np.mean(v_scale ** 2), 1e-6)))
        multiplier = float(np.clip(multiplier, 0.5, 3.0))
        scale = np.maximum(0.25, scale * multiplier)
    metadata = {
        "best_epoch": best_epoch,
        "validation_mae": best_mae,
        "scale_multiplier": multiplier,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "train_patients": int(frame.iloc[sub_idx]["patient_id"].nunique()),
        "validation_patients": int(frame.iloc[val_idx]["patient_id"].nunique()),
        "device": str(DEVICE),
    }
    return prediction, scale, np.full(len(prediction), df), metadata


def amber_probability(mu: np.ndarray, scale: np.ndarray, df: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    threshold = 0.85 * baseline
    return student_t.cdf((threshold - mu) / scale, df=df)


def build_outer_predictions(
    pairs: pd.DataFrame,
    current_curves: np.ndarray,
    previous_curves: np.ndarray,
    folds: int,
    epochs: int,
    patience: int,
    target_layer: str,
    cnn_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_frame = pairs[
        pairs["within_180_days"]
        & pairs[f"{target_layer}_pre_first_event"]
        & (pairs["target_visit_order"] >= 3)
    ].reset_index().rename(columns={"index": "source_index"})
    curve_current = current_curves[model_frame["source_index"].to_numpy(int)]
    curve_previous = previous_curves[model_frame["source_index"].to_numpy(int)]
    y_strat = model_frame[f"{target_layer}_incident_0p15"].astype(int).to_numpy()
    groups = model_frame["patient_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    prediction_rows = []
    cnn_meta_rows = []
    clinical_features = CORE_SCALAR_FEATURES
    engineered_features = CORE_SCALAR_FEATURES + ENGINEERED_SOURCE_FEATURES
    for fold, (train_idx, test_idx) in enumerate(splitter.split(model_frame, y_strat, groups), start=1):
        train = model_frame.iloc[train_idx]
        test = model_frame.iloc[test_idx]
        baseline_column = f"baseline_{target_layer}_gls"
        current_column = f"current_{target_layer}_gls"
        target_column = f"target_{target_layer}_gls"
        current_decline_column = f"current_{target_layer}_relative_decline"
        target_decline_column = f"target_{target_layer}_relative_decline"
        baseline = test[baseline_column].to_numpy(float)
        truth = test[target_column].to_numpy(float)
        models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]] = {}
        persistence_mu = test[current_column].to_numpy(float)
        persistence_resid = train[target_column].to_numpy(float) - train[current_column].to_numpy(float)
        persistence_scale = max(0.25, float(np.sqrt(np.mean(persistence_resid ** 2))))
        models["Persistence"] = (
            persistence_mu,
            np.full(len(test), persistence_scale),
            np.full(len(test), 4.0),
            {},
        )
        for name, features in [("Clinical ridge", clinical_features), ("Engineered ridge", engineered_features)]:
            mu, scale, alpha = fit_ridge_predict(train, test, features, target_column)
            models[name] = (mu, np.full(len(test), scale), np.full(len(test), 4.0), {"alpha": alpha})
        repeat_mu, repeat_scale = [], []
        repeat_metadata = []
        for repeat in range(cnn_repeats):
            cnn_mu, cnn_scale, cnn_df, cnn_meta = train_one_model(
                model_frame,
                curve_current,
                curve_previous,
                train_idx,
                test_idx,
                clinical_features,
                epochs,
                patience,
                SEED + fold + repeat * 1000,
                target_column,
            )
            repeat_mu.append(cnn_mu)
            repeat_scale.append(cnn_scale)
            repeat_metadata.append(cnn_meta)
            cnn_meta_rows.append({"target_layer": target_layer, "fold": fold, "repeat": repeat + 1, **cnn_meta})
        mu_stack = np.stack(repeat_mu)
        scale_stack = np.stack(repeat_scale)
        ensemble_mu = mu_stack.mean(axis=0)
        ensemble_variance = np.maximum(0.25 ** 2, np.mean(scale_stack ** 2 + mu_stack ** 2, axis=0) - ensemble_mu ** 2)
        ensemble_scale = np.sqrt(ensemble_variance)
        ensemble_meta = {
            "cnn_repeats": cnn_repeats,
            "median_best_epoch": float(np.median([m["best_epoch"] for m in repeat_metadata])),
            "parameter_count_per_model": repeat_metadata[0]["parameter_count"],
            "device": str(DEVICE),
        }
        models["Endo–Mid curve CNN"] = (
            ensemble_mu,
            ensemble_scale,
            np.full(len(test), 4.0),
            ensemble_meta,
        )
        for name, (mu, scale, df, metadata) in models.items():
            probability = amber_probability(mu, scale, df, baseline)
            for local, (_, row) in enumerate(test.iterrows()):
                prediction_rows.append({
                    "fold": fold,
                    "target_layer": target_layer,
                    "model": name,
                    "sample_id": row["sample_id"],
                    "patient_id": row["patient_id"],
                    "current_visit_order": row["current_visit_order"],
                    "target_visit_order": row["target_visit_order"],
                    "is_final_transition": row["is_final_transition"],
                    "is_selected_transition": row[f"{target_layer}_selected_transition"],
                    "selection_status": row[f"{target_layer}_selection_status"],
                    "dropped_late_visits": row[f"{target_layer}_dropped_late_visits"],
                    "followup_days": row["followup_days"],
                    "baseline_gls": row[baseline_column],
                    "current_gls": row[current_column],
                    "target_gls": truth[local],
                    "current_relative_decline": row[current_decline_column],
                    "target_relative_decline": row[target_decline_column],
                    "eligible": True,
                    "incident_amber": bool(row[f"target_{target_layer}_relative_decline"] > PRIMARY_THRESHOLD),
                    "predicted_next_gls": mu[local],
                    "prediction_scale": scale[local],
                    "amber_probability": probability[local],
                    "model_metadata": json.dumps(metadata, sort_keys=True),
                })
    return pd.DataFrame(prediction_rows), pd.DataFrame(cnn_meta_rows)


def safe_auc(y: np.ndarray, p: np.ndarray, kind: str) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p) if kind == "ap" else roc_auc_score(y, p))


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subsets = {
        "all_pre_event_180d": np.ones(len(predictions), dtype=bool),
        "selected_first_event_180d": predictions["is_selected_transition"].astype(bool).to_numpy(),
    }
    for (target_layer, model), model_group in predictions.groupby(["target_layer", "model"], sort=False):
        for subset_name in subsets:
            if subset_name == "selected_first_event_180d":
                mask = model_group["is_selected_transition"].astype(bool).to_numpy()
            else:
                mask = np.ones(len(model_group), dtype=bool)
            subset = model_group.iloc[np.flatnonzero(mask)]
            if len(subset) == 0:
                continue
            truth = subset["target_gls"].to_numpy(float)
            mu = subset["predicted_next_gls"].to_numpy(float)
            event = subset["incident_amber"].astype(int).to_numpy()
            probability = subset["amber_probability"].to_numpy(float)
            pred_binary = probability >= 0.5
            top_k = max(1, int(math.ceil(0.20 * len(subset))))
            top_binary = np.zeros(len(subset), dtype=bool)
            top_binary[np.argsort(-probability)[:top_k]] = True
            rows.append({
                "target_layer": target_layer,
                "model": model,
                "subset": subset_name,
                "n_transitions": len(subset),
                "n_patients": subset["patient_id"].nunique(),
                "n_incident_events": int(event.sum()),
                "mae_next_gls": float(mean_absolute_error(truth, mu)),
                "rmse_next_gls": float(np.sqrt(mean_squared_error(truth, mu))),
                "average_precision": safe_auc(event, probability, "ap"),
                "roc_auc": safe_auc(event, probability, "roc"),
                "brier_score": float(brier_score_loss(event, probability)),
                "sensitivity_at_p50": float(recall_score(event, pred_binary, zero_division=0)),
                "precision_at_p50": float(precision_score(event, pred_binary, zero_division=0)),
                "alerts_at_p50": int(pred_binary.sum()),
                "sensitivity_top20pct": float(recall_score(event, top_binary, zero_division=0)),
                "precision_top20pct": float(precision_score(event, top_binary, zero_division=0)),
                "alerts_top20pct": int(top_binary.sum()),
            })
    return pd.DataFrame(rows)


def patient_bootstrap(predictions: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for (target_layer, model), group in predictions.groupby(["target_layer", "model"], sort=False):
        subset = group[group["is_selected_transition"]].copy()
        patients = subset["patient_id"].unique()
        maes, aps, briers = [], [], []
        for _ in range(n_boot):
            sampled = rng.choice(patients, size=len(patients), replace=True)
            chunks = [subset[subset["patient_id"] == p] for p in sampled]
            boot = pd.concat(chunks, ignore_index=True)
            y = boot["incident_amber"].astype(int).to_numpy()
            p = boot["amber_probability"].to_numpy(float)
            maes.append(mean_absolute_error(boot["target_gls"], boot["predicted_next_gls"]))
            if len(np.unique(y)) > 1:
                aps.append(average_precision_score(y, p))
                briers.append(brier_score_loss(y, p))
        rows.append({
            "target_layer": target_layer,
            "model": model,
            "subset": "selected_first_event_180d",
            "mae_ci_low": np.percentile(maes, 2.5),
            "mae_ci_high": np.percentile(maes, 97.5),
            "average_precision_ci_low": np.percentile(aps, 2.5) if aps else np.nan,
            "average_precision_ci_high": np.percentile(aps, 97.5) if aps else np.nan,
            "brier_ci_low": np.percentile(briers, 2.5) if briers else np.nan,
            "brier_ci_high": np.percentile(briers, 97.5) if briers else np.nan,
        })
    return pd.DataFrame(rows)


def make_plots(audit: pd.DataFrame, metrics: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "amber_alert_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    audit_view = audit[(audit["subset"] == "first_event_population")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, layer in zip(axes, ["mid", "endo"]):
        g = audit_view[audit_view["layer"] == layer]
        x = np.arange(len(g))
        ax.bar(x - 0.18, g["incident_events"], width=0.36, label="First deterioration", color="#d9822b")
        ax.bar(x + 0.18, g["n_transitions"] - g["incident_events"], width=0.36, label="No deterioration", color="#4c78a8")
        ax.set_xticks(x, [f"{100*t:.1f}%" for t in g["relative_decline_threshold"]])
        ax.set_title(f"{layer.capitalize()} GLS")
        ax.set_xlabel("Relative decline threshold")
        ax.set_ylabel("Final transitions")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("First-event population after the 180-day and three-visit rules")
    fig.tight_layout()
    fig.savefig(fig_dir / "amber_label_audit.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    primary = metrics[(metrics["target_layer"] == "mid") & (metrics["subset"] == "selected_first_event_180d")].copy()
    order = ["Persistence", "Clinical ridge", "Engineered ridge", "Endo–Mid curve CNN"]
    primary["order"] = primary["model"].map({m: i for i, m in enumerate(order)})
    primary = primary.sort_values("order")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].barh(primary["model"], primary["mae_next_gls"], color="#4c78a8")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MAE, strain points (lower is better)")
    axes[0].set_title("Next Mid GLS")
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].barh(primary["model"], primary["average_precision"], color="#d9822b")
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Average precision (higher is better)")
    axes[1].set_title("Incident amber alert")
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle("Patient-held-out performance on selected first-event transitions")
    fig.tight_layout()
    fig.savefig(fig_dir / "amber_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    cnn = predictions[(predictions["target_layer"] == "mid") & (predictions["model"] == "Endo–Mid curve CNN") & predictions["is_selected_transition"]]
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    colors = np.where(cnn["incident_amber"], "#d62728", "#4c78a8")
    ax.scatter(cnn["target_gls"], cnn["predicted_next_gls"], c=colors, alpha=0.85)
    low = min(cnn["target_gls"].min(), cnn["predicted_next_gls"].min()) - 0.5
    high = max(cnn["target_gls"].max(), cnn["predicted_next_gls"].max()) + 0.5
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_xlim(low, high); ax.set_ylim(low, high)
    ax.set_xlabel("Observed next Mid GLS magnitude")
    ax.set_ylabel("Predicted next Mid GLS magnitude")
    ax.set_title("Curve CNN: selected first-event predictions")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "amber_cnn_observed_vs_predicted.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame.loc[:, columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append("" if not np.isfinite(value) else f"{value:.3f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    pairs: pd.DataFrame,
    audit: pd.DataFrame,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    cnn_meta: pd.DataFrame,
) -> None:
    final_audit = audit[
        (audit["subset"] == "first_event_population")
        & (audit["relative_decline_threshold"] == PRIMARY_THRESHOLD)
    ]
    final_metrics = metrics[(metrics["target_layer"] == "mid") & (metrics["subset"] == "selected_first_event_180d")].copy()
    final_metrics = final_metrics.merge(bootstrap, on=["target_layer", "model", "subset"], how="left")
    endo_metrics = metrics[(metrics["target_layer"] == "endo") & (metrics["subset"] == "selected_first_event_180d")].copy()
    endo_metrics = endo_metrics.merge(bootstrap, on=["target_layer", "model", "subset"], how="left")
    best_mae = final_metrics.sort_values("mae_next_gls").iloc[0]
    best_ap = final_metrics.sort_values("average_precision", ascending=False).iloc[0]
    cnn = final_metrics[final_metrics["model"] == "Endo–Mid curve CNN"].iloc[0]
    clinical = final_metrics[final_metrics["model"] == "Clinical ridge"].iloc[0]
    interpretation = (
        "The curve CNN improved both next-GLS error and incident-alert ranking over the clinical ridge."
        if cnn["mae_next_gls"] < clinical["mae_next_gls"] and cnn["average_precision"] > clinical["average_precision"]
        else "The curve CNN did not consistently improve both forecasting error and incident-alert ranking over the clinical ridge."
    )
    report = [
        "# Amber deterioration alert: execution report",
        "",
        "## Bottom line",
        "",
        interpretation,
        "",
        f"The best selected-transition GLS MAE was {best_mae['mae_next_gls']:.2f} strain points ({best_mae['model']}). "
        f"The best incident-alert average precision was {best_ap['average_precision']:.2f} ({best_ap['model']}).",
        "",
        "All performance estimates are exploratory because the independent sample contains only 33 patients and few incident events.",
        "",
        "## Frozen endpoint",
        "",
        "Primary layer: Mid GLS. Each patient is truncated at the first >15% relative reduction from baseline; if no event occurs, the latest qualifying follow-up is used. "
        "The selected timeline must contain at least three visits and the prediction-to-outcome interval must be ≤180 days. For non-events with a late final interval, visits are dropped from the end until the latest qualifying endpoint is found.",
        "",
        "## Label audit",
        "",
        markdown_table(final_audit, ["layer", "n_transitions", "incident_events", "incident_rate_among_eligible", "recovered_by_dropping_late_visits", "excluded_patients"]),
        "",
        "## Patient-held-out selected-transition performance",
        "",
        markdown_table(final_metrics, ["model", "n_transitions", "n_incident_events", "mae_next_gls", "mae_ci_low", "mae_ci_high", "average_precision", "average_precision_ci_low", "average_precision_ci_high", "brier_score", "sensitivity_at_p50", "precision_at_p50", "alerts_at_p50"]),
        "",
        "At a descriptive 20% alert budget, the following table shows how many first events were captured among the highest-risk selected patients:",
        "",
        markdown_table(final_metrics, ["model", "sensitivity_top20pct", "precision_top20pct", "alerts_top20pct"]),
        "",
        "## Endo-target sensitivity analysis",
        "",
        markdown_table(endo_metrics, ["model", "n_transitions", "n_incident_events", "mae_next_gls", "average_precision", "brier_score", "sensitivity_top20pct", "precision_top20pct"]),
        "",
        "## Model implementation",
        "",
        f"The Endo–Mid model used a shared three-view 1D CNN with current and previous curve embeddings, explicit trajectory differences, and a scalar branch. "
        f"Each CNN contained {int(cnn_meta['parameter_count'].median())} trainable parameters. Three seeded CNN fits were averaged within each fold to reduce small-sample training instability. A Student-t head predicted next GLS and uncertainty; a masked auxiliary head predicted EF.",
        "",
        "## Interpretation rule",
        "",
        "A complex curve model should be retained only if it improves out-of-patient next-GLS error and alert ranking over the clinical trajectory model. "
        "Threshold-at-0.5 sensitivity and precision are shown descriptively; the operating probability cutoff must be chosen in a larger development cohort.",
    ]
    (output_dir / "amber_alert_execution_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    curve_map = load_curve_map(args.curves.resolve())
    pairs, current_curves, previous_curves = build_pairs(args.visits.resolve(), curve_map)
    audit = label_audit(pairs)
    prediction_frames = []
    metadata_frames = []
    for target_layer in ["mid", "endo"]:
        layer_predictions, layer_metadata = build_outer_predictions(
            pairs, current_curves, previous_curves, args.folds, args.epochs, args.patience, target_layer, args.cnn_repeats
        )
        prediction_frames.append(layer_predictions)
        metadata_frames.append(layer_metadata)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    cnn_meta = pd.concat(metadata_frames, ignore_index=True)
    metrics = evaluate_predictions(predictions)
    bootstrap = patient_bootstrap(predictions)
    make_plots(audit, metrics, predictions, output_dir)
    write_report(output_dir, pairs, audit, metrics, bootstrap, cnn_meta)

    pairs.to_parquet(output_dir / "amber_alert_pairs.parquet", index=False, compression="zstd")
    pairs.to_csv(output_dir / "amber_alert_pairs.csv", index=False)
    audit.to_csv(output_dir / "amber_alert_label_audit.csv", index=False)
    predictions.to_parquet(output_dir / "amber_alert_oof_predictions.parquet", index=False, compression="zstd")
    predictions.to_csv(output_dir / "amber_alert_oof_predictions.csv", index=False)
    metrics.to_csv(output_dir / "amber_alert_model_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "amber_alert_bootstrap_intervals.csv", index=False)
    cnn_meta.to_csv(output_dir / "amber_alert_cnn_fold_metadata.csv", index=False)
    spec = {
        "primary_layer": "mid",
        "sensitivity_layer": "endo",
        "relative_decline_threshold": PRIMARY_THRESHOLD,
        "horizon_days": HORIZON_DAYS,
        "target": "next_mid_gls",
        "alert_probability": "P(next_mid_gls < 0.85 * baseline_mid_gls)",
        "population_definition": "first >15% relative GLS decline; otherwise latest qualifying non-event follow-up",
        "minimum_selected_timeline_visits": 3,
        "late_non_event_rule": "drop visits from the end until the latest prediction-to-outcome interval <=180 days",
        "late_first_event_rule": "exclude if the first-crossing interval exceeds 180 days; do not relabel an earlier non-event",
        "split": "5-fold stratified patient-group cross-validation",
        "curve_input": "3 views x [normalized Endo, normalized Mid, Endo-Mid] x 96 points; current and previous visit",
        "scalar_features": CORE_SCALAR_FEATURES,
        "models": ["Persistence", "Clinical ridge", "Engineered ridge", "Endo-Mid curve CNN"],
        "seed": SEED,
        "training_device": str(DEVICE),
        "cnn_repeats": args.cnn_repeats,
        "patients": int(pairs["patient_id"].nunique()),
        "all_transitions": int(len(pairs)),
        "transitions_within_180_days": int(pairs["within_180_days"].sum()),
    }
    (output_dir / "amber_alert_model_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(output_dir / "amber_alert_execution_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
