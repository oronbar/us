from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from torch import nn

from cardiotoxicity_early_detection import VARIABILITY_SUFFIXES, curve_feature_table, sha256_file


DEFAULT_INPUT = Path(r"D:\us\amber_full_105_preprocessed")
DEFAULT_RAW = Path(r"D:\DS\anonymized_reports")
DEFAULT_OUTPUT = Path(r"D:\us\cardiotoxicity_next_visit_gpu_results")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    metric: str
    baseline: str
    threshold: float
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Next-visit cardiotoxicity alert study with GPU segment curves.")
    parser.add_argument("--visits", type=Path, default=DEFAULT_INPUT / "Ichilov_july_visits.parquet")
    parser.add_argument("--curves", type=Path, default=DEFAULT_INPUT / "Ichilov_july_dataset.parquet")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def task_specs() -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for baseline in ["first", "roll2", "roll3"]:
        for threshold in [0.10, 0.12, 0.15, 0.20]:
            pct = int(round(threshold * 100))
            baseline_text = {
                "first": "first visit",
                "roll2": "mean of the last 2 observed visits",
                "roll3": "mean of the last 3 observed visits",
            }[baseline]
            tasks.append(
                TaskSpec(
                    name=f"mid_{baseline}_rel{pct}",
                    metric="mid",
                    baseline=baseline,
                    threshold=threshold,
                    description=f"Next Mid-GLS magnitude is >={pct}% below {baseline_text}",
                )
            )
    for baseline in ["first", "roll2", "roll3"]:
        tasks.append(
            TaskSpec(
                name=f"endo_{baseline}_rel15",
                metric="endo",
                baseline=baseline,
                threshold=0.15,
                description=f"Next Endo-GLS magnitude is >=15% below {baseline}",
            )
        )
    for baseline in ["first", "roll2", "roll3"]:
        for threshold in [0.10, 0.15]:
            pct = int(round(threshold * 100))
            tasks.append(
                TaskSpec(
                    name=f"ef_{baseline}_rel{pct}",
                    metric="ef",
                    baseline=baseline,
                    threshold=threshold,
                    description=f"Next biplane EF is >={pct}% below {baseline}",
                )
            )
    return tasks


def safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def metric_value(row: pd.Series, metric: str) -> float:
    if metric == "mid":
        return abs(safe_float(row["gls_mid_peak_avg"]))
    if metric == "endo":
        return abs(safe_float(row["gls_endo_peak_avg"]))
    return safe_float(row["ef_biplane"])


def build_visit_curve_tensors(curves_path: Path) -> dict[str, np.ndarray]:
    columns = ["visit_id", "curve_family", "layer", "segment_number", "resampled_values"]
    frame = pd.read_parquet(curves_path, columns=columns)
    frame = frame[
        (frame["curve_family"] == "longitudinal_strain")
        & frame["layer"].isin(["endo", "mid"])
        & frame["segment_number"].notna()
    ].copy()
    aggregated: dict[tuple[str, str, int], list[np.ndarray]] = {}
    for row in frame.itertuples(index=False):
        key = (str(row.visit_id), str(row.layer), int(row.segment_number))
        aggregated.setdefault(key, []).append(np.asarray(row.resampled_values, dtype=np.float32))
    result: dict[str, np.ndarray] = {}
    for visit_id in frame["visit_id"].astype(str).unique():
        tensor = np.full((18, 2, 96), np.nan, dtype=np.float32)
        complete = True
        for segment in range(1, 19):
            for layer_index, layer in enumerate(["endo", "mid"]):
                curves = aggregated.get((visit_id, layer, segment), [])
                if not curves:
                    complete = False
                    continue
                tensor[segment - 1, layer_index] = np.nanmean(np.stack(curves), axis=0)
        if complete and np.isfinite(tensor).all():
            result[visit_id] = tensor
    return result


def build_transitions(
    visits: pd.DataFrame,
    transmural_visits: pd.DataFrame,
    curve_tensors: dict[str, np.ndarray],
    tasks: list[TaskSpec],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, list[str]]]:
    visits = visits.sort_values(["patient_id", "visit_order"]).copy()
    tm = transmural_visits.set_index("visit_id")
    tm_columns = [column for column in tm.columns if column.startswith("tm_")]
    variability_columns = [
        f"{layer}_{suffix}"
        for layer in ["endo", "mid"]
        for suffix in VARIABILITY_SUFFIXES
        if f"{layer}_{suffix}" in visits.columns
    ]

    rows: list[dict[str, object]] = []
    tensors: list[np.ndarray] = []
    for patient_id, patient in visits.groupby("patient_id", sort=False):
        patient = patient.sort_values("visit_order").reset_index(drop=True)
        first = patient.iloc[0]
        first_values = {metric: metric_value(first, metric) for metric in ["mid", "endo", "ef"]}
        patient_row_indices: list[int] = []

        for current_index in range(len(patient) - 1):
            current = patient.iloc[current_index]
            target = patient.iloc[current_index + 1]
            previous = patient.iloc[max(0, current_index - 1)]
            current_id = str(current["visit_id"])
            previous_id = str(previous["visit_id"])
            if current_id not in curve_tensors or previous_id not in curve_tensors:
                continue

            current_values = {metric: metric_value(current, metric) for metric in ["mid", "endo", "ef"]}
            previous_values = {metric: metric_value(previous, metric) for metric in ["mid", "endo", "ef"]}
            dt = max(safe_float(current["days_since_baseline"]) - safe_float(previous["days_since_baseline"]), 1.0)
            row: dict[str, object] = {
                "transition_id": f"{patient_id}|V{int(current['visit_order']):02d}->V{int(target['visit_order']):02d}",
                "patient_id": str(patient_id),
                "current_visit_id": current_id,
                "target_visit_id": str(target["visit_id"]),
                "current_visit_order": int(current["visit_order"]),
                "target_visit_order": int(target["visit_order"]),
                "history_visits": int(current_index + 1),
                "days_since_first": safe_float(current["days_since_baseline"]),
                "days_since_previous": 0.0 if current_index == 0 else dt,
                "has_previous_visit": int(current_index > 0),
                "current_mid_gls": current_values["mid"],
                "current_endo_gls": current_values["endo"],
                "current_ef": current_values["ef"],
                "first_mid_gls": first_values["mid"],
                "first_endo_gls": first_values["endo"],
                "first_ef": first_values["ef"],
                "current_mid_decline_from_first": 1.0 - current_values["mid"] / first_values["mid"],
                "current_endo_decline_from_first": 1.0 - current_values["endo"] / first_values["endo"],
                "current_ef_decline_from_first": 1.0 - current_values["ef"] / first_values["ef"] if np.isfinite(current_values["ef"]) and np.isfinite(first_values["ef"]) else np.nan,
                "last_mid_relative_change": 0.0 if current_index == 0 else 1.0 - current_values["mid"] / previous_values["mid"],
                "last_endo_relative_change": 0.0 if current_index == 0 else 1.0 - current_values["endo"] / previous_values["endo"],
                "last_ef_relative_change": 0.0 if current_index == 0 or not np.isfinite(current_values["ef"]) or not np.isfinite(previous_values["ef"]) else 1.0 - current_values["ef"] / previous_values["ef"],
                "mid_decline_slope_per_100d": 0.0 if current_index == 0 else (1.0 - current_values["mid"] / previous_values["mid"]) * 100.0 / dt,
                "endo_decline_slope_per_100d": 0.0 if current_index == 0 else (1.0 - current_values["endo"] / previous_values["endo"]) * 100.0 / dt,
                "ef_decline_slope_per_100d": 0.0 if current_index == 0 or not np.isfinite(current_values["ef"]) or not np.isfinite(previous_values["ef"]) else (1.0 - current_values["ef"] / previous_values["ef"]) * 100.0 / dt,
                "current_endo_mid_gap": current_values["endo"] - current_values["mid"],
                "last_endo_mid_gap_change": 0.0 if current_index == 0 else (current_values["endo"] - current_values["mid"]) - (previous_values["endo"] - previous_values["mid"]),
            }

            for n in [2, 3]:
                if current_index + 1 >= n:
                    history = patient.iloc[current_index - n + 1 : current_index + 1]
                    for metric in ["mid", "endo", "ef"]:
                        values = np.asarray([metric_value(r, metric) for _, r in history.iterrows()], dtype=float)
                        baseline = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
                        row[f"current_{metric}_decline_from_roll{n}"] = 1.0 - current_values[metric] / baseline if np.isfinite(current_values[metric]) and np.isfinite(baseline) else np.nan
                else:
                    for metric in ["mid", "endo", "ef"]:
                        row[f"current_{metric}_decline_from_roll{n}"] = np.nan

            for column in variability_columns:
                current_value = safe_float(current[column])
                previous_value = safe_float(previous[column])
                row[f"cur_var__{column}"] = current_value
                row[f"d_var__{column}"] = 0.0 if current_index == 0 else current_value - previous_value

            for column in tm_columns:
                current_value = tm.loc[current_id, column]
                previous_value = tm.loc[previous_id, column]
                if isinstance(current_value, pd.Series):
                    current_value = current_value.mean()
                if isinstance(previous_value, pd.Series):
                    previous_value = previous_value.mean()
                current_value = safe_float(current_value)
                previous_value = safe_float(previous_value)
                row[f"cur_{column}"] = current_value
                row[f"d_{column}"] = 0.0 if current_index == 0 else current_value - previous_value

            current_tensor = curve_tensors[current_id]
            previous_tensor = curve_tensors[previous_id]
            endo = current_tensor[:, 0]
            mid = current_tensor[:, 1]
            delta_endo = endo - previous_tensor[:, 0] if current_index > 0 else np.zeros_like(endo)
            delta_mid = mid - previous_tensor[:, 1] if current_index > 0 else np.zeros_like(mid)
            tensor = np.stack(
                [endo, mid, endo - mid, delta_endo, delta_mid, delta_endo - delta_mid], axis=1
            ).astype(np.float32)
            tensor = np.clip(tensor / 30.0, -2.0, 2.0)
            patient_row_indices.append(len(rows))
            rows.append(row)
            tensors.append(tensor)

        # Labels are added after all rows for this patient are known.
        for spec in tasks:
            valid_rows: list[tuple[int, int, float]] = []
            for local_transition, global_index in enumerate(patient_row_indices):
                target_index = local_transition + 1
                if target_index >= len(patient):
                    continue
                target_value = metric_value(patient.iloc[target_index], spec.metric)
                if spec.baseline == "first":
                    baseline_value = first_values[spec.metric]
                else:
                    n = int(spec.baseline[-1])
                    if target_index < n:
                        continue
                    history = patient.iloc[target_index - n : target_index]
                    history_values = np.asarray(
                        [metric_value(r, spec.metric) for _, r in history.iterrows()], dtype=float
                    )
                    baseline_value = float(np.nanmean(history_values)) if np.isfinite(history_values).all() else np.nan
                if not np.isfinite(target_value) or not np.isfinite(baseline_value) or baseline_value == 0:
                    continue
                decline = 1.0 - target_value / baseline_value
                valid_rows.append((global_index, local_transition, decline))

            if spec.baseline == "first":
                crossings = [item for item in valid_rows if item[2] >= spec.threshold]
                crossing_local = crossings[0][1] if crossings else None
                for global_index, local_transition, decline in valid_rows:
                    eligible = crossing_local is None or local_transition <= crossing_local
                    rows[global_index][f"mask__{spec.name}"] = bool(eligible)
                    rows[global_index][f"label__{spec.name}"] = int(
                        crossing_local is not None and local_transition == crossing_local
                    )
                    rows[global_index][f"decline__{spec.name}"] = decline
            else:
                valid_index_set = {item[0] for item in valid_rows}
                for global_index in patient_row_indices:
                    rows[global_index][f"mask__{spec.name}"] = global_index in valid_index_set
                    rows[global_index][f"label__{spec.name}"] = 0
                    rows[global_index][f"decline__{spec.name}"] = np.nan
                for global_index, _, decline in valid_rows:
                    rows[global_index][f"label__{spec.name}"] = int(decline >= spec.threshold)
                    rows[global_index][f"decline__{spec.name}"] = decline

        # Ensure every label/mask exists for rows with missing EF or unavailable history.
        for global_index in patient_row_indices:
            for spec in tasks:
                rows[global_index].setdefault(f"mask__{spec.name}", False)
                rows[global_index].setdefault(f"label__{spec.name}", 0)
                rows[global_index].setdefault(f"decline__{spec.name}", np.nan)

    transitions = pd.DataFrame(rows)
    clinical = [
        "history_visits",
        "days_since_first",
        "days_since_previous",
        "has_previous_visit",
        "current_mid_gls",
        "current_endo_gls",
        "current_ef",
        "first_mid_gls",
        "first_endo_gls",
        "first_ef",
        "current_mid_decline_from_first",
        "current_endo_decline_from_first",
        "current_ef_decline_from_first",
        "last_mid_relative_change",
        "last_endo_relative_change",
        "last_ef_relative_change",
        "mid_decline_slope_per_100d",
        "endo_decline_slope_per_100d",
        "ef_decline_slope_per_100d",
        "current_endo_mid_gap",
        "last_endo_mid_gap_change",
        "current_mid_decline_from_roll2",
        "current_endo_decline_from_roll2",
        "current_ef_decline_from_roll2",
        "current_mid_decline_from_roll3",
        "current_endo_decline_from_roll3",
        "current_ef_decline_from_roll3",
    ]
    transmural = [column for column in transitions.columns if column.startswith(("cur_tm_", "d_tm_"))]
    variability = [column for column in transitions.columns if column.startswith(("cur_var__", "d_var__"))]
    feature_sets = {
        "clinical": clinical,
        "clinical_plus_transmural": clinical + transmural,
        "clinical_plus_variability": clinical + variability,
        "combined": clinical + transmural + variability,
        "gpu_scalars": clinical + variability,
    }
    return transitions, np.stack(tensors), feature_sets


def label_audit(transitions: pd.DataFrame, tasks: list[TaskSpec]) -> pd.DataFrame:
    rows = []
    for spec in tasks:
        mask = transitions[f"mask__{spec.name}"].astype(bool)
        labels = transitions.loc[mask, f"label__{spec.name}"].astype(int)
        rows.append(
            {
                "task": spec.name,
                "metric": spec.metric,
                "baseline": spec.baseline,
                "relative_threshold": spec.threshold,
                "eligible_transitions": int(mask.sum()),
                "events": int(labels.sum()),
                "event_rate": float(labels.mean()) if len(labels) else np.nan,
                "event_patients": int(transitions.loc[mask & (transitions[f"label__{spec.name}"] == 1), "patient_id"].nunique()),
                "description": spec.description,
            }
        )
    return pd.DataFrame(rows)


def patient_splits(
    transitions: pd.DataFrame, repeats: int, seed: int
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    patient_table = pd.DataFrame({"patient_id": sorted(transitions["patient_id"].unique())})
    primary = transitions[
        transitions["mask__mid_first_rel15"].astype(bool)
        & (transitions["label__mid_first_rel15"] == 1)
    ]["patient_id"].unique()
    patient_table["primary_event"] = patient_table["patient_id"].isin(primary).astype(int)
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    splits: list[dict[str, object]] = []
    assignments = []
    for split_index, (train, test) in enumerate(
        splitter.split(patient_table["patient_id"], patient_table["primary_event"])
    ):
        repeat = split_index // 5
        fold = split_index % 5
        train_patients = set(patient_table.iloc[train]["patient_id"])
        test_patients = set(patient_table.iloc[test]["patient_id"])
        splits.append(
            {
                "split_index": split_index,
                "repeat": repeat,
                "fold": fold,
                "train_patients": train_patients,
                "test_patients": test_patients,
            }
        )
        for patient_id in test_patients:
            assignments.append(
                {"repeat": repeat, "fold": fold, "patient_id": patient_id, "role": "test"}
            )
    return splits, pd.DataFrame(assignments)


def robust_linear_pipeline(penalty: str, seed: int) -> Pipeline:
    model = LogisticRegression(
        penalty=penalty,
        C=0.3,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
            (
                "clip",
                FunctionTransformer(
                    np.clip,
                    kw_args={"a_min": -5.0, "a_max": 5.0},
                    feature_names_out="one-to-one",
                ),
            ),
            ("model", model),
        ]
    )


def usable_features(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    result = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().mean() >= 0.55 and values.nunique(dropna=True) > 1:
            result.append(column)
    return result


def cpu_oof_predictions(
    transitions: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    active_tasks: list[TaskSpec],
    splits: list[dict[str, object]],
    seed: int,
) -> pd.DataFrame:
    model_specs = [
        ("clinical_ridge", "clinical", "l2"),
        ("clinical_plus_transmural_sparse", "clinical_plus_transmural", "l1"),
        ("clinical_plus_variability_sparse", "clinical_plus_variability", "l1"),
        ("combined_extra_trees", "combined", "trees"),
    ]
    accumulators: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for spec in active_tasks:
        for model_name, _, _ in model_specs:
            accumulators[(spec.name, model_name)] = (
                np.zeros(len(transitions), dtype=float),
                np.zeros(len(transitions), dtype=int),
            )

    for split in splits:
        train_patient = transitions["patient_id"].isin(split["train_patients"]).to_numpy()
        test_patient = transitions["patient_id"].isin(split["test_patients"]).to_numpy()
        for task_index, spec in enumerate(active_tasks):
            task_mask = transitions[f"mask__{spec.name}"].astype(bool).to_numpy()
            train_index = np.flatnonzero(train_patient & task_mask)
            test_index = np.flatnonzero(test_patient & task_mask)
            y_train = transitions.iloc[train_index][f"label__{spec.name}"].to_numpy(int)
            if len(test_index) == 0 or np.unique(y_train).size < 2:
                continue
            for model_index, (model_name, feature_set, kind) in enumerate(model_specs):
                columns = usable_features(transitions.iloc[train_index], feature_sets[feature_set])
                if kind == "trees":
                    model: object = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "model",
                                ExtraTreesClassifier(
                                    n_estimators=300,
                                    min_samples_leaf=5,
                                    max_features=0.5,
                                    class_weight="balanced_subsample",
                                    n_jobs=-1,
                                    random_state=seed + int(split["split_index"]) * 100 + task_index * 10 + model_index,
                                ),
                            ),
                        ]
                    )
                else:
                    model = robust_linear_pipeline(
                        kind, seed + int(split["split_index"]) * 100 + task_index * 10 + model_index
                    )
                model.fit(transitions.iloc[train_index][columns].astype(float), y_train)
                score = model.predict_proba(transitions.iloc[test_index][columns].astype(float))[:, 1]
                score_sum, score_count = accumulators[(spec.name, model_name)]
                score_sum[test_index] += score
                score_count[test_index] += 1

    rows: list[dict[str, object]] = []
    for spec in active_tasks:
        mask = transitions[f"mask__{spec.name}"].astype(bool).to_numpy()
        for model_name, _, _ in model_specs:
            score_sum, score_count = accumulators[(spec.name, model_name)]
            valid = mask & (score_count > 0)
            for index in np.flatnonzero(valid):
                rows.append(
                    {
                        "task": spec.name,
                        "model": model_name,
                        "transition_id": transitions.iloc[index]["transition_id"],
                        "patient_id": transitions.iloc[index]["patient_id"],
                        "label": int(transitions.iloc[index][f"label__{spec.name}"]),
                        "score": float(score_sum[index] / score_count[index]),
                        "prediction_repeats": int(score_count[index]),
                    }
                )
    return pd.DataFrame(rows)


class SegmentCurveAlertNet(nn.Module):
    def __init__(self, scalar_features: int, tasks: int):
        super().__init__()
        self.segment_encoder = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 24, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_features, 48),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(48, 32),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(24 * 3 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(64, tasks),
        )

    def forward(self, curves: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        batch, segments, channels, time_points = curves.shape
        encoded = self.segment_encoder(curves.reshape(batch * segments, channels, time_points)).squeeze(-1)
        encoded = encoded.reshape(batch, segments, -1)
        pooled = torch.cat(
            [encoded.mean(dim=1), encoded.std(dim=1), encoded.max(dim=1).values], dim=1
        )
        scalar_embedding = self.scalar_encoder(scalars)
        return self.head(torch.cat([pooled, scalar_embedding], dim=1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight, reduction="none"
    )
    return (loss * masks).sum() / masks.sum().clamp_min(1.0)


def gpu_oof_predictions(
    transitions: pd.DataFrame,
    curve_inputs: np.ndarray,
    feature_sets: dict[str, list[str]],
    active_tasks: list[TaskSpec],
    splits: list[dict[str, object]],
    epochs: int,
    patience: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested GPU model but is not available.")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    scalar_columns = usable_features(transitions, feature_sets["gpu_scalars"])
    labels = np.column_stack(
        [transitions[f"label__{spec.name}"].to_numpy(np.float32) for spec in active_tasks]
    )
    masks = np.column_stack(
        [transitions[f"mask__{spec.name}"].to_numpy(np.float32) for spec in active_tasks]
    )
    score_sum = np.zeros((len(transitions), len(active_tasks)), dtype=float)
    score_count = np.zeros((len(transitions), len(active_tasks)), dtype=int)
    log_rows = []
    parameter_count = None

    patient_primary = (
        transitions.groupby("patient_id")["label__mid_first_rel15"].max().astype(int).to_dict()
    )
    for split in splits:
        split_seed = seed + int(split["split_index"]) * 101
        set_seed(split_seed)
        train_patients = sorted(split["train_patients"])
        patient_labels = [patient_primary.get(patient, 0) for patient in train_patients]
        try:
            fit_patients, val_patients = train_test_split(
                train_patients,
                test_size=0.2,
                random_state=split_seed,
                stratify=patient_labels,
            )
        except ValueError:
            fit_patients, val_patients = train_test_split(
                train_patients, test_size=0.2, random_state=split_seed
            )
        fit_index = np.flatnonzero(transitions["patient_id"].isin(fit_patients).to_numpy())
        val_index = np.flatnonzero(transitions["patient_id"].isin(val_patients).to_numpy())
        test_index = np.flatnonzero(transitions["patient_id"].isin(split["test_patients"]).to_numpy())

        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        fit_scalars = imputer.fit_transform(transitions.iloc[fit_index][scalar_columns].astype(float))
        fit_scalars = np.clip(scaler.fit_transform(fit_scalars), -5.0, 5.0).astype(np.float32)
        val_scalars = np.clip(
            scaler.transform(imputer.transform(transitions.iloc[val_index][scalar_columns].astype(float))),
            -5.0,
            5.0,
        ).astype(np.float32)
        test_scalars = np.clip(
            scaler.transform(imputer.transform(transitions.iloc[test_index][scalar_columns].astype(float))),
            -5.0,
            5.0,
        ).astype(np.float32)

        fit_labels = labels[fit_index]
        fit_masks = masks[fit_index].copy()
        for task_index in range(len(active_tasks)):
            task_y = fit_labels[fit_masks[:, task_index] > 0, task_index]
            if task_y.sum() < 2 or (len(task_y) - task_y.sum()) < 2:
                fit_masks[:, task_index] = 0
        positives = (fit_labels * fit_masks).sum(axis=0)
        negatives = ((1.0 - fit_labels) * fit_masks).sum(axis=0)
        weights = np.clip(negatives / np.maximum(positives, 1.0), 1.0, 20.0).astype(np.float32)

        model = SegmentCurveAlertNet(len(scalar_columns), len(active_tasks)).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-3)
        amp_scaler = torch.amp.GradScaler("cuda", enabled=True)
        best_state = copy.deepcopy(model.state_dict())
        best_loss = math.inf
        best_epoch = 0
        stale = 0
        start = time.time()
        batch_size = 64

        for epoch in range(epochs):
            model.train()
            order = np.random.default_rng(split_seed + epoch).permutation(len(fit_index))
            for start_index in range(0, len(order), batch_size):
                local = order[start_index : start_index + batch_size]
                global_index = fit_index[local]
                batch_curve = torch.as_tensor(curve_inputs[global_index], device=device)
                batch_scalar = torch.as_tensor(fit_scalars[local], device=device)
                batch_label = torch.as_tensor(fit_labels[local], device=device)
                batch_mask = torch.as_tensor(fit_masks[local], device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    logits = model(batch_curve, batch_scalar)
                    loss = masked_loss(
                        logits,
                        batch_label,
                        batch_mask,
                        torch.as_tensor(weights, device=device),
                    )
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()

            model.eval()
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True):
                val_logits = model(
                    torch.as_tensor(curve_inputs[val_index], device=device),
                    torch.as_tensor(val_scalars, device=device),
                )
                val_loss = masked_loss(
                    val_logits,
                    torch.as_tensor(labels[val_index], device=device),
                    torch.as_tensor(masks[val_index], device=device),
                    torch.as_tensor(weights, device=device),
                ).item()
            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True):
            probabilities = torch.sigmoid(
                model(
                    torch.as_tensor(curve_inputs[test_index], device=device),
                    torch.as_tensor(test_scalars, device=device),
                )
            ).float().cpu().numpy()
        for task_index in range(len(active_tasks)):
            valid = masks[test_index, task_index] > 0
            global_valid = test_index[valid]
            score_sum[global_valid, task_index] += probabilities[valid, task_index]
            score_count[global_valid, task_index] += 1
        log_rows.append(
            {
                "repeat": split["repeat"],
                "fold": split["fold"],
                "device": torch.cuda.get_device_name(0),
                "best_epoch": best_epoch,
                "epochs_run": epoch + 1,
                "best_validation_loss": best_loss,
                "seconds": time.time() - start,
                "parameter_count": parameter_count,
                "scalar_features": len(scalar_columns),
            }
        )
        del model
        torch.cuda.empty_cache()

    prediction_rows: list[dict[str, object]] = []
    for task_index, spec in enumerate(active_tasks):
        valid = (masks[:, task_index] > 0) & (score_count[:, task_index] > 0)
        for index in np.flatnonzero(valid):
            prediction_rows.append(
                {
                    "task": spec.name,
                    "model": "gpu_segment_curve_net",
                    "transition_id": transitions.iloc[index]["transition_id"],
                    "patient_id": transitions.iloc[index]["patient_id"],
                    "label": int(labels[index, task_index]),
                    "score": float(score_sum[index, task_index] / score_count[index, task_index]),
                    "prediction_repeats": int(score_count[index, task_index]),
                }
            )
    gpu_metadata = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "parameter_count": int(parameter_count or 0),
        "curve_input": "18 segments x 6 channels x 96 phase points",
        "curve_channels": [
            "current_endo",
            "current_mid",
            "current_endo_minus_mid",
            "change_endo_from_previous",
            "change_mid_from_previous",
            "change_endo_minus_mid_from_previous",
        ],
        "scalar_features": len(scalar_columns),
    }
    return pd.DataFrame(prediction_rows), pd.DataFrame(log_rows), gpu_metadata


def metric_values(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    k = max(1, int(math.ceil(0.20 * len(y))))
    selected = np.argsort(-score)[:k]
    true_positive = int(y[selected].sum())
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "brier_score": float(brier_score_loss(y, score)),
        "sensitivity_top20pct": float(true_positive / y.sum()) if y.sum() else np.nan,
        "precision_top20pct": float(true_positive / k),
    }


def cluster_sample_indices(patient_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    patients = np.unique(patient_ids)
    sampled = rng.choice(patients, size=len(patients), replace=True)
    pieces = [np.flatnonzero(patient_ids == patient) for patient in sampled]
    return np.concatenate(pieces)


def evaluate_predictions(
    predictions: pd.DataFrame, bootstraps: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    for group_index, ((task, model), group) in enumerate(
        predictions.groupby(["task", "model"], sort=False)
    ):
        y = group["label"].to_numpy(int)
        score = group["score"].to_numpy(float)
        if np.unique(y).size < 2:
            continue
        row = {"task": task, "model": model, **metric_values(y, score)}
        rng = np.random.default_rng(seed + group_index)
        samples = {key: [] for key in ["roc_auc", "average_precision", "brier_score"]}
        patients = group["patient_id"].to_numpy(str)
        for _ in range(bootstraps):
            index = cluster_sample_indices(patients, rng)
            if np.unique(y[index]).size < 2:
                continue
            result = metric_values(y[index], score[index])
            for key in samples:
                samples[key].append(result[key])
        for key, values in samples.items():
            row[f"{key}_ci_low"] = float(np.quantile(values, 0.025))
            row[f"{key}_ci_high"] = float(np.quantile(values, 0.975))
        metrics.append(row)
    metric_frame = pd.DataFrame(metrics)
    deltas = paired_deltas(predictions, bootstraps, seed)
    return metric_frame, deltas


def paired_deltas(predictions: pd.DataFrame, bootstraps: int, seed: int) -> pd.DataFrame:
    rows = []
    for task_index, (task, task_group) in enumerate(predictions.groupby("task", sort=False)):
        base = task_group[task_group["model"] == "clinical_ridge"][
            ["transition_id", "patient_id", "label", "score"]
        ].rename(columns={"score": "base_score"})
        if base.empty:
            continue
        for model_index, (model, group) in enumerate(task_group.groupby("model", sort=False)):
            if model == "clinical_ridge":
                continue
            merged = group[["transition_id", "patient_id", "label", "score"]].merge(
                base,
                on=["transition_id", "patient_id", "label"],
                how="inner",
            )
            y = merged["label"].to_numpy(int)
            score = merged["score"].to_numpy(float)
            base_score = merged["base_score"].to_numpy(float)
            observed = metric_values(y, score)
            observed_base = metric_values(y, base_score)
            rng = np.random.default_rng(seed + task_index * 100 + model_index)
            samples = {key: [] for key in ["roc_auc", "average_precision", "brier_score"]}
            patients = merged["patient_id"].to_numpy(str)
            for _ in range(bootstraps):
                index = cluster_sample_indices(patients, rng)
                if np.unique(y[index]).size < 2:
                    continue
                m = metric_values(y[index], score[index])
                b = metric_values(y[index], base_score[index])
                for key in samples:
                    samples[key].append(m[key] - b[key])
            row: dict[str, object] = {
                "task": task,
                "model": model,
                "reference_model": "clinical_ridge",
                "n": len(y),
            }
            for key, values in samples.items():
                row[f"delta_{key}"] = observed[key] - observed_base[key]
                row[f"delta_{key}_ci_low"] = float(np.quantile(values, 0.025))
                row[f"delta_{key}_ci_high"] = float(np.quantile(values, 0.975))
            rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    if frame.empty:
        return "No estimable rows."
    display = frame[columns].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        )
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in display.to_numpy()]
    return "\n".join([header, rule] + body)


def make_plots(
    output_dir: Path,
    audit: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    deltas: pd.DataFrame,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    mid = audit[audit["metric"] == "mid"].copy()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = {"first": "#4472C4", "roll2": "#70AD47", "roll3": "#ED7D31"}
    for offset, baseline in enumerate(["first", "roll2", "roll3"]):
        subset = mid[mid["baseline"] == baseline].sort_values("relative_threshold")
        x = np.arange(len(subset)) + (offset - 1) * 0.25
        ax.bar(x, subset["event_rate"] * 100, width=0.24, label=baseline, color=colors[baseline])
        for xx, event, total, rate in zip(x, subset["events"], subset["eligible_transitions"], subset["event_rate"]):
            ax.text(xx, rate * 100 + 1, f"{int(event)}/{int(total)}", ha="center", fontsize=7, rotation=90)
    thresholds = sorted(mid["relative_threshold"].unique())
    ax.set_xticks(np.arange(len(thresholds)), [f"{int(t*100)}%" for t in thresholds])
    ax.set_ylabel("Next-visit event rate (%)")
    ax.set_xlabel("Relative Mid-GLS deterioration threshold")
    ax.set_title("Next-visit event prevalence by baseline definition")
    ax.legend(title="Baseline")
    ax.set_ylim(0, max(mid["event_rate"] * 100) * 1.35)
    fig.tight_layout()
    fig.savefig(figures / "next_visit_event_rates.png", dpi=180)
    plt.close(fig)

    mid_tasks = audit[(audit["metric"] == "mid") & (audit["events"] >= 8)]["task"]
    plot_delta = deltas[
        deltas["task"].isin(mid_tasks)
        & deltas["model"].isin(
            [
                "clinical_plus_transmural_sparse",
                "clinical_plus_variability_sparse",
                "combined_extra_trees",
                "gpu_segment_curve_net",
            ]
        )
    ].copy()
    fig, ax = plt.subplots(figsize=(12, 6.5))
    models = list(plot_delta["model"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(models)))
    task_order = [task for task in mid_tasks if task in set(plot_delta["task"])]
    for model_index, (model, color) in enumerate(zip(models, palette)):
        subset = plot_delta[plot_delta["model"] == model].set_index("task").reindex(task_order)
        x = np.arange(len(task_order)) + (model_index - 1.5) * 0.14
        ax.scatter(x, subset["delta_average_precision"], label=model.replace("_", " "), color=color)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(task_order)), [task.replace("_", "\n") for task in task_order], fontsize=7)
    ax.set_ylabel("Average-precision difference vs clinical ridge")
    ax.set_title("Incremental next-visit alert ranking")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "mid_gls_ap_deltas.png", dpi=180)
    plt.close(fig)

    primary = predictions[predictions["task"] == "mid_first_rel15"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model, color in zip(primary["model"].unique(), plt.cm.tab10(np.linspace(0, 1, primary["model"].nunique()))):
        group = primary[primary["model"] == model].sort_values("score", ascending=False).reset_index(drop=True)
        recall = group["label"].cumsum() / max(group["label"].sum(), 1)
        budget = (np.arange(len(group)) + 1) / len(group)
        ax.plot(budget, recall, label=model.replace("_", " "), color=color)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")
    ax.set_xlabel("Fraction of current visits alerted")
    ax.set_ylabel("Fraction of next-visit events captured")
    ax.set_title("Next-visit alert curve: first baseline, 15% Mid-GLS")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "primary_next_visit_alert_curve.png", dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    transitions: pd.DataFrame,
    audit: pd.DataFrame,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    gpu_metadata: dict[str, object],
    cv_repeats: int,
) -> None:
    primary = metrics[metrics["task"] == "mid_first_rel15"].sort_values(
        "average_precision", ascending=False
    )
    clinical_primary = primary[primary["model"] == "clinical_ridge"].iloc[0]
    best_primary = primary.iloc[0]
    mid_audit = audit[audit["metric"] == "mid"][
        ["task", "baseline", "relative_threshold", "eligible_transitions", "events", "event_rate"]
    ]
    summary_rows = []
    for task, task_metrics in metrics[
        metrics["task"].isin(mid_audit[mid_audit["events"] >= 8]["task"])
    ].groupby("task", sort=False):
        clinical = task_metrics[task_metrics["model"] == "clinical_ridge"].iloc[0]
        best = task_metrics.sort_values("average_precision", ascending=False).iloc[0]
        delta = deltas[(deltas["task"] == task) & (deltas["model"] == best["model"])]
        summary_rows.append(
            {
                "task": task,
                "events/n": f"{int(best['events'])}/{int(best['n'])}",
                "clinical_AP": clinical["average_precision"],
                "best_model": best["model"],
                "best_AP": best["average_precision"],
                "best_AUC": best["roc_auc"],
                "delta_AP_CI": "reference"
                if delta.empty
                else f"{delta.iloc[0]['delta_average_precision']:+.3f} [{delta.iloc[0]['delta_average_precision_ci_low']:+.3f}, {delta.iloc[0]['delta_average_precision_ci_high']:+.3f}]",
            }
        )
    summary = pd.DataFrame(summary_rows)
    stable = deltas[
        (deltas["task"].str.startswith("mid_"))
        & (deltas["delta_average_precision_ci_low"] > 0)
    ]
    if stable.empty:
        bottom_line = "No Endo-Mid, variability, nonlinear, or GPU model showed a bootstrap-stable average-precision gain over the clinical trajectory across the Mid-GLS next-visit tasks."
    else:
        row = stable.sort_values("delta_average_precision", ascending=False).iloc[0]
        bottom_line = f"The strongest stable Mid-GLS improvement was {row['model']} for {row['task']} (delta AP {row['delta_average_precision']:+.3f})."

    report = f"""# Next-visit cardiotoxicity alert study

## Bottom line

{bottom_line}

Primary task—first visit as baseline and 15% relative Mid-GLS deterioration at the immediately following visit: clinical ridge AUC {clinical_primary['roc_auc']:.3f}, AP {clinical_primary['average_precision']:.3f}; best model **{best_primary['model']}** AUC {best_primary['roc_auc']:.3f}, AP {best_primary['average_precision']:.3f}.

This is patient-held-out exploratory validation, not a clinical alert system.

## What was predicted

- One sample is **current visit → immediately next visit**. All {len(transitions)} adjacent transitions from {transitions['patient_id'].nunique()} patients were used when eligible.
- `first`: deterioration is relative to visit 1. Only transitions up to the first threshold crossing are eligible; the visit immediately before that crossing is the positive alert visit.
- `roll2` / `roll3`: the next visit is compared with the mean of the last 2 or 3 visits available at the current visit.
- Every endpoint is a **relative decline**. No absolute-decline labels were used.
- Mid-GLS thresholds: 10%, 12%, 15%, and 20%. Endo-GLS and relative EF decline are sensitivity analyses.

## Features

| family | features |
| --- | --- |
| Clinical trajectory ({len(feature_sets['clinical'])}) | Current/baseline Mid and Endo GLS, EF, relative change from first/rolling baselines, last-visit change, slopes, intervals, history length, Endo–Mid GLS gap. |
| Endo–Mid engineered ({len(feature_sets['clinical_plus_transmural']) - len(feature_sets['clinical'])}) | Segment-paired amplitude and time-to-peak gaps, curve/shape distances, Endo–Mid coherence, phase lag, and early change in these quantities. |
| Inter-segment variability ({len(feature_sets['clinical_plus_variability']) - len(feature_sets['clinical'])}) | Robust peak-amplitude dispersion, circular time-to-peak dispersion, curve/shape incoherence, regional gradients, graph roughness, and change from the previous visit. |
| Raw GPU curves | 18 segments × 96 phase points with current Endo, current Mid, Endo−Mid, and the three corresponding changes from the previous visit. |

## Models

| model | implementation |
| --- | --- |
| `clinical_ridge` | L2 logistic regression on clinical trajectory features. |
| `clinical_plus_transmural_sparse` | L1 logistic regression adding engineered Endo–Mid features. |
| `clinical_plus_variability_sparse` | L1 logistic regression adding segment-variability features. |
| `combined_extra_trees` | Constrained nonlinear Extra Trees using all engineered features. |
| `gpu_segment_curve_net` | {gpu_metadata['parameter_count']:,}-parameter shared 1D segment CNN, mean/std/max segment pooling, plus clinical/variability scalars; trained on **{gpu_metadata['device']}**. |

All visits from a patient stay in the same fold. Results average {cv_repeats} repeated five-fold patient splits. Confidence intervals use patient-cluster bootstrap.

## Relative-label audit

{markdown_table(mid_audit, ['task', 'baseline', 'relative_threshold', 'eligible_transitions', 'events', 'event_rate'])}

Rolling-3 strict thresholds have very few events and are sensitivity checks only.

## Mid-GLS result summary

{markdown_table(summary, ['task', 'events/n', 'clinical_AP', 'best_model', 'best_AP', 'best_AUC', 'delta_AP_CI'])}

`delta_AP_CI` compares the best model with clinical ridge. An interval crossing zero is not a stable improvement.

## Primary 15% first-baseline task

{markdown_table(primary, ['model', 'n', 'events', 'roc_auc', 'roc_auc_ci_low', 'roc_auc_ci_high', 'average_precision', 'average_precision_ci_low', 'average_precision_ci_high', 'sensitivity_top20pct', 'precision_top20pct'])}

## Interpretation

- Fixed-first labels answer “will the next visit be the first threshold crossing?” and best match a surveillance alert.
- Rolling baselines adapt to drift but can label repeated episodes and have fewer eligible transitions, especially for `roll3`.
- A GPU model is worth retaining only if it improves patient-held-out ranking across neighboring thresholds or baseline definitions, not one isolated task.
- The [2022 ESC definitions](https://academic.oup.com/ehjcimaging/article/23/10/e333/6675075) use >15% relative GLS decline as one component of CTRCD. This dataset lacks biomarkers, treatment exposure, symptoms, and adjudication, so these are imaging-deterioration alerts only.

## Files

- `next_visit_transitions.parquet`: one row per current→next visit sample and every relative label.
- `label_audit.csv`: sample/event counts for all baseline and threshold definitions.
- `oof_predictions.parquet`, `model_metrics.csv`, `model_deltas_vs_clinical.csv`.
- `gpu_training_log.csv`, `patient_fold_assignments.csv`, `feature_manifest.csv`.
- `figures/`: event-rate, AP-delta, and alert-budget figures.
"""
    (output_dir / "next_visit_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    set_seed(args.seed)

    visits = pd.read_parquet(args.visits.resolve())
    tasks = task_specs()
    transmural_visits = curve_feature_table(args.curves.resolve())
    curve_tensors = build_visit_curve_tensors(args.curves.resolve())
    transitions, curve_inputs, feature_sets = build_transitions(
        visits, transmural_visits, curve_tensors, tasks
    )
    audit = label_audit(transitions, tasks)
    active_names = set(
        audit[
            (audit["eligible_transitions"] >= 30)
            & (audit["events"] >= 8)
            & ((audit["eligible_transitions"] - audit["events"]) >= 8)
        ]["task"]
    )
    active_tasks = [spec for spec in tasks if spec.name in active_names]
    splits, assignments = patient_splits(transitions, args.cv_repeats, args.seed)

    transitions.to_parquet(output_dir / "next_visit_transitions.parquet", index=False)
    audit.to_csv(output_dir / "label_audit.csv", index=False)
    assignments.to_csv(output_dir / "patient_fold_assignments.csv", index=False)
    feature_manifest = []
    for family, columns in feature_sets.items():
        for column in columns:
            feature_manifest.append({"feature_set": family, "feature": column})
    pd.DataFrame(feature_manifest).to_csv(output_dir / "feature_manifest.csv", index=False)

    cpu_predictions = cpu_oof_predictions(
        transitions, feature_sets, active_tasks, splits, args.seed
    )
    gpu_predictions, gpu_log, gpu_metadata = gpu_oof_predictions(
        transitions,
        curve_inputs,
        feature_sets,
        active_tasks,
        splits,
        args.epochs,
        args.patience,
        args.seed,
    )
    predictions = pd.concat([cpu_predictions, gpu_predictions], ignore_index=True)
    metrics, deltas = evaluate_predictions(predictions, args.bootstraps, args.seed)
    predictions.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    deltas.to_csv(output_dir / "model_deltas_vs_clinical.csv", index=False)
    gpu_log.to_csv(output_dir / "gpu_training_log.csv", index=False)
    make_plots(output_dir, audit, metrics, predictions, deltas)
    write_report(
        output_dir,
        transitions,
        audit,
        metrics,
        deltas,
        feature_sets,
        gpu_metadata,
        args.cv_repeats,
    )

    metadata = {
        "raw_directory": str(args.raw_dir.resolve()),
        "raw_txt_files": len(list(args.raw_dir.resolve().glob("*.txt"))),
        "visits_path": str(args.visits.resolve()),
        "visits_sha256": sha256_file(args.visits.resolve()),
        "curves_path": str(args.curves.resolve()),
        "curves_sha256": sha256_file(args.curves.resolve()),
        "patients": int(transitions["patient_id"].nunique()),
        "adjacent_transitions": len(transitions),
        "active_tasks": len(active_tasks),
        "active_task_names": [spec.name for spec in active_tasks],
        "cv_repeats": args.cv_repeats,
        "folds_per_repeat": 5,
        "bootstraps": args.bootstraps,
        "seed": args.seed,
        "gpu": gpu_metadata,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
