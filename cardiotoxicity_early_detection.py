from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline


DEFAULT_INPUT = Path(r"D:\us\amber_full_105_preprocessed")
DEFAULT_RAW = Path(r"D:\DS\anonymized_reports")
DEFAULT_OUTPUT = Path(r"D:\us\cardiotoxicity_early_detection_results")


VARIABILITY_SUFFIXES = [
    "peak_abs_robust_sd",
    "peak_abs_cv",
    "time_to_peak_norm_circular_std",
    "vendor_peak_systolic_abs_robust_sd",
    "vendor_time_to_peak_norm_circular_std",
    "post_systolic_fraction",
    "curve_dispersion_rms",
    "curve_pairwise_rmse",
    "curve_integrated_robust_mad",
    "shape_dispersion_rms",
    "shape_pairwise_rmse",
    "shape_incoherence",
    "within_view_peak_robust_sd_mean",
    "within_ring_peak_robust_sd_mean",
    "spatial_peak_graph_roughness",
    "spatial_timing_graph_roughness",
    "impaired_segment_fraction_lt15",
    "apical_basal_peak_gradient",
]


@dataclass(frozen=True)
class OutcomeSpec:
    name: str
    label_column: str
    mask_column: str
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-visit early-detection study for final cardiotoxicity deterioration."
    )
    parser.add_argument("--visits", type=Path, default=DEFAULT_INPUT / "Ichilov_july_visits.parquet")
    parser.add_argument("--curves", type=Path, default=DEFAULT_INPUT / "Ichilov_july_dataset.parquet")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--stability-bootstraps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def robust_sd(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    center = np.median(values)
    return float(1.4826 * np.median(np.abs(values - center)))


def finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.nanstd(a[mask]) < 1e-10 or np.nanstd(b[mask]) < 1e-10:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def add_distribution_features(row: dict[str, object], name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        for suffix in ["mean", "std", "robust_sd", "iqr", "p90_p10", "median"]:
            row[f"tm_{name}_{suffix}"] = np.nan
        return
    row[f"tm_{name}_mean"] = float(np.mean(values))
    row[f"tm_{name}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    row[f"tm_{name}_robust_sd"] = robust_sd(values)
    row[f"tm_{name}_iqr"] = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    row[f"tm_{name}_p90_p10"] = float(np.quantile(values, 0.90) - np.quantile(values, 0.10))
    row[f"tm_{name}_median"] = float(np.median(values))


def normalized_shape(curves: np.ndarray) -> np.ndarray:
    curves = np.asarray(curves, dtype=float)
    scale = np.nanmax(np.abs(curves), axis=1, keepdims=True)
    invalid = ~np.isfinite(scale) | (scale < 3.0)
    scale[invalid] = 1.0
    result = curves / scale
    result[np.repeat(invalid, curves.shape[1], axis=1)] = np.nan
    return result


def curve_feature_table(curves_path: Path) -> pd.DataFrame:
    columns = [
        "analysis_id",
        "visit_id",
        "patient_id",
        "curve_family",
        "layer",
        "segment_number",
        "peak_abs",
        "time_to_peak_norm",
        "vendor_peak_systolic_abs",
        "vendor_time_to_peak_norm",
        "resampled_values",
    ]
    curves = pd.read_parquet(curves_path, columns=columns)
    curves = curves[
        (curves["curve_family"] == "longitudinal_strain")
        & curves["layer"].isin(["endo", "mid"])
        & curves["segment_number"].notna()
    ].copy()

    analysis_rows: list[dict[str, object]] = []
    for analysis_id, group in curves.groupby("analysis_id", sort=False):
        endo = group[group["layer"] == "endo"].set_index("segment_number").sort_index()
        mid = group[group["layer"] == "mid"].set_index("segment_number").sort_index()
        segments = endo.index.intersection(mid.index)
        if len(segments) < 12:
            continue
        endo = endo.loc[segments]
        mid = mid.loc[segments]
        endo_curves = np.vstack([np.asarray(v, dtype=float) for v in endo["resampled_values"]])
        mid_curves = np.vstack([np.asarray(v, dtype=float) for v in mid["resampled_values"]])
        if endo_curves.shape != mid_curves.shape or endo_curves.shape[1] < 20:
            continue

        diff = endo_curves - mid_curves
        shape_diff = normalized_shape(endo_curves) - normalized_shape(mid_curves)
        peak_gap = endo["peak_abs"].to_numpy(float) - mid["peak_abs"].to_numpy(float)
        endo_peak = endo["peak_abs"].to_numpy(float)
        mid_peak = mid["peak_abs"].to_numpy(float)
        peak_ratio = np.full_like(endo_peak, np.nan, dtype=float)
        ratio_valid = np.isfinite(endo_peak) & np.isfinite(mid_peak) & (endo_peak >= 3.0) & (mid_peak >= 3.0)
        peak_ratio[ratio_valid] = np.clip(endo_peak[ratio_valid] / mid_peak[ratio_valid], 0.0, 5.0)
        ttp_gap = endo["time_to_peak_norm"].to_numpy(float) - mid["time_to_peak_norm"].to_numpy(float)
        vendor_peak_gap = (
            endo["vendor_peak_systolic_abs"].to_numpy(float)
            - mid["vendor_peak_systolic_abs"].to_numpy(float)
        )
        vendor_ttp_gap = (
            endo["vendor_time_to_peak_norm"].to_numpy(float)
            - mid["vendor_time_to_peak_norm"].to_numpy(float)
        )
        curve_rms = np.sqrt(np.nanmean(diff**2, axis=1))
        curve_abs_area = np.nanmean(np.abs(diff), axis=1)
        shape_rms = np.sqrt(np.nanmean(shape_diff**2, axis=1))
        segment_corr = np.asarray(
            [finite_corr(endo_curves[i], mid_curves[i]) for i in range(len(segments))], dtype=float
        )

        row: dict[str, object] = {
            "analysis_id": analysis_id,
            "visit_id": str(group["visit_id"].iloc[0]),
            "patient_id": str(group["patient_id"].iloc[0]),
            "tm_paired_segments": int(len(segments)),
            "tm_peak_layer_correlation": finite_corr(
                endo["peak_abs"].to_numpy(float), mid["peak_abs"].to_numpy(float)
            ),
            "tm_ttp_layer_correlation": finite_corr(
                endo["time_to_peak_norm"].to_numpy(float), mid["time_to_peak_norm"].to_numpy(float)
            ),
            "tm_fraction_endo_peak_gt_mid": float(np.mean(peak_gap > 0)),
            "tm_fraction_ttp_discordant_gt_5pct_cycle": float(np.mean(np.abs(ttp_gap) > 0.05)),
        }
        add_distribution_features(row, "peak_gap", peak_gap)
        add_distribution_features(row, "peak_ratio", peak_ratio)
        add_distribution_features(row, "ttp_gap", ttp_gap)
        add_distribution_features(row, "vendor_peak_gap", vendor_peak_gap)
        add_distribution_features(row, "vendor_ttp_gap", vendor_ttp_gap)
        add_distribution_features(row, "curve_rms", curve_rms)
        add_distribution_features(row, "curve_abs_area", curve_abs_area)
        add_distribution_features(row, "shape_rms", shape_rms)
        add_distribution_features(row, "segment_curve_correlation", segment_corr)

        mean_diff = np.nanmean(diff, axis=0)
        sd_diff = np.nanstd(diff, axis=0)
        mean_shape_diff = np.nanmean(shape_diff, axis=0)
        for prefix, waveform in [
            ("mean_gap", mean_diff),
            ("sd_gap", sd_diff),
            ("mean_shape_gap", mean_shape_diff),
        ]:
            coefficients = dct(np.nan_to_num(waveform, nan=0.0), norm="ortho")[:8]
            for index, value in enumerate(coefficients, start=1):
                row[f"tm_{prefix}_dct{index:02d}"] = float(value)

        endo_consensus = np.nanmean(endo_curves, axis=0)
        mid_consensus = np.nanmean(mid_curves, axis=0)
        row["tm_consensus_curve_correlation"] = finite_corr(endo_consensus, mid_consensus)
        centered_endo = np.nan_to_num(endo_consensus - np.nanmean(endo_consensus))
        centered_mid = np.nan_to_num(mid_consensus - np.nanmean(mid_consensus))
        cross = np.correlate(centered_endo, centered_mid, mode="full")
        row["tm_consensus_phase_lag_fraction"] = float((np.argmax(cross) - (len(endo_consensus) - 1)) / max(len(endo_consensus) - 1, 1))
        analysis_rows.append(row)

    analysis = pd.DataFrame(analysis_rows)
    numeric = [c for c in analysis.columns if c not in {"analysis_id", "visit_id", "patient_id"}]
    visit = analysis.groupby(["visit_id", "patient_id"], as_index=False)[numeric].mean()
    return visit


def build_landmarks(visits: pd.DataFrame, transmural: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    visits = visits.sort_values(["patient_id", "visit_order"]).copy()
    transmural = transmural.set_index("visit_id")
    variability_columns = [
        f"{layer}_{suffix}"
        for layer in ["endo", "mid"]
        for suffix in VARIABILITY_SUFFIXES
        if f"{layer}_{suffix}" in visits.columns
    ]
    tm_columns = [c for c in transmural.columns if c.startswith("tm_")]

    rows: list[dict[str, object]] = []
    clinical_features: list[str] = []
    variability_features: list[str] = []
    transmural_features: list[str] = []

    for patient_id, group in visits.groupby("patient_id", sort=False):
        group = group.sort_values("visit_order").reset_index(drop=True)
        if len(group) < 3:
            continue
        v1, v2, final = group.iloc[0], group.iloc[1], group.iloc[-1]
        v1_mid, v2_mid, final_mid = abs(v1["gls_mid_peak_avg"]), abs(v2["gls_mid_peak_avg"]), abs(final["gls_mid_peak_avg"])
        v1_endo, v2_endo, final_endo = abs(v1["gls_endo_peak_avg"]), abs(v2["gls_endo_peak_avg"]), abs(final["gls_endo_peak_avg"])
        v1_ef, v2_ef, final_ef = v1["ef_biplane"], v2["ef_biplane"], final["ef_biplane"]

        row: dict[str, object] = {
            "patient_id": str(patient_id),
            "n_visits": int(len(group)),
            "v1_visit_id": str(v1["visit_id"]),
            "v2_visit_id": str(v2["visit_id"]),
            "final_visit_id": str(final["visit_id"]),
            "days_v1_v2": float(v2["days_since_baseline"] - v1["days_since_baseline"]),
            "days_v2_final": float(final["days_since_baseline"] - v2["days_since_baseline"]),
            "days_v1_final": float(final["days_since_baseline"] - v1["days_since_baseline"]),
            "v1_mid_gls": float(v1_mid),
            "v2_mid_gls": float(v2_mid),
            "early_mid_abs_decline": float(v1_mid - v2_mid),
            "early_mid_rel_decline": float(1.0 - v2_mid / v1_mid),
            "v1_endo_gls": float(v1_endo),
            "v2_endo_gls": float(v2_endo),
            "early_endo_abs_decline": float(v1_endo - v2_endo),
            "early_endo_rel_decline": float(1.0 - v2_endo / v1_endo),
            "v1_ef": float(v1_ef) if pd.notna(v1_ef) else np.nan,
            "v2_ef": float(v2_ef) if pd.notna(v2_ef) else np.nan,
            "early_ef_abs_decline": float(v1_ef - v2_ef) if pd.notna(v1_ef) and pd.notna(v2_ef) else np.nan,
            "v1_gls_layer_gap": float(v1_endo - v1_mid),
            "v2_gls_layer_gap": float(v2_endo - v2_mid),
            "early_gls_layer_gap_change": float((v2_endo - v2_mid) - (v1_endo - v1_mid)),
            "final_mid_relative_decline": float(1.0 - final_mid / v1_mid),
            "future_mid_relative_decline": float(1.0 - final_mid / v2_mid),
            "final_mid_absolute_decline": float(v1_mid - final_mid),
            "future_mid_absolute_decline": float(v2_mid - final_mid),
            "final_endo_relative_decline": float(1.0 - final_endo / v1_endo),
            "future_endo_relative_decline": float(1.0 - final_endo / v2_endo),
            "final_ef_absolute_decline": float(v1_ef - final_ef) if pd.notna(v1_ef) and pd.notna(final_ef) else np.nan,
            "future_ef_absolute_decline": float(v2_ef - final_ef) if pd.notna(v2_ef) and pd.notna(final_ef) else np.nan,
        }

        for threshold in [0.10, 0.12, 0.15, 0.20]:
            tag = int(round(threshold * 100))
            row[f"label_mid_rel_{tag}"] = int(row["final_mid_relative_decline"] >= threshold)
            row[f"eligible_mid_rel_{tag}"] = True
            row[f"label_incident_mid_rel_{tag}"] = int(row["final_mid_relative_decline"] >= threshold)
            row[f"eligible_incident_mid_rel_{tag}"] = bool(row["early_mid_rel_decline"] < threshold)
        for threshold in [0.10, 0.15]:
            tag = int(round(threshold * 100))
            row[f"label_endo_rel_{tag}"] = int(row["final_endo_relative_decline"] >= threshold)
            row[f"eligible_endo_rel_{tag}"] = True
        row["label_mid_abs_2p9"] = int(row["final_mid_absolute_decline"] >= 2.9)
        row["eligible_mid_abs_2p9"] = True

        ef_valid = pd.notna(row["final_ef_absolute_decline"])
        for threshold in [5, 7, 10]:
            row[f"label_ef_abs_{threshold}"] = int(ef_valid and row["final_ef_absolute_decline"] >= threshold)
            row[f"eligible_ef_abs_{threshold}"] = bool(ef_valid)
        incident_ef_valid = ef_valid and pd.notna(row["early_ef_abs_decline"])
        row["label_incident_ef_abs_5"] = int(ef_valid and row["final_ef_absolute_decline"] >= 5)
        row["eligible_incident_ef_abs_5"] = bool(incident_ef_valid and row["early_ef_abs_decline"] < 5)
        row["label_concordant_mid15_ef5"] = int(
            ef_valid and row["final_mid_relative_decline"] >= 0.15 and row["final_ef_absolute_decline"] >= 5
        )
        row["eligible_concordant_mid15_ef5"] = bool(ef_valid)
        row["label_either_mid15_ef5"] = int(
            row["final_mid_relative_decline"] >= 0.15
            or (ef_valid and row["final_ef_absolute_decline"] >= 5)
        )
        row["eligible_either_mid15_ef5"] = bool(ef_valid)

        for column in variability_columns:
            row[f"v1_var__{column}"] = v1[column]
            row[f"v2_var__{column}"] = v2[column]
            row[f"d_var__{column}"] = v2[column] - v1[column]

        for column in tm_columns:
            if str(v1["visit_id"]) in transmural.index and str(v2["visit_id"]) in transmural.index:
                v1_value = transmural.loc[str(v1["visit_id"]), column]
                v2_value = transmural.loc[str(v2["visit_id"]), column]
                if isinstance(v1_value, pd.Series):
                    v1_value = v1_value.mean()
                if isinstance(v2_value, pd.Series):
                    v2_value = v2_value.mean()
                row[f"v1_{column}"] = v1_value
                row[f"v2_{column}"] = v2_value
                row[f"d_{column}"] = v2_value - v1_value
            else:
                row[f"v1_{column}"] = np.nan
                row[f"v2_{column}"] = np.nan
                row[f"d_{column}"] = np.nan
        rows.append(row)

    landmarks = pd.DataFrame(rows)
    clinical_features = [
        "days_v1_v2",
        "v1_mid_gls",
        "v2_mid_gls",
        "early_mid_abs_decline",
        "early_mid_rel_decline",
        "v1_endo_gls",
        "v2_endo_gls",
        "early_endo_abs_decline",
        "early_endo_rel_decline",
        "v1_ef",
        "v2_ef",
        "early_ef_abs_decline",
    ]
    variability_features = [c for c in landmarks.columns if "_var__" in c]
    transmural_features = [c for c in landmarks.columns if c.startswith(("v1_tm_", "v2_tm_", "d_tm_"))]
    feature_sets = {
        "clinical": clinical_features,
        "transmural": transmural_features,
        "variability": variability_features,
        "clinical_plus_transmural": clinical_features + transmural_features,
        "clinical_plus_variability": clinical_features + variability_features,
        "combined": clinical_features + transmural_features + variability_features,
    }
    return landmarks, feature_sets


def outcome_specs() -> list[OutcomeSpec]:
    return [
        OutcomeSpec("mid_gls_relative_drop_10", "label_mid_rel_10", "eligible_mid_rel_10", "Final Mid-GLS magnitude drop >=10% from visit 1"),
        OutcomeSpec("mid_gls_relative_drop_12", "label_mid_rel_12", "eligible_mid_rel_12", "Final Mid-GLS magnitude drop >=12% from visit 1"),
        OutcomeSpec("mid_gls_relative_drop_15", "label_mid_rel_15", "eligible_mid_rel_15", "Final Mid-GLS magnitude drop >=15% from visit 1"),
        OutcomeSpec("mid_gls_relative_drop_20", "label_mid_rel_20", "eligible_mid_rel_20", "Final Mid-GLS magnitude drop >=20% from visit 1"),
        OutcomeSpec("mid_gls_absolute_drop_2p9", "label_mid_abs_2p9", "eligible_mid_abs_2p9", "Final Mid-GLS magnitude drop >=2.9 points from visit 1"),
        OutcomeSpec("endo_gls_relative_drop_10", "label_endo_rel_10", "eligible_endo_rel_10", "Final Endo-GLS magnitude drop >=10% from visit 1"),
        OutcomeSpec("endo_gls_relative_drop_15", "label_endo_rel_15", "eligible_endo_rel_15", "Final Endo-GLS magnitude drop >=15% from visit 1"),
        OutcomeSpec("ef_absolute_drop_5", "label_ef_abs_5", "eligible_ef_abs_5", "Final biplane EF drop >=5 points from visit 1"),
        OutcomeSpec("ef_absolute_drop_7", "label_ef_abs_7", "eligible_ef_abs_7", "Final biplane EF drop >=7 points from visit 1"),
        OutcomeSpec("ef_absolute_drop_10", "label_ef_abs_10", "eligible_ef_abs_10", "Final biplane EF drop >=10 points from visit 1"),
        OutcomeSpec("concordant_mid15_ef5", "label_concordant_mid15_ef5", "eligible_concordant_mid15_ef5", "Both Mid-GLS >=15% and EF >=5-point final decline"),
        OutcomeSpec("either_mid15_or_ef5", "label_either_mid15_ef5", "eligible_either_mid15_ef5", "Either Mid-GLS >=15% or EF >=5-point final decline"),
        OutcomeSpec("incident_mid_gls_relative_drop_10", "label_incident_mid_rel_10", "eligible_incident_mid_rel_10", "Crosses 10% Mid-GLS decline only after visit 2"),
        OutcomeSpec("incident_mid_gls_relative_drop_15", "label_incident_mid_rel_15", "eligible_incident_mid_rel_15", "Crosses 15% Mid-GLS decline only after visit 2"),
        OutcomeSpec("incident_ef_absolute_drop_5", "label_incident_ef_abs_5", "eligible_incident_ef_abs_5", "Crosses 5-point EF decline only after visit 2"),
    ]


def fdr_bh(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    result = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return result
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    reverse = np.empty_like(order)
    reverse[order] = np.arange(len(order))
    result[valid] = adjusted[reverse]
    return result


def usable_features(frame: pd.DataFrame, columns: list[str], minimum_fraction: float = 0.60) -> list[str]:
    result = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().mean() >= minimum_fraction and series.nunique(dropna=True) > 1:
            result.append(column)
    return result


def logistic_pipeline(kind: str, seed: int) -> tuple[Pipeline, dict[str, list[float]]]:
    if kind == "ridge":
        model = LogisticRegression(
            penalty="l2", solver="liblinear", max_iter=5000, random_state=seed
        )
        grid = {"model__C": [0.03, 0.3, 3.0]}
    else:
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=5000,
            random_state=seed,
        )
        grid = {"model__C": [0.03, 0.3, 3.0]}
    pipe = Pipeline(
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
    return pipe, grid


def nested_classification_predictions(
    x: pd.DataFrame,
    y: np.ndarray,
    kind: str,
    repeats: int,
    seed: int,
) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=2)
    outer_splits = max(2, min(5, int(counts.min())))
    splitter = RepeatedStratifiedKFold(
        n_splits=outer_splits, n_repeats=repeats, random_state=seed
    )
    pred_sum = np.zeros(len(y), dtype=float)
    pred_count = np.zeros(len(y), dtype=int)
    for fold, (train, test) in enumerate(splitter.split(x, y)):
        if kind == "extra_trees":
            model = Pipeline(
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
                            random_state=seed + fold,
                        ),
                    ),
                ]
            )
            model.fit(x.iloc[train], y[train])
        else:
            pipe, grid = logistic_pipeline(kind, seed + fold)
            train_counts = np.bincount(y[train].astype(int), minlength=2)
            inner_splits = max(2, min(3, int(train_counts.min())))
            inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed + fold)
            model = GridSearchCV(
                pipe,
                grid,
                cv=inner,
                scoring="average_precision",
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            model.fit(x.iloc[train], y[train])
        pred_sum[test] += model.predict_proba(x.iloc[test])[:, 1]
        pred_count[test] += 1
    return pred_sum / np.maximum(pred_count, 1)


def classification_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    n = len(y)
    k = max(1, int(math.ceil(0.20 * n)))
    selected = np.argsort(-score)[:k]
    tp = int(y[selected].sum())
    return {
        "n": n,
        "events": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "brier_score": float(brier_score_loss(y, score)),
        "sensitivity_top20pct": float(tp / y.sum()) if y.sum() else np.nan,
        "precision_top20pct": float(tp / k),
        "alerts_top20pct": k,
    }


def bootstrap_classification(
    y: np.ndarray, score: np.ndarray, bootstraps: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ["roc_auc", "average_precision", "brier_score", "sensitivity_top20pct", "precision_top20pct"]}
    for _ in range(bootstraps):
        index = rng.integers(0, len(y), len(y))
        if np.unique(y[index]).size < 2:
            continue
        metric = classification_metrics(y[index], score[index])
        for key in values:
            values[key].append(metric[key])
    result: dict[str, float] = {}
    for key, samples in values.items():
        result[f"{key}_ci_low"] = float(np.quantile(samples, 0.025))
        result[f"{key}_ci_high"] = float(np.quantile(samples, 0.975))
    return result


def classification_study(
    landmarks: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    specs: list[OutcomeSpec],
    repeats: int,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_specs = [
        ("clinical_ridge", "clinical", "ridge"),
        ("transmural_only_sparse", "transmural", "sparse"),
        ("variability_only_sparse", "variability", "sparse"),
        ("clinical_plus_transmural", "clinical_plus_transmural", "sparse"),
        ("clinical_plus_variability", "clinical_plus_variability", "sparse"),
        ("combined_sparse", "combined", "sparse"),
        ("combined_extra_trees", "combined", "extra_trees"),
    ]
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for outcome_index, spec in enumerate(specs):
        eligible = landmarks[spec.mask_column].fillna(False).astype(bool)
        cohort = landmarks.loc[eligible].reset_index(drop=True)
        y = cohort[spec.label_column].to_numpy(int)
        audit_rows.append(
            {
                "outcome": spec.name,
                "description": spec.description,
                "eligible_patients": len(cohort),
                "events": int(y.sum()),
                "event_rate": float(y.mean()) if len(y) else np.nan,
            }
        )
        if len(cohort) < 30 or min(y.sum(), len(y) - y.sum()) < 5:
            continue
        for model_index, (model_name, set_name, kind) in enumerate(model_specs):
            if kind == "extra_trees" and min(y.sum(), len(y) - y.sum()) < 10:
                continue
            columns = usable_features(cohort, feature_sets[set_name])
            x = cohort[columns].astype(float)
            score = nested_classification_predictions(
                x,
                y,
                kind=kind,
                repeats=repeats,
                seed=seed + outcome_index * 1000 + model_index * 100,
            )
            metric = classification_metrics(y, score)
            metric.update(
                bootstrap_classification(
                    y,
                    score,
                    bootstraps=bootstraps,
                    seed=seed + outcome_index * 1000 + model_index,
                )
            )
            metric_rows.append(
                {
                    "outcome": spec.name,
                    "model": model_name,
                    "feature_set": set_name,
                    "n_features": len(columns),
                    **metric,
                }
            )
            for patient_id, truth, probability in zip(cohort["patient_id"], y, score):
                prediction_rows.append(
                    {
                        "outcome": spec.name,
                        "model": model_name,
                        "patient_id": patient_id,
                        "label": int(truth),
                        "score": float(probability),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    audit = pd.DataFrame(audit_rows)
    deltas = paired_classification_deltas(predictions, bootstraps, seed)
    return audit, metrics, predictions, deltas


def paired_classification_deltas(predictions: pd.DataFrame, bootstraps: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed + 991)
    for outcome, group in predictions.groupby("outcome", sort=False):
        base = group[group["model"] == "clinical_ridge"][["patient_id", "label", "score"]].rename(columns={"score": "base_score"})
        if base.empty:
            continue
        for model, candidate in group.groupby("model", sort=False):
            if model == "clinical_ridge":
                continue
            merged = candidate[["patient_id", "label", "score"]].merge(base, on=["patient_id", "label"], how="inner")
            y = merged["label"].to_numpy(int)
            score = merged["score"].to_numpy(float)
            base_score = merged["base_score"].to_numpy(float)
            observed = classification_metrics(y, score)
            observed_base = classification_metrics(y, base_score)
            samples = {"roc_auc": [], "average_precision": [], "brier_score": []}
            for _ in range(bootstraps):
                index = rng.integers(0, len(y), len(y))
                if np.unique(y[index]).size < 2:
                    continue
                m = classification_metrics(y[index], score[index])
                b = classification_metrics(y[index], base_score[index])
                for key in samples:
                    samples[key].append(m[key] - b[key])
            row: dict[str, object] = {"outcome": outcome, "model": model, "reference_model": "clinical_ridge", "n": len(y)}
            for key, values in samples.items():
                row[f"delta_{key}"] = observed[key] - observed_base[key]
                row[f"delta_{key}_ci_low"] = float(np.quantile(values, 0.025))
                row[f"delta_{key}_ci_high"] = float(np.quantile(values, 0.975))
            rows.append(row)
    return pd.DataFrame(rows)


def nested_regression_predictions(
    x: pd.DataFrame,
    y: np.ndarray,
    kind: str,
    repeats: int,
    seed: int,
) -> np.ndarray:
    splitter = RepeatedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    pred_sum = np.zeros(len(y), dtype=float)
    pred_count = np.zeros(len(y), dtype=int)
    for fold, (train, test) in enumerate(splitter.split(x)):
        if kind == "extra_trees":
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=300,
                            min_samples_leaf=5,
                            max_features=0.5,
                            n_jobs=-1,
                            random_state=seed + fold,
                        ),
                    ),
                ]
            )
            model.fit(x.iloc[train], y[train])
        else:
            pipe = Pipeline(
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
                    ("model", Ridge()),
                ]
            )
            inner = KFold(n_splits=3, shuffle=True, random_state=seed + fold)
            model = GridSearchCV(
                pipe,
                {"model__alpha": [1.0, 10.0, 100.0, 1000.0]},
                cv=inner,
                scoring="neg_root_mean_squared_error",
                n_jobs=1,
                refit=True,
            )
            model.fit(x.iloc[train], y[train])
        pred_sum[test] += model.predict(x.iloc[test])
        pred_count[test] += 1
    return pred_sum / np.maximum(pred_count, 1)


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rho = spearmanr(y, pred, nan_policy="omit").statistic
    return {
        "n": len(y),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "spearman_rho": float(rho),
    }


def continuous_study(
    landmarks: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    repeats: int,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = [
        "final_mid_relative_decline",
        "future_mid_relative_decline",
        "final_ef_absolute_decline",
        "future_ef_absolute_decline",
    ]
    model_specs = [
        ("clinical_ridge", "clinical", "ridge"),
        ("clinical_plus_transmural_ridge", "clinical_plus_transmural", "ridge"),
        ("clinical_plus_variability_ridge", "clinical_plus_variability", "ridge"),
        ("combined_ridge", "combined", "ridge"),
        ("combined_extra_trees", "combined", "extra_trees"),
    ]
    predictions: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    for outcome_index, outcome in enumerate(outcomes):
        cohort = landmarks[landmarks[outcome].notna()].reset_index(drop=True)
        y = cohort[outcome].to_numpy(float)
        for model_index, (model_name, set_name, kind) in enumerate(model_specs):
            columns = usable_features(cohort, feature_sets[set_name])
            pred = nested_regression_predictions(
                cohort[columns].astype(float),
                y,
                kind=kind,
                repeats=repeats,
                seed=seed + outcome_index * 1000 + model_index * 100,
            )
            metrics.append(
                {
                    "outcome": outcome,
                    "model": model_name,
                    "feature_set": set_name,
                    "n_features": len(columns),
                    **regression_metrics(y, pred),
                }
            )
            for patient_id, truth, estimate in zip(cohort["patient_id"], y, pred):
                predictions.append(
                    {
                        "outcome": outcome,
                        "model": model_name,
                        "patient_id": patient_id,
                        "observed": float(truth),
                        "prediction": float(estimate),
                    }
                )
    pred_frame = pd.DataFrame(predictions)
    metric_frame = pd.DataFrame(metrics)
    delta_frame = paired_regression_deltas(pred_frame, bootstraps, seed)
    return metric_frame, pred_frame, delta_frame


def paired_regression_deltas(predictions: pd.DataFrame, bootstraps: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed + 1991)
    for outcome, group in predictions.groupby("outcome", sort=False):
        base = group[group["model"] == "clinical_ridge"][["patient_id", "observed", "prediction"]].rename(columns={"prediction": "base_prediction"})
        for model, candidate in group.groupby("model", sort=False):
            if model == "clinical_ridge":
                continue
            merged = candidate.merge(base, on=["patient_id", "observed"], how="inner")
            y = merged["observed"].to_numpy(float)
            pred = merged["prediction"].to_numpy(float)
            base_pred = merged["base_prediction"].to_numpy(float)
            observed = regression_metrics(y, pred)
            observed_base = regression_metrics(y, base_pred)
            rmse_delta: list[float] = []
            mae_delta: list[float] = []
            rho_delta: list[float] = []
            for _ in range(bootstraps):
                index = rng.integers(0, len(y), len(y))
                m = regression_metrics(y[index], pred[index])
                b = regression_metrics(y[index], base_pred[index])
                rmse_delta.append(m["rmse"] - b["rmse"])
                mae_delta.append(m["mae"] - b["mae"])
                rho_delta.append(m["spearman_rho"] - b["spearman_rho"])
            rows.append(
                {
                    "outcome": outcome,
                    "model": model,
                    "reference_model": "clinical_ridge",
                    "n": len(y),
                    "delta_rmse": observed["rmse"] - observed_base["rmse"],
                    "delta_rmse_ci_low": float(np.nanquantile(rmse_delta, 0.025)),
                    "delta_rmse_ci_high": float(np.nanquantile(rmse_delta, 0.975)),
                    "delta_mae": observed["mae"] - observed_base["mae"],
                    "delta_mae_ci_low": float(np.nanquantile(mae_delta, 0.025)),
                    "delta_mae_ci_high": float(np.nanquantile(mae_delta, 0.975)),
                    "delta_spearman_rho": observed["spearman_rho"] - observed_base["spearman_rho"],
                    "delta_spearman_rho_ci_low": float(np.nanquantile(rho_delta, 0.025)),
                    "delta_spearman_rho_ci_high": float(np.nanquantile(rho_delta, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def residualize(vector: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(vector)), covariates])
    beta, *_ = np.linalg.lstsq(design, vector, rcond=None)
    return vector - design @ beta


def partial_association_study(
    landmarks: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    candidates = [("transmural", c) for c in feature_sets["transmural"]] + [
        ("variability", c) for c in feature_sets["variability"]
    ]
    outcomes = [
        "final_mid_relative_decline",
        "future_mid_relative_decline",
        "final_ef_absolute_decline",
        "future_ef_absolute_decline",
    ]
    covariate_columns = feature_sets["clinical"]
    rows: list[dict[str, object]] = []
    for outcome in outcomes:
        for family, feature in candidates:
            needed = [outcome, feature] + covariate_columns
            frame = landmarks[needed].copy()
            mask = frame[outcome].notna() & frame[feature].notna()
            frame = frame.loc[mask]
            if len(frame) < 30 or frame[feature].nunique() < 4:
                continue
            covariates = SimpleImputer(strategy="median").fit_transform(frame[covariate_columns])
            covariates = RobustScaler(quantile_range=(10.0, 90.0)).fit_transform(covariates)
            covariates = np.clip(covariates, -5.0, 5.0)
            x = residualize(frame[feature].to_numpy(float), covariates)
            y = residualize(frame[outcome].to_numpy(float), covariates)
            statistic = spearmanr(x, y)
            rows.append(
                {
                    "outcome": outcome,
                    "feature_family": family,
                    "feature": feature,
                    "n": len(frame),
                    "partial_spearman_rho": float(statistic.statistic),
                    "p_value": float(statistic.pvalue),
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["fdr_q"] = result.groupby("outcome")["p_value"].transform(fdr_bh)

    rng = np.random.default_rng(seed + 2991)
    for outcome, group in result.groupby("outcome"):
        top_indices = group.nsmallest(20, "p_value").index
        for index in top_indices:
            row = result.loc[index]
            feature = row["feature"]
            frame = landmarks[[outcome, feature] + covariate_columns].copy()
            frame = frame[frame[outcome].notna() & frame[feature].notna()]
            covariates = SimpleImputer(strategy="median").fit_transform(frame[covariate_columns])
            covariates = RobustScaler(quantile_range=(10.0, 90.0)).fit_transform(covariates)
            covariates = np.clip(covariates, -5.0, 5.0)
            x = residualize(frame[feature].to_numpy(float), covariates)
            y = residualize(frame[outcome].to_numpy(float), covariates)
            samples: list[float] = []
            for _ in range(min(bootstraps, 1000)):
                sample = rng.integers(0, len(frame), len(frame))
                samples.append(float(spearmanr(x[sample], y[sample]).statistic))
            result.loc[index, "ci_low"] = float(np.nanquantile(samples, 0.025))
            result.loc[index, "ci_high"] = float(np.nanquantile(samples, 0.975))
    return result.sort_values(["outcome", "p_value"])


def coefficient_stability(
    landmarks: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    cohort = landmarks[landmarks["eligible_mid_rel_15"]].reset_index(drop=True)
    y = cohort["label_mid_rel_15"].to_numpy(int)
    columns = usable_features(cohort, feature_sets["combined"])
    x = cohort[columns].astype(float)
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    x_scaled = scaler.fit_transform(imputer.fit_transform(x))
    x_scaled = np.clip(x_scaled, -5.0, 5.0)
    base = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=5000,
        random_state=seed,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(base, {"C": [0.03, 0.3, 3.0]}, cv=cv, scoring="average_precision", n_jobs=1)
    search.fit(x_scaled, y)
    chosen_c = float(search.best_params_["C"])
    rng = np.random.default_rng(seed + 3991)
    coefficients = np.zeros((bootstraps, len(columns)), dtype=float)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for b in range(bootstraps):
            index = np.concatenate(
                [
                    rng.choice(positive, size=len(positive), replace=True),
                    rng.choice(negative, size=len(negative), replace=True),
                ]
            )
            model = clone(base).set_params(C=chosen_c, random_state=seed + b)
            model.fit(x_scaled[index], y[index])
            coefficients[b] = model.coef_[0]
    nonzero = np.abs(coefficients) > 1e-6
    median = np.median(coefficients, axis=0)
    rows = []
    for j, column in enumerate(columns):
        selected = nonzero[:, j]
        if selected.any():
            dominant_sign = np.sign(np.median(coefficients[selected, j]))
            sign_consistency = float(np.mean(np.sign(coefficients[selected, j]) == dominant_sign))
        else:
            sign_consistency = np.nan
        family = "clinical" if column in feature_sets["clinical"] else ("transmural" if column in feature_sets["transmural"] else "variability")
        rows.append(
            {
                "outcome": "mid_gls_relative_drop_15",
                "feature": column,
                "feature_family": family,
                "selected_c": chosen_c,
                "selection_fraction": float(selected.mean()),
                "sign_consistency_when_selected": sign_consistency,
                "coefficient_median": float(median[j]),
                "coefficient_q25": float(np.quantile(coefficients[:, j], 0.25)),
                "coefficient_q75": float(np.quantile(coefficients[:, j], 0.75)),
                "importance_score": float(abs(median[j]) * selected.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_score", ascending=False)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    if frame.empty:
        return "No estimable rows."
    display = frame[columns].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in display.to_numpy()]
    return "\n".join([header, rule] + rows)


def make_plots(
    output_dir: Path,
    audit: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    continuous_deltas: pd.DataFrame,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    final_order = [
        "mid_gls_relative_drop_10",
        "mid_gls_relative_drop_12",
        "mid_gls_relative_drop_15",
        "mid_gls_relative_drop_20",
        "mid_gls_absolute_drop_2p9",
        "ef_absolute_drop_5",
        "ef_absolute_drop_7",
        "ef_absolute_drop_10",
    ]
    plot_audit = audit[audit["outcome"].isin(final_order)].set_index("outcome").reindex(final_order).dropna().reset_index()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(np.arange(len(plot_audit)), plot_audit["event_rate"] * 100, color="#4472C4")
    ax.set_xticks(np.arange(len(plot_audit)), [x.replace("_", "\n") for x in plot_audit["outcome"]], fontsize=8)
    ax.set_ylabel("Event rate (%)")
    ax.set_title("Final-visit deterioration prevalence is threshold-sensitive")
    for bar, events, total in zip(bars, plot_audit["events"], plot_audit["eligible_patients"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{int(events)}/{int(total)}", ha="center", fontsize=8)
    ax.set_ylim(0, max(plot_audit["event_rate"] * 100) * 1.25)
    fig.tight_layout()
    fig.savefig(figures / "threshold_event_rates.png", dpi=180)
    plt.close(fig)

    selected_outcomes = [
        "mid_gls_relative_drop_10",
        "mid_gls_relative_drop_15",
        "mid_gls_relative_drop_20",
        "ef_absolute_drop_5",
        "incident_mid_gls_relative_drop_15",
    ]
    selected_models = [
        "clinical_ridge",
        "clinical_plus_transmural",
        "clinical_plus_variability",
        "combined_sparse",
        "combined_extra_trees",
    ]
    plot_metrics = metrics[
        metrics["outcome"].isin(selected_outcomes) & metrics["model"].isin(selected_models)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_models)))
    for ax, metric_name, title in zip(axes, ["roc_auc", "average_precision"], ["ROC AUC", "Average precision"]):
        for offset_index, (model, color) in enumerate(zip(selected_models, colors)):
            subset = plot_metrics[plot_metrics["model"] == model].set_index("outcome").reindex(selected_outcomes)
            x = np.arange(len(selected_outcomes)) + (offset_index - 2) * 0.13
            ax.scatter(x, subset[metric_name], label=model.replace("_", " "), color=color, s=35)
        ax.set_xticks(np.arange(len(selected_outcomes)), [o.replace("_", "\n") for o in selected_outcomes], fontsize=7)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Repeated nested-CV performance")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Early-visit feature families: patient-held-out performance")
    fig.tight_layout()
    fig.savefig(figures / "classification_model_comparison.png", dpi=180)
    plt.close(fig)

    primary = predictions[predictions["outcome"] == "mid_gls_relative_drop_15"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model, color in zip(selected_models, colors):
        group = primary[primary["model"] == model].sort_values("score", ascending=False).reset_index(drop=True)
        if group.empty:
            continue
        cumulative_recall = group["label"].cumsum() / max(group["label"].sum(), 1)
        alert_fraction = (np.arange(len(group)) + 1) / len(group)
        ax.plot(alert_fraction, cumulative_recall, label=model.replace("_", " "), color=color)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random ranking")
    ax.set_xlabel("Fraction of patients alerted")
    ax.set_ylabel("Fraction of deterioration events captured")
    ax.set_title("Mid-GLS >=15% final decline: alert-budget curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "primary_alert_budget_curve.png", dpi=180)
    plt.close(fig)

    future = continuous_deltas[continuous_deltas["outcome"].isin(["future_mid_relative_decline", "future_ef_absolute_decline"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, outcome, title in zip(
        axes,
        ["future_mid_relative_decline", "future_ef_absolute_decline"],
        ["Future Mid-GLS relative decline", "Future EF absolute decline"],
    ):
        subset = future[future["outcome"] == outcome].reset_index(drop=True)
        labels = [model.replace("_", " ") for model in subset["model"]]
        y = np.arange(len(subset))
        ax.errorbar(
            subset["delta_rmse"],
            y,
            xerr=[subset["delta_rmse"] - subset["delta_rmse_ci_low"], subset["delta_rmse_ci_high"] - subset["delta_rmse"]],
            fmt="o",
            color="#C00000",
            ecolor="#777777",
            capsize=3,
        )
        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(y, labels, fontsize=8)
        ax.set_xlabel("RMSE difference vs clinical ridge\n(negative is better)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Incremental prediction of deterioration after visit 2")
    fig.tight_layout()
    fig.savefig(figures / "future_continuous_rmse_deltas.png", dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    landmarks: pd.DataFrame,
    audit: pd.DataFrame,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    continuous_metrics: pd.DataFrame,
    continuous_deltas: pd.DataFrame,
    associations: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    primary_outcomes = [
        "mid_gls_relative_drop_10",
        "mid_gls_relative_drop_12",
        "mid_gls_relative_drop_15",
        "mid_gls_relative_drop_20",
        "ef_absolute_drop_5",
        "incident_mid_gls_relative_drop_15",
        "incident_ef_absolute_drop_5",
    ]
    audit_main = audit[audit["outcome"].isin(primary_outcomes)]
    metric_main = metrics[metrics["outcome"].isin(primary_outcomes)][
        ["outcome", "model", "n", "events", "roc_auc", "average_precision", "sensitivity_top20pct", "precision_top20pct"]
    ]
    primary_delta = deltas[deltas["outcome"].isin(["mid_gls_relative_drop_15", "incident_mid_gls_relative_drop_15", "ef_absolute_drop_5"])]
    future_delta = continuous_deltas[continuous_deltas["outcome"].isin(["future_mid_relative_decline", "future_ef_absolute_decline"])]
    top_associations = associations.nsmallest(12, "p_value")[
        ["outcome", "feature_family", "feature", "n", "partial_spearman_rho", "ci_low", "ci_high", "p_value", "fdr_q"]
    ]
    top_stability = stability.head(15)[
        ["feature_family", "feature", "selection_fraction", "sign_consistency_when_selected", "coefficient_median", "importance_score"]
    ]

    p15 = metrics[metrics["outcome"] == "mid_gls_relative_drop_15"].sort_values("average_precision", ascending=False)
    best = p15.iloc[0]
    clinical = p15[p15["model"] == "clinical_ridge"].iloc[0]
    tm_delta = primary_delta[(primary_delta["outcome"] == "mid_gls_relative_drop_15") & (primary_delta["model"] == "clinical_plus_transmural")]
    var_delta = primary_delta[(primary_delta["outcome"] == "mid_gls_relative_drop_15") & (primary_delta["model"] == "clinical_plus_variability")]

    def delta_sentence(frame: pd.DataFrame, label: str) -> str:
        if frame.empty:
            return f"{label}: not estimable."
        row = frame.iloc[0]
        return (
            f"{label}: delta AP {row['delta_average_precision']:+.3f} "
            f"(95% bootstrap CI {row['delta_average_precision_ci_low']:+.3f} to {row['delta_average_precision_ci_high']:+.3f}); "
            f"delta ROC AUC {row['delta_roc_auc']:+.3f} "
            f"({row['delta_roc_auc_ci_low']:+.3f} to {row['delta_roc_auc_ci_high']:+.3f})."
        )

    tm_supported = not tm_delta.empty and tm_delta.iloc[0]["delta_average_precision_ci_low"] > 0
    var_supported = not var_delta.empty and var_delta.iloc[0]["delta_average_precision_ci_low"] > 0
    if tm_supported or var_supported:
        bottom_line = "At least one proposed feature family showed a bootstrap-stable incremental ranking gain for the primary threshold, but external validation is still required."
    else:
        bottom_line = "Neither proposed feature family showed a bootstrap-stable incremental gain over the early clinical trajectory for the primary threshold; any apparent gains are exploratory."

    incident_rows = metrics[
        (metrics["outcome"] == "incident_mid_gls_relative_drop_15")
        & metrics["model"].isin(["clinical_ridge", "clinical_plus_transmural"])
    ].set_index("model")
    repeatability_rows = metrics[
        (metrics["outcome"] == "mid_gls_absolute_drop_2p9")
        & metrics["model"].isin(["clinical_ridge", "combined_extra_trees"])
    ].set_index("model")
    incident_lead = (
        f"For incident 15% Mid-GLS decline after visit 2, clinical plus Endo-Mid features reached ROC AUC "
        f"{incident_rows.loc['clinical_plus_transmural', 'roc_auc']:.3f} and AP {incident_rows.loc['clinical_plus_transmural', 'average_precision']:.3f}, "
        f"versus {incident_rows.loc['clinical_ridge', 'roc_auc']:.3f} and {incident_rows.loc['clinical_ridge', 'average_precision']:.3f} for clinical ridge."
    )
    repeatability_lead = (
        f"For the 2.9-point absolute Mid-GLS drop, combined Extra Trees reached ROC AUC "
        f"{repeatability_rows.loc['combined_extra_trees', 'roc_auc']:.3f} and AP {repeatability_rows.loc['combined_extra_trees', 'average_precision']:.3f}, "
        f"versus {repeatability_rows.loc['clinical_ridge', 'roc_auc']:.3f} and {repeatability_rows.loc['clinical_ridge', 'average_precision']:.3f}."
    )
    future_all_worse = bool((future_delta["delta_rmse"] > 0).all())
    fdr_hits = int((associations["fdr_q"] < 0.10).sum())
    ef10_lt50 = int(
        (
            (landmarks["final_ef_absolute_decline"] >= 10)
            & ((landmarks["v1_ef"] - landmarks["final_ef_absolute_decline"]) < 50)
        ).fillna(False).sum()
    )

    report = f"""# Early detection of final-visit cardiotoxicity deterioration

## Bottom line

{bottom_line}

For the primary final Mid-GLS relative decline threshold of 15%, the best repeated nested-CV model was **{best['model']}** (ROC AUC {best['roc_auc']:.3f}, average precision {best['average_precision']:.3f}) versus the early clinical ridge (ROC AUC {clinical['roc_auc']:.3f}, average precision {clinical['average_precision']:.3f}).

{delta_sentence(tm_delta, 'Endo-Mid curve-difference features added to the clinical trajectory')}

{delta_sentence(var_delta, 'Inter-segment variability features added to the clinical trajectory')}

This is a small, retrospective, internally validated exploratory analysis—not a deployable clinical model.

## Study design

- Source: 416 anonymized AutoStrainCap exports, reconstructed as 400 true visits in 103 patients.
- Landmark cohort: {len(landmarks)} patients with at least three true visits.
- Predictor window: visit 1 plus visit 2 only. The outcome is the last available true visit.
- Median visit-2 landmark: {landmarks['days_v1_v2'].median():.1f} days after baseline; median remaining prediction horizon: {landmarks['days_v2_final'].median():.1f} days.
- “Incident” analyses exclude patients who already crossed the deterioration threshold by visit 2.
- Mid is the software's mid-myocardial layer; Mid-GLS is primary, with Endo-GLS and EF sensitivity endpoints.
- Technical reanalyses were averaged within Study UID before longitudinal modeling.
- Near-zero segment amplitudes were excluded from Endo/Mid ratios; all linear models used fold-local robust scaling and bounded transformed values so one tracking failure could not dominate a held-out prediction.

## Threshold audit

{markdown_table(audit_main, ['outcome', 'eligible_patients', 'events', 'event_rate'])}

The 20% Mid-GLS and 10-point EF thresholds have few events and should be read as stress tests, not stable model-development endpoints.

The 15% relative GLS threshold was chosen because it is used in the [2022 ESC cardio-oncology definitions](https://academic.oup.com/ehjcimaging/article/23/10/e333/6675075), but those definitions combine GLS with LVEF and/or biomarkers. This dataset lacks biomarkers, symptoms, treatment exposure, and adjudication, so every label here means **imaging deterioration**, not diagnosed cancer therapy-related cardiac dysfunction. The EF 5/7/10-point labels are exploratory; only {ef10_lt50} landmark patients had both an EF decline of at least 10 points and final EF below 50%.

## Signals worth independent follow-up

- {incident_lead} Its paired ranking improvements did not exclude zero, so this is the most interesting Endo-Mid lead—not confirmation.
- {repeatability_lead} Ranking intervals also crossed zero; the Brier-score improvement was borderline/stable and needs replication.
- No candidate feature survived FDR at q<0.10 ({fdr_hits} discoveries). {'Every feature-extended model had worse point-estimate RMSE than clinical ridge for continuous future GLS and EF decline.' if future_all_worse else 'No feature family consistently reduced continuous future-decline error.'}

## Approaches tested

1. A low-dimensional clinical ridge using baseline and visit-2 Mid/Endo GLS, EF, early changes, and time to visit 2.
2. Sparse L1-logistic models using direct paired Endo-Mid segment-curve features: transmural amplitude and timing gaps, curve/shape distance, layer coherence, phase lag, and fixed DCT waveform coefficients.
3. Sparse L1-logistic models using inter-segment heterogeneity: robust peak dispersion, normalized time-to-peak dispersion, curve/shape incoherence, spatial roughness, regional gradients, and their visit-1 to visit-2 changes.
4. Combined sparse-logistic and constrained Extra Trees models to test multivariable and nonlinear signals.
5. Threshold-independent ridge/Extra Trees regression of continuous final and post-visit-2 GLS/EF decline.
6. Clinical-adjusted partial Spearman screens and bootstrap sparse-logistic coefficient stability for interpretation.

All classification scores are averages of repeated patient-held-out outer folds. Logistic regularization was chosen inside each training fold. Model comparison uses paired patient bootstrap intervals.

## Classification results

{markdown_table(metric_main, ['outcome', 'model', 'n', 'events', 'roc_auc', 'average_precision', 'sensitivity_top20pct', 'precision_top20pct'])}

### Incremental performance at the most useful endpoints

{markdown_table(primary_delta, ['outcome', 'model', 'delta_roc_auc', 'delta_roc_auc_ci_low', 'delta_roc_auc_ci_high', 'delta_average_precision', 'delta_average_precision_ci_low', 'delta_average_precision_ci_high'])}

Positive deltas favor the proposed model except Brier-score deltas, where negative is better. Top-20% sensitivity/precision is an alert-budget summary, not a tuned clinical operating point.

## Threshold-independent prediction

{markdown_table(continuous_metrics, ['outcome', 'model', 'n', 'rmse', 'mae', 'spearman_rho'])}

### Incremental prediction after visit 2

{markdown_table(future_delta, ['outcome', 'model', 'delta_rmse', 'delta_rmse_ci_low', 'delta_rmse_ci_high', 'delta_spearman_rho', 'delta_spearman_rho_ci_low', 'delta_spearman_rho_ci_high'])}

Negative delta RMSE is better. These outcomes ask the hardest and cleanest question: after accounting for measurements already available at visit 2, do the curve hypotheses predict further deterioration?

## Clinical-adjusted feature screen

Each candidate feature and outcome was separately residualized for the full early clinical trajectory, then correlated. FDR is across all Endo-Mid and variability candidates for each outcome.

{markdown_table(top_associations, ['outcome', 'feature_family', 'feature', 'n', 'partial_spearman_rho', 'ci_low', 'ci_high', 'p_value', 'fdr_q'])}

## Primary-model coefficient stability

The combined sparse logistic model was refit in class-stratified bootstrap samples. Correlated features can substitute for one another, so selection fractions are more informative than any single full-cohort coefficient.

{markdown_table(top_stability, ['feature_family', 'feature', 'selection_fraction', 'sign_consistency_when_selected', 'coefficient_median', 'importance_score'])}

## Interpretation and next modeling direction

- A feature family is considered supported only if it improves patient-held-out ranking and continuous future-decline prediction, with paired intervals that do not comfortably include no improvement.
- If direct Endo-Mid waveform summaries help but scalar gaps do not, the next model should be a segment-aware functional mixed model rather than a larger black-box CNN.
- If variability helps mainly through visit-1 to visit-2 changes, the next design should model segment-specific random slopes and persistent abnormal segments, with technical-reanalysis variance explicitly separated from biological change.
- If neither survives the incident and continuous analyses, the dataset is likely too small/noisy for supervised curve modeling; the highest-value next step is more patients, reproducible reprocessing, and an independently adjudicated endpoint.

## Limitations

- Only {len(landmarks)} landmark patients and no external test cohort.
- Many correlated candidate features and several exploratory endpoints; FDR is applied only to the univariate screen, not to the model-comparison table.
- The final visit occurs at varying follow-up times; time to the final visit was deliberately not used as an early predictor because it is not generally known at the landmark.
- EF is missing for some visits and its largest-drop threshold has very few events.
- Strain and EF are analysis outputs, not adjudicated clinical cardiotoxicity. Treatment, dose, biomarkers, symptoms, and comorbidities are unavailable.
- Internal technical reanalysis variation is large enough to explain some moderate patient-level changes.

## Files

- `landmark_patient_features.parquet`: patient-level predictors/outcomes.
- `label_audit.csv`: threshold prevalence and eligible populations.
- `classification_metrics.csv`, `classification_predictions.parquet`, `classification_deltas_vs_clinical.csv`.
- `continuous_metrics.csv`, `continuous_predictions.parquet`, `continuous_deltas_vs_clinical.csv`.
- `partial_feature_associations.csv`, `primary_sparse_logistic_stability.csv`, `feature_manifest.csv`.
- `figures/`: threshold, model comparison, alert-budget, and continuous-outcome figures.
"""
    (output_dir / "early_detection_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    visits = pd.read_parquet(args.visits.resolve())
    transmural = curve_feature_table(args.curves.resolve())
    landmarks, feature_sets = build_landmarks(visits, transmural)
    specs = outcome_specs()

    landmarks.to_parquet(output_dir / "landmark_patient_features.parquet", index=False)
    transmural.to_parquet(output_dir / "visit_transmural_features.parquet", index=False)
    manifest_rows = []
    for family, columns in feature_sets.items():
        for column in columns:
            manifest_rows.append({"feature_set": family, "feature": column})
    pd.DataFrame(manifest_rows).to_csv(output_dir / "feature_manifest.csv", index=False)

    audit, metrics, predictions, deltas = classification_study(
        landmarks,
        feature_sets,
        specs,
        repeats=args.cv_repeats,
        bootstraps=args.bootstraps,
        seed=args.seed,
    )
    audit.to_csv(output_dir / "label_audit.csv", index=False)
    metrics.to_csv(output_dir / "classification_metrics.csv", index=False)
    predictions.to_parquet(output_dir / "classification_predictions.parquet", index=False)
    deltas.to_csv(output_dir / "classification_deltas_vs_clinical.csv", index=False)

    continuous_metrics, continuous_predictions, continuous_deltas = continuous_study(
        landmarks,
        feature_sets,
        repeats=args.cv_repeats,
        bootstraps=args.bootstraps,
        seed=args.seed,
    )
    continuous_metrics.to_csv(output_dir / "continuous_metrics.csv", index=False)
    continuous_predictions.to_parquet(output_dir / "continuous_predictions.parquet", index=False)
    continuous_deltas.to_csv(output_dir / "continuous_deltas_vs_clinical.csv", index=False)

    associations = partial_association_study(
        landmarks,
        feature_sets,
        bootstraps=args.bootstraps,
        seed=args.seed,
    )
    associations.to_csv(output_dir / "partial_feature_associations.csv", index=False)
    stability = coefficient_stability(
        landmarks,
        feature_sets,
        bootstraps=args.stability_bootstraps,
        seed=args.seed,
    )
    stability.to_csv(output_dir / "primary_sparse_logistic_stability.csv", index=False)

    make_plots(output_dir, audit, metrics, predictions, continuous_deltas)
    write_report(
        output_dir,
        landmarks,
        audit,
        metrics,
        deltas,
        continuous_metrics,
        continuous_deltas,
        associations,
        stability,
    )

    raw_files = sorted(args.raw_dir.resolve().glob("*.txt")) if args.raw_dir.exists() else []
    metadata = {
        "raw_directory": str(args.raw_dir.resolve()),
        "raw_txt_files": len(raw_files),
        "visits_path": str(args.visits.resolve()),
        "visits_sha256": sha256_file(args.visits.resolve()),
        "curves_path": str(args.curves.resolve()),
        "curves_sha256": sha256_file(args.curves.resolve()),
        "patients_in_visit_table": int(visits["patient_id"].nunique()),
        "true_visits": int(visits["visit_id"].nunique()),
        "landmark_patients": len(landmarks),
        "transmural_visit_feature_rows": len(transmural),
        "clinical_feature_count": len(feature_sets["clinical"]),
        "transmural_feature_count": len(feature_sets["transmural"]),
        "variability_feature_count": len(feature_sets["variability"]),
        "classification_outcomes": len(specs),
        "feature_quality_control": {
            "minimum_segment_peak_for_ratio": 3.0,
            "peak_ratio_clip": [0.0, 5.0],
            "linear_preprocessing": "training-fold median imputation, RobustScaler(10th-90th percentile), transformed-value clip [-5, 5]",
        },
        "cv_repeats": args.cv_repeats,
        "bootstraps": args.bootstraps,
        "stability_bootstraps": args.stability_bootstraps,
        "seed": args.seed,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
