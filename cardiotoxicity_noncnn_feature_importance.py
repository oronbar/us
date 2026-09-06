from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

import cardiotoxicity_next_visit_gpu as core


TASK = "mid_first_rel15"
MODEL_SPECS = [
    ("clinical_ridge", "clinical", "l2"),
    ("clinical_plus_transmural_sparse", "clinical_plus_transmural", "l1"),
    ("clinical_plus_variability_sparse", "clinical_plus_variability", "l1"),
    ("combined_extra_trees", "combined", "trees"),
]
MODEL_LABELS = {
    "clinical_ridge": "Clinical ridge",
    "clinical_plus_transmural_sparse": "Clinical + Endo–Mid",
    "clinical_plus_variability_sparse": "Clinical + variability",
    "combined_extra_trees": "Combined Extra Trees",
}
CLINICAL_DESCRIPTIONS = {
    "history_visits": "Number of visits observed so far",
    "days_since_first": "Days since the baseline visit",
    "days_since_previous": "Days since the previous visit",
    "has_previous_visit": "Whether a prior visit is available",
    "current_mid_gls": "Current Mid-GLS magnitude",
    "current_endo_gls": "Current Endo-GLS magnitude",
    "current_ef": "Current ejection fraction",
    "first_mid_gls": "Baseline Mid-GLS magnitude",
    "first_endo_gls": "Baseline Endo-GLS magnitude",
    "first_ef": "Baseline ejection fraction",
    "current_mid_decline_from_first": "Current relative Mid-GLS decline from baseline",
    "current_endo_decline_from_first": "Current relative Endo-GLS decline from baseline",
    "current_ef_decline_from_first": "Current relative EF decline from baseline",
    "last_mid_relative_change": "Most recent relative Mid-GLS change",
    "last_endo_relative_change": "Most recent relative Endo-GLS change",
    "last_ef_relative_change": "Most recent relative EF change",
    "mid_decline_slope_per_100d": "Mid-GLS decline slope per 100 days",
    "endo_decline_slope_per_100d": "Endo-GLS decline slope per 100 days",
    "ef_decline_slope_per_100d": "EF decline slope per 100 days",
    "current_endo_mid_gap": "Current Endo minus Mid GLS gap",
    "last_endo_mid_gap_change": "Change in the Endo–Mid GLS gap",
    "current_mid_decline_from_roll2": "Current Mid-GLS decline from prior-2-visit mean",
    "current_endo_decline_from_roll2": "Current Endo-GLS decline from prior-2-visit mean",
    "current_ef_decline_from_roll2": "Current EF decline from prior-2-visit mean",
    "current_mid_decline_from_roll3": "Current Mid-GLS decline from prior-3-visit mean",
    "current_endo_decline_from_roll3": "Current Endo-GLS decline from prior-3-visit mean",
    "current_ef_decline_from_roll3": "Current EF decline from prior-3-visit mean",
}
VARIABILITY_DESCRIPTIONS = {
    "peak_abs_robust_sd": "robust dispersion of peak amplitude",
    "peak_abs_cv": "coefficient of variation of peak amplitude",
    "time_to_peak_norm_circular_std": "circular dispersion of time to peak",
    "vendor_peak_systolic_abs_robust_sd": "vendor peak-systolic amplitude dispersion",
    "vendor_time_to_peak_norm_circular_std": "vendor time-to-peak dispersion",
    "post_systolic_fraction": "fraction of post-systolic segments",
    "curve_dispersion_rms": "RMS dispersion of segment curves",
    "curve_pairwise_rmse": "pairwise segment-curve RMSE",
    "curve_integrated_robust_mad": "integrated robust curve dispersion",
    "shape_dispersion_rms": "RMS dispersion of normalized curve shape",
    "shape_pairwise_rmse": "pairwise normalized-shape RMSE",
    "shape_incoherence": "segment shape incoherence",
    "within_view_peak_robust_sd_mean": "mean within-view peak dispersion",
    "within_ring_peak_robust_sd_mean": "mean within-ring peak dispersion",
    "spatial_peak_graph_roughness": "spatial roughness of peak amplitude",
    "spatial_timing_graph_roughness": "spatial roughness of timing",
    "impaired_segment_fraction_lt15": "fraction of segments with strain magnitude below 15",
    "apical_basal_peak_gradient": "apical-to-basal peak gradient",
}

FEATURE_PLAIN_NAMES = {
    "first_endo_gls": "Baseline Endocardial GLS magnitude",
    "first_mid_gls": "Baseline Mid-wall GLS magnitude",
    "current_mid_decline_from_first": "Current relative Mid-GLS decline from baseline",
    "current_endo_decline_from_first": "Current relative Endo-GLS decline from baseline",
    "last_ef_relative_change": "Most recent relative EF change",
    "d_tm_sd_gap_dct04": (
        "Change in phase pattern of segmental layer-gap variability"
    ),
    "cur_tm_segment_curve_correlation_std": (
        "Variation across segments in Endo/Mid curve similarity"
    ),
    "d_tm_peak_gap_mean": "Change in the mean Endo–Mid peak-strain gap",
    "cur_tm_sd_gap_dct06": (
        "Current finer phase pattern of segmental Endo–Mid gap variability"
    ),
    "cur_tm_mean_gap_dct02": "Broad current temporal trend in the Endo–Mid curve gap",
    "cur_tm_mean_shape_gap_dct01": (
        "Overall normalized Endo–Mid shape separation"
    ),
    "cur_tm_mean_shape_gap_dct04": (
        "Intermediate phase pattern of normalized Endo–Mid shape separation"
    ),
    "cur_tm_mean_shape_gap_dct07": (
        "Finer temporal pattern of normalized shape separation"
    ),
    "cur_var__mid_peak_abs_robust_sd": (
        "Robust between-segment dispersion of Mid-wall peak strain"
    ),
    "cur_var__endo_vendor_peak_systolic_abs_robust_sd": (
        "Robust between-segment dispersion of vendor Endo peak-systolic strain"
    ),
    "cur_var__endo_vendor_time_to_peak_norm_circular_std": (
        "Endocardial segment time-to-peak dispersion"
    ),
    "cur_var__endo_peak_abs_robust_sd": (
        "Robust between-segment dispersion of Endocardial peak strain"
    ),
    "d_tm_vendor_peak_gap_mean": (
        "Change in mean vendor Endo–Mid peak gap"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patient-held-out feature importance for the non-CNN cardiotoxicity models."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("cardiotoxicity_next_visit_gpu_results"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cardiotoxicity_feature_importance_results"),
    )
    parser.add_argument("--screen-repeats", type=int, default=0)
    parser.add_argument("--refined-repeats", type=int, default=30)
    parser.add_argument("--bootstraps", type=int, default=500)
    parser.add_argument("--top-candidates", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild the Markdown report from existing result tables.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Rebuild plots from existing feature-importance result tables.",
    )
    return parser.parse_args()


def feature_family(feature: str) -> str:
    if feature.startswith(("cur_tm_", "d_tm_")):
        return "Endo–Mid"
    if feature.startswith(("cur_var__", "d_var__")):
        return "variability"
    return "clinical"


def plain_phrase(value: str) -> str:
    replacements = {
        "ttp": "time-to-peak",
        "dct": "DCT",
        "rms": "RMS",
        "rmse": "RMSE",
        "iqr": "IQR",
        "p90 p10": "P90–P10",
    }
    phrase = value.replace("_", " ")
    for old, new in replacements.items():
        phrase = phrase.replace(old, new)
    return phrase


def feature_description(feature: str) -> str:
    if feature in CLINICAL_DESCRIPTIONS:
        return CLINICAL_DESCRIPTIONS[feature]
    if feature.startswith(("cur_var__", "d_var__")):
        timing = "Current visit" if feature.startswith("cur_") else "Change since previous visit"
        body = feature.split("__", 1)[1]
        layer = "Endo" if body.startswith("endo_") else "Mid"
        suffix = body[len(layer.lower()) + 1 :]
        phrase = VARIABILITY_DESCRIPTIONS.get(suffix, plain_phrase(suffix))
        return f"{timing}, {layer}: {phrase}"
    if feature.startswith(("cur_tm_", "d_tm_")):
        timing = "Current visit" if feature.startswith("cur_") else "Change since previous visit"
        body = feature[7:] if feature.startswith("cur_tm_") else feature[5:]
        return f"{timing}: Endo–Mid {plain_phrase(body)}"
    return plain_phrase(feature)


def plain_feature_name(feature: str) -> str:
    return FEATURE_PLAIN_NAMES.get(feature, feature_description(feature))


def load_feature_sets(manifest: pd.DataFrame) -> dict[str, list[str]]:
    return {
        name: group["feature"].tolist()
        for name, group in manifest.groupby("feature_set", sort=False)
    }


def load_splits(
    assignments: pd.DataFrame, patients: set[str]
) -> list[dict[str, object]]:
    splits = []
    for (repeat, fold), group in assignments.groupby(["repeat", "fold"], sort=True):
        test_patients = set(group["patient_id"].astype(str))
        splits.append(
            {
                "split_index": int(repeat) * 5 + int(fold),
                "repeat": int(repeat),
                "fold": int(fold),
                "train_patients": patients - test_patients,
                "test_patients": test_patients,
            }
        )
    return splits


def build_model(kind: str, random_state: int) -> Pipeline:
    if kind != "trees":
        return core.robust_linear_pipeline(kind, random_state)
    return Pipeline(
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
                    random_state=random_state,
                ),
            ),
        ]
    )


def safe_metric(metric, y: np.ndarray, score: np.ndarray) -> float:
    if np.unique(y).size < 2:
        return np.nan
    return float(metric(y, score))


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def cluster_bootstrap_drops(
    transitions: pd.DataFrame,
    valid_index: np.ndarray,
    y: np.ndarray,
    base_score: np.ndarray,
    permuted_score: np.ndarray,
    bootstraps: int,
    seed: int,
) -> dict[str, float]:
    patient_values = transitions.iloc[valid_index]["patient_id"].astype(str).to_numpy()
    unique_patients = np.unique(patient_values)
    patient_rows = {
        patient: np.flatnonzero(patient_values == patient) for patient in unique_patients
    }
    rng = np.random.default_rng(seed)
    auc_drops = []
    ap_drops = []
    for _ in range(bootstraps):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        index = np.concatenate([patient_rows[patient] for patient in sampled])
        y_boot = y[index]
        if np.unique(y_boot).size < 2:
            continue
        auc_drops.append(
            roc_auc_score(y_boot, base_score[index])
            - roc_auc_score(y_boot, permuted_score[index])
        )
        ap_drops.append(
            average_precision_score(y_boot, base_score[index])
            - average_precision_score(y_boot, permuted_score[index])
        )
    return {
        "auc_drop_boot_low": float(np.quantile(auc_drops, 0.025)),
        "auc_drop_boot_high": float(np.quantile(auc_drops, 0.975)),
        "ap_drop_boot_low": float(np.quantile(ap_drops, 0.025)),
        "ap_drop_boot_high": float(np.quantile(ap_drops, 0.975)),
    }


def screen_models(
    transitions: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    splits: list[dict[str, object]],
    active_task_index: int,
    seed: int,
    repeats: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, object]]], pd.DataFrame]:
    task_mask = transitions[f"mask__{TASK}"].astype(bool).to_numpy()
    label_column = f"label__{TASK}"
    artifacts: dict[str, list[dict[str, object]]] = {
        model_name: [] for model_name, _, _ in MODEL_SPECS
    }
    screen_rows: list[dict[str, object]] = []
    base_score_sum = {
        model_name: np.zeros(len(transitions), dtype=float)
        for model_name, _, _ in MODEL_SPECS
    }
    base_score_count = {
        model_name: np.zeros(len(transitions), dtype=int)
        for model_name, _, _ in MODEL_SPECS
    }

    for split in splits:
        train_patient = transitions["patient_id"].isin(split["train_patients"]).to_numpy()
        test_patient = transitions["patient_id"].isin(split["test_patients"]).to_numpy()
        train_index = np.flatnonzero(train_patient & task_mask)
        test_index = np.flatnonzero(test_patient & task_mask)
        y_train = transitions.iloc[train_index][label_column].to_numpy(int)
        y_test = transitions.iloc[test_index][label_column].to_numpy(int)
        for model_index, (model_name, feature_set, kind) in enumerate(MODEL_SPECS):
            all_columns = feature_sets[feature_set]
            columns = core.usable_features(transitions.iloc[train_index], all_columns)
            random_state = (
                seed
                + int(split["split_index"]) * 100
                + active_task_index * 10
                + model_index
            )
            model = build_model(kind, random_state)
            x_train = transitions.iloc[train_index][columns].astype(float)
            x_test = transitions.iloc[test_index][columns].astype(float)
            model.fit(x_train, y_train)
            base_score = model.predict_proba(x_test)[:, 1]
            base_score_sum[model_name][test_index] += base_score
            base_score_count[model_name][test_index] += 1

            if kind == "trees":
                native_values = model.named_steps["model"].feature_importances_
                coefficients = np.full(len(columns), np.nan)
                model.named_steps["model"].n_jobs = 1
            else:
                coefficients = model.named_steps["model"].coef_[0]
                native_values = np.abs(coefficients)

            if repeats > 0:
                scoring = {
                    "roc_auc": "roc_auc",
                    "average_precision": "average_precision",
                }
                result = permutation_importance(
                    model,
                    x_test,
                    y_test,
                    scoring=scoring,
                    n_repeats=repeats,
                    random_state=stable_seed(
                        seed, "screen", model_name, split["repeat"], split["fold"]
                    ),
                    n_jobs=n_jobs,
                )
                auc_importance = result["roc_auc"].importances
                ap_importance = result["average_precision"].importances
            else:
                auc_importance = np.zeros((len(columns), 1), dtype=float)
                ap_importance = np.zeros((len(columns), 1), dtype=float)
            column_index = {column: index for index, column in enumerate(columns)}
            for feature in all_columns:
                if feature in column_index:
                    index = column_index[feature]
                    auc_drop = float(np.mean(auc_importance[index]))
                    ap_drop = float(np.mean(ap_importance[index]))
                    native = float(native_values[index])
                    coefficient = float(coefficients[index])
                    used = True
                else:
                    auc_drop = 0.0
                    ap_drop = 0.0
                    native = 0.0
                    coefficient = 0.0 if kind != "trees" else np.nan
                    used = False
                screen_rows.append(
                    {
                        "model": model_name,
                        "feature": feature,
                        "feature_family": feature_family(feature),
                        "description": feature_description(feature),
                        "repeat": split["repeat"],
                        "fold": split["fold"],
                        "test_rows": len(test_index),
                        "used": used,
                        "auc_drop": auc_drop,
                        "ap_drop": ap_drop,
                        "native_importance": native,
                        "coefficient": coefficient,
                    }
                )
            artifacts[model_name].append(
                {
                    "split": split,
                    "model": model,
                    "kind": kind,
                    "columns": columns,
                    "test_index": test_index,
                    "x_test": x_test.reset_index(drop=True),
                    "y_test": y_test,
                    "base_score": base_score,
                }
            )

    base_rows = []
    valid_index = np.flatnonzero(task_mask)
    y = transitions.iloc[valid_index][label_column].to_numpy(int)
    for model_name, _, _ in MODEL_SPECS:
        score = (
            base_score_sum[model_name][valid_index]
            / base_score_count[model_name][valid_index]
        )
        base_rows.append(
            {
                "model": model_name,
                "n": len(valid_index),
                "events": int(y.sum()),
                "roc_auc_reproduced": roc_auc_score(y, score),
                "average_precision_reproduced": average_precision_score(y, score),
                "prediction_repeats_min": int(
                    base_score_count[model_name][valid_index].min()
                ),
                "prediction_repeats_max": int(
                    base_score_count[model_name][valid_index].max()
                ),
            }
        )
    return pd.DataFrame(screen_rows), artifacts, pd.DataFrame(base_rows)


def summarize_screen(screen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, feature), group in screen.groupby(["model", "feature"], sort=False):
        coefficients = group["coefficient"].dropna().to_numpy(float)
        selected = np.abs(coefficients) > 1e-10
        selected_coefficients = coefficients[selected]
        if len(selected_coefficients):
            sign_consistency = max(
                np.mean(selected_coefficients > 0),
                np.mean(selected_coefficients < 0),
            )
            coefficient_median = float(np.median(selected_coefficients))
        else:
            sign_consistency = np.nan
            coefficient_median = np.nan
        rows.append(
            {
                "model": model,
                "feature": feature,
                "feature_family": group["feature_family"].iloc[0],
                "description": group["description"].iloc[0],
                "used_fraction": float(group["used"].mean()),
                "auc_drop_screen": float(group["auc_drop"].mean()),
                "auc_drop_split_sd": float(group["auc_drop"].std(ddof=1)),
                "auc_positive_split_fraction": float((group["auc_drop"] > 0).mean()),
                "ap_drop_screen": float(group["ap_drop"].mean()),
                "ap_drop_split_sd": float(group["ap_drop"].std(ddof=1)),
                "ap_positive_split_fraction": float((group["ap_drop"] > 0).mean()),
                "native_importance_mean": float(group["native_importance"].mean()),
                "selection_fraction": (
                    float(selected.mean()) if len(coefficients) else np.nan
                ),
                "coefficient_median_when_selected": coefficient_median,
                "coefficient_sign_consistency": sign_consistency,
            }
        )
    summary = pd.DataFrame(rows)
    summary["auc_rank_screen"] = summary.groupby("model")[
        "auc_drop_screen"
    ].rank(method="min", ascending=False)
    summary["ap_rank_screen"] = summary.groupby("model")[
        "ap_drop_screen"
    ].rank(method="min", ascending=False)
    summary["native_rank"] = summary.groupby("model")[
        "native_importance_mean"
    ].rank(method="min", ascending=False)
    return summary


def candidate_features(
    screen_summary: pd.DataFrame, top_candidates: int
) -> dict[str, list[str]]:
    result = {}
    for model, group in screen_summary.groupby("model", sort=False):
        if group["auc_drop_screen"].abs().sum() > 0:
            candidate = group[
                (group["auc_rank_screen"] <= 18)
                | (group["ap_rank_screen"] <= 18)
                | (group["native_rank"] <= 15)
            ].copy()
            candidate["best_rank"] = candidate[
                ["auc_rank_screen", "ap_rank_screen", "native_rank"]
            ].min(axis=1)
            candidate["rank_sum"] = candidate[
                ["auc_rank_screen", "ap_rank_screen", "native_rank"]
            ].sum(axis=1)
            candidate = candidate.sort_values(
                ["best_rank", "rank_sum", "auc_drop_screen"],
                ascending=[True, True, False],
            ).head(top_candidates)
        else:
            candidate = group.nsmallest(
                min(top_candidates, len(group)), "native_rank"
            )
        result[model] = candidate["feature"].tolist()
    return result


def refined_permutation(
    transitions: pd.DataFrame,
    artifacts: dict[str, list[dict[str, object]]],
    candidates: dict[str, list[str]],
    repeats: int,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_index = np.flatnonzero(transitions[f"mask__{TASK}"].astype(bool).to_numpy())
    y = transitions.iloc[valid_index][f"label__{TASK}"].to_numpy(int)
    refined_rows = []
    group_rows = []

    for model_name, _, _ in MODEL_SPECS:
        model_artifacts = artifacts[model_name]
        base_sum = np.zeros(len(transitions), dtype=float)
        base_count = np.zeros(len(transitions), dtype=int)
        for artifact in model_artifacts:
            test_index = artifact["test_index"]
            base_sum[test_index] += artifact["base_score"]
            base_count[test_index] += 1
        base_score = base_sum[valid_index] / base_count[valid_index]
        base_auc = roc_auc_score(y, base_score)
        base_ap = average_precision_score(y, base_score)

        def evaluate_permutation(
            name: str, permuted_columns: list[str], item_type: str
        ) -> dict[str, object]:
            permuted_sum = np.zeros((repeats, len(transitions)), dtype=float)
            permuted_count = np.zeros(len(transitions), dtype=int)
            for artifact in model_artifacts:
                test_index = artifact["test_index"]
                x_test = artifact["x_test"]
                active_columns = [
                    column for column in permuted_columns if column in artifact["columns"]
                ]
                permuted_count[test_index] += 1
                if not active_columns:
                    permuted_sum[:, test_index] += artifact["base_score"][None, :]
                    continue
                permuted_frames = []
                for repeat in range(repeats):
                    rng = np.random.default_rng(
                        stable_seed(
                            seed,
                            "refined",
                            model_name,
                            name,
                            artifact["split"]["repeat"],
                            artifact["split"]["fold"],
                            repeat,
                        )
                    )
                    order = rng.permutation(len(x_test))
                    x_permuted = x_test.copy()
                    x_permuted.loc[:, active_columns] = (
                        x_test.loc[order, active_columns].to_numpy()
                    )
                    permuted_frames.append(x_permuted)
                stacked = pd.concat(permuted_frames, ignore_index=True)
                prediction = artifact["model"].predict_proba(stacked)[:, 1]
                prediction = prediction.reshape(repeats, len(test_index))
                permuted_sum[:, test_index] += prediction
            permuted_scores = (
                permuted_sum[:, valid_index] / permuted_count[valid_index][None, :]
            )
            auc_drops = np.array(
                [
                    base_auc - roc_auc_score(y, permuted_scores[repeat])
                    for repeat in range(repeats)
                ]
            )
            ap_drops = np.array(
                [
                    base_ap - average_precision_score(y, permuted_scores[repeat])
                    for repeat in range(repeats)
                ]
            )
            mean_permuted_score = permuted_scores.mean(axis=0)
            auc_drop_mean_score = base_auc - roc_auc_score(y, mean_permuted_score)
            ap_drop_mean_score = base_ap - average_precision_score(
                y, mean_permuted_score
            )
            bootstrap = cluster_bootstrap_drops(
                transitions,
                valid_index,
                y,
                base_score,
                mean_permuted_score,
                bootstraps,
                stable_seed(seed, "bootstrap", model_name, name),
            )
            return {
                "model": model_name,
                "item_type": item_type,
                "item": name,
                "base_auc": base_auc,
                "base_ap": base_ap,
                "auc_drop_mean": float(auc_drop_mean_score),
                "auc_drop_across_permutations_mean": float(auc_drops.mean()),
                "auc_drop_permutation_low": float(np.quantile(auc_drops, 0.025)),
                "auc_drop_permutation_high": float(np.quantile(auc_drops, 0.975)),
                "ap_drop_mean": float(ap_drop_mean_score),
                "ap_drop_across_permutations_mean": float(ap_drops.mean()),
                "ap_drop_permutation_low": float(np.quantile(ap_drops, 0.025)),
                "ap_drop_permutation_high": float(np.quantile(ap_drops, 0.975)),
                **bootstrap,
            }

        for feature in candidates[model_name]:
            refined_rows.append(
                evaluate_permutation(feature, [feature], "feature")
            )

        model_columns = {
            column
            for artifact in model_artifacts
            for column in artifact["columns"]
        }
        for family in ["clinical", "Endo–Mid", "variability"]:
            family_columns = [
                column
                for column in model_columns
                if feature_family(column) == family
            ]
            if family_columns:
                group_rows.append(
                    evaluate_permutation(family, family_columns, "family")
                )

    return pd.DataFrame(refined_rows), pd.DataFrame(group_rows)


def add_consensus(refined: pd.DataFrame) -> pd.DataFrame:
    frame = refined.copy()
    frame["rank_within_model"] = frame.groupby("model")["auc_drop_mean"].rank(
        method="min", ascending=False
    )
    max_positive = frame.groupby("model")["auc_drop_mean"].transform(
        lambda values: max(float(values.max()), 1e-9)
    )
    frame["normalized_positive_auc_importance"] = (
        frame["auc_drop_mean"].clip(lower=0) / max_positive
    )
    rows = []
    for feature, group in frame.groupby("item", sort=False):
        rows.append(
            {
                "feature": feature,
                "feature_family": feature_family(feature),
                "description": feature_description(feature),
                "models_evaluated": int(group["model"].nunique()),
                "models_top10": int((group["rank_within_model"] <= 10).sum()),
                "mean_normalized_positive_auc_importance": float(
                    group["normalized_positive_auc_importance"].mean()
                ),
                "best_auc_drop": float(group["auc_drop_mean"].max()),
                "mean_auc_drop": float(group["auc_drop_mean"].mean()),
                "best_ap_drop": float(group["ap_drop_mean"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "models_top10",
            "mean_normalized_positive_auc_importance",
            "best_auc_drop",
        ],
        ascending=False,
    )


def make_plots(
    output: Path,
    top_features: pd.DataFrame,
    group_importance: pd.DataFrame,
) -> None:
    colors = {
        "clinical": "#277da1",
        "Endo–Mid": "#f8961e",
        "variability": "#43aa8b",
    }
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    for axis, (model, group) in zip(
        axes.flat, top_features.groupby("model", sort=False)
    ):
        plot = group.nlargest(10, "auc_drop_mean").sort_values("auc_drop_mean")
        descriptions = [
            textwrap.fill(plain_feature_name(feature), 32)
            for feature in plot["item"]
        ]
        lower = np.maximum(
            plot["auc_drop_mean"].to_numpy()
            - plot["auc_drop_boot_low"].to_numpy(),
            0,
        )
        upper = np.maximum(
            plot["auc_drop_boot_high"].to_numpy()
            - plot["auc_drop_mean"].to_numpy(),
            0,
        )
        axis.barh(
            descriptions,
            plot["auc_drop_mean"],
            xerr=np.vstack([lower, upper]),
            color=[colors[feature_family(feature)] for feature in plot["item"]],
            alpha=0.9,
            error_kw={"elinewidth": 1, "capsize": 2},
        )
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.set_title(MODEL_LABELS[model])
        axis.set_xlabel("Held-out AUC decrease after permutation")
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle(
        "Top non-CNN predictors of next-visit 15% relative Mid-GLS deterioration",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(
        output / "top_feature_permutation_importance.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

    plot = group_importance.copy()
    family_display = {
        "clinical": "Clinical",
        "Endo–Mid": "Endo–Mid",
        "variability": "Variability",
    }
    plot["item"] = plot["item"].map(family_display).fillna(plot["item"])
    display_colors = {
        "Clinical": colors["clinical"],
        "Endo–Mid": colors["Endo–Mid"],
        "Variability": colors["variability"],
    }
    plot["model_label"] = plot["model"].map(MODEL_LABELS)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for axis, metric, label in [
        (axes[0], "auc_drop_mean", "AUC decrease"),
        (axes[1], "ap_drop_mean", "AP decrease"),
    ]:
        pivot = plot.pivot(index="model_label", columns="item", values=metric).fillna(0)
        pivot = pivot.reindex([MODEL_LABELS[name] for name, _, _ in MODEL_SPECS])
        pivot.plot(
            kind="bar",
            ax=axis,
            color=[display_colors.get(column, "#777777") for column in pivot.columns],
        )
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_xlabel("")
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(title="Feature family")
    figure.suptitle(
        "Feature-family dependence in held-out predictions",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(
        output / "feature_family_permutation_importance.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame[columns].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows])


def write_report(
    output: Path,
    base_metrics: pd.DataFrame,
    top_features: pd.DataFrame,
    group_importance: pd.DataFrame,
    consensus: pd.DataFrame,
    screen_summary: pd.DataFrame,
    saved_metrics: pd.DataFrame,
) -> None:
    del screen_summary  # Retained in the signature for backward compatibility.
    metric_rows = saved_metrics[
        saved_metrics["task"].eq(TASK)
        & saved_metrics["model"].isin(MODEL_LABELS)
    ][["model", "roc_auc", "average_precision"]].copy()
    metric_rows["model"] = metric_rows["model"].map(MODEL_LABELS)
    metric_rows = metric_rows.rename(
        columns={"roc_auc": "AUC", "average_precision": "AP"}
    )

    prevalence = float(base_metrics["events"].iloc[0] / base_metrics["n"].iloc[0])

    def feature_row(model_name: str, feature: str) -> pd.Series:
        rows = top_features[
            top_features["model"].eq(model_name)
            & top_features["item"].eq(feature)
        ]
        if rows.empty:
            raise KeyError(f"Missing importance row: {model_name}/{feature}")
        return rows.iloc[0]

    def result_text(row: pd.Series) -> str:
        stable = "yes" if row["auc_drop_boot_low"] > 0 else "no"
        return (
            f"AUC decrease **{row['auc_drop_mean']:.3f}** "
            f"(patient-bootstrap 95% CI {row['auc_drop_boot_low']:.3f} to "
            f"{row['auc_drop_boot_high']:.3f}); AP decrease "
            f"**{row['ap_drop_mean']:.3f}**; CI excludes zero: **{stable}**."
        )

    stable_features = (
        top_features[top_features["auc_drop_boot_low"] > 0]
        .sort_values("auc_drop_mean", ascending=False)
        .drop_duplicates("item")
        .copy()
    )
    stable_features["Plain-language feature"] = stable_features["item"].map(
        plain_feature_name
    )
    stable_features["Technical name"] = stable_features["item"]
    stable_features["Model"] = stable_features["model"].map(MODEL_LABELS)
    stable_features["Family"] = stable_features["feature_family"]
    stable_features["AUC decrease"] = stable_features["auc_drop_mean"]
    stable_features["AUC 95% CI"] = stable_features.apply(
        lambda row: (
            f"{row['auc_drop_boot_low']:.3f} to {row['auc_drop_boot_high']:.3f}"
        ),
        axis=1,
    )
    stable_features["AP decrease"] = stable_features["ap_drop_mean"]

    ranking_sections = []
    for model_name, _, kind in MODEL_SPECS:
        model_rows = top_features[top_features["model"].eq(model_name)].nlargest(
            5, "auc_drop_mean"
        ).copy()
        model_rows["Plain-language feature"] = model_rows["item"].map(
            plain_feature_name
        )
        model_rows["Technical name"] = model_rows["item"]
        model_rows["Family"] = model_rows["feature_family"]
        model_rows["AUC decrease"] = model_rows["auc_drop_mean"]
        model_rows["AUC 95% CI"] = model_rows.apply(
            lambda row: (
                f"{row['auc_drop_boot_low']:.3f} to "
                f"{row['auc_drop_boot_high']:.3f}"
            ),
            axis=1,
        )
        model_rows["AP decrease"] = model_rows["ap_drop_mean"]
        model_rows["Stable"] = np.where(
            model_rows["auc_drop_boot_low"] > 0, "Yes", "No"
        )
        if kind == "trees":
            model_rows["Model direction"] = "Nonlinear / interaction-dependent"
        else:
            model_rows["Model direction"] = np.where(
                model_rows["coefficient_median_when_selected"] >= 0,
                "Higher value -> higher predicted risk",
                "Higher value -> lower predicted risk",
            )
        ranking_sections.append(
            f"### {MODEL_LABELS[model_name]}\n\n"
            + markdown_table(
                model_rows,
                [
                    "Plain-language feature",
                    "Technical name",
                    "Family",
                    "AUC decrease",
                    "AUC 95% CI",
                    "AP decrease",
                    "Stable",
                    "Model direction",
                ],
            )
        )

    group_table = group_importance.copy()
    group_table["model"] = group_table["model"].map(MODEL_LABELS)
    group_table = group_table.rename(
        columns={
            "item": "family",
            "auc_drop_mean": "AUC drop",
            "ap_drop_mean": "AP drop",
            "auc_drop_boot_low": "AUC CI low",
            "auc_drop_boot_high": "AUC CI high",
        }
    )
    del consensus

    clinical_row = feature_row("clinical_ridge", "first_endo_gls")
    transmural_row = feature_row(
        "clinical_plus_transmural_sparse", "d_tm_sd_gap_dct04"
    )
    variability_row = feature_row(
        "clinical_plus_variability_sparse", "first_mid_gls"
    )
    variability_specific_row = feature_row(
        "clinical_plus_variability_sparse",
        "cur_var__mid_peak_abs_robust_sd",
    )
    trees_row = feature_row(
        "combined_extra_trees", "cur_tm_mean_shape_gap_dct01"
    )

    report = f"""# Clear report: feature importance in the non-CNN cardiotoxicity models

## Executive summary

The clearest engineered signal came from the **relationship between the Endocardial
and Mid-wall strain curves**, particularly how that relationship varied across
segments and changed between visits. Inter-segment variability contributed less and
was less stable.

- Strongest engineered feature: **change in the phase pattern of segmental Endo–Mid
  gap variability** (`d_tm_sd_gap_dct04`).
- Strongest Extra Trees feature: **overall normalized Endo–Mid curve-shape
  separation** (`cur_tm_mean_shape_gap_dct01`).
- Baseline GLS remained important, but baseline Mid-GLS is mathematically connected
  to the relative-decline outcome and must be interpreted cautiously.
- Most individual variability features had confidence intervals crossing zero.

## Prediction task and dataset

At every eligible visit, the model uses information available up to that visit to
predict whether the **immediately following visit** will show at least a **15% relative
decline in Mid-GLS from the first-visit baseline**:

`relative decline = 1 - (next-visit Mid-GLS magnitude / first-visit Mid-GLS magnitude)`

An event is recorded when `relative decline >= 0.15`.

- 103 patients
- 238 eligible current-to-next-visit predictions
- 49 deterioration events ({prevalence:.1%})
- 189 non-events ({1.0 - prevalence:.1%})
- All visits from the same patient remain in the same held-out fold.

## Models and performance

{markdown_table(metric_rows, ['model', 'AUC', 'AP'])}

Random guessing has AUC 0.500. Its expected AP equals the event rate, approximately
{prevalence:.3f}.

## How predictive contribution was measured

- The four existing non-CNN models were reconstructed using the original three
  repeated five-fold patient splits and the same preprocessing, regularization, and
  random seeds.
- The strongest candidates were screened using standardized coefficient magnitude for
  the logistic models and tree importance for Extra Trees.
- Final ranking used **held-out permutation importance**: one feature was randomly
  shuffled, the model predicted again, and the decrease in AUC/AP was measured.
- A large positive decrease means the model genuinely depended on that feature in
  held-out patients. A value near zero means the feature was redundant, unused, or too
  noisy. A negative value means shuffling improved performance.
- Confidence intervals resample patients while keeping all transitions belonging to a
  sampled patient together.

## How to read the feature names

| Name component | Meaning |
|---|---|
| `first_` | Value from the patient's first visit, used as the baseline |
| `cur_` | Value at the current visit from which the next visit is predicted |
| `d_` | Current value minus the value at the immediately previous visit |
| `tm_` | Transmural feature comparing matched Endocardial and Mid-wall curves |
| `var__` | Variability between myocardial segments |
| `gap` | Endocardial value minus Mid-wall value |
| `shape_gap` | Endo–Mid difference after each curve is normalized by its own amplitude |
| `DCT01`, `DCT04`, etc. | Coefficients summarizing the complete time-dependent curve pattern |

### What DCT means here

DCT stands for **Discrete Cosine Transform**. It converts a 96-point curve into a
small number of coefficients:

- `DCT01`: the broad overall level or offset of the curve.
- `DCT02`: a slow trend across the cardiac cycle.
- `DCT04`: a more detailed phase-dependent pattern.
- `DCT07`: a still finer temporal pattern.

A DCT coefficient is not a value at one time point. It summarizes a pattern spread
across the cardiac cycle. Higher-numbered coefficients are progressively less
intuitive physiologically and should be interpreted as curve-shape descriptors.

## Most important feature in each model

### 1. Clinical ridge: baseline Endocardial GLS magnitude

**Technical name:** `first_endo_gls`

The source Endocardial GLS is negative. The model uses its positive magnitude:

`first_endo_gls = absolute value of first-visit gls_endo_peak_avg`

For example, a baseline Endo-GLS of `-21.4%` becomes `21.4%`. This value is then
carried forward for every prediction belonging to that patient.

The positive standardized coefficient means that, conditional on the other clinical
trajectory variables, a larger baseline Endo-GLS magnitude produced a higher predicted
risk. This may reflect physiological reserve, correlation with baseline Mid-GLS, or
regression to the mean; it should not be interpreted as evidence that better baseline
function causes cardiotoxicity.

**Importance:** {result_text(clinical_row)}

### 2. Clinical + Endo–Mid sparse model: change in segmental Endo–Mid gap variability

**Technical name:** `d_tm_sd_gap_dct04`

This feature is calculated in six steps:

1. Pair the Endocardial and Mid-wall longitudinal-strain curves from the same segment.
2. At each cardiac-cycle point, calculate the raw layer gap for every segment:
   `segment gap_s(t) = Endo_s(t) - Mid_s(t)`.
3. At each time point, calculate the standard deviation of that gap across the matched
   segments: `SD_gap(t) = SD across segments of segment gap_s(t)`.
4. `SD_gap(t)` is now a complete curve describing *when* segments disagree in their
   Endo–Mid separation.
5. Apply the DCT and retain coefficient 4 (`DCT04`, mathematical index `k=3`).
6. Subtract the previous visit's coefficient from the current visit's coefficient:
   `current DCT04 - previous DCT04`.

In compact form:

`paired Endo/Mid curves -> segment gap curves -> SD across segments -> DCT04 -> current minus previous`

It does **not** simply mean that overall variability increased. It means that a
particular phase-dependent pattern of Endo–Mid heterogeneity changed between visits.
The sparse logistic model selected it in every outer fold, with a positive median
standardized coefficient: an increase in this component raised predicted risk.

**Importance:** {result_text(transmural_row)}

### 3. Clinical + variability sparse model: baseline Mid-wall GLS magnitude

**Technical name:** `first_mid_gls`

The calculation is:

`first_mid_gls = absolute value of first-visit gls_mid_peak_avg`

For example, `-18.2%` becomes `18.2%`. This was the strongest feature in the model
that also contained segment-variability features.

Important caution: the outcome is itself defined relative to this value:

`event if 1 - (next Mid-GLS / first Mid-GLS) >= 0.15`

Therefore, baseline Mid-GLS is mathematically coupled to the label. This is not future
data leakage, because baseline is known at prediction time, but it can amplify
baseline measurement error and regression-to-the-mean effects. Its confidence
interval also crossed zero.

**Importance:** {result_text(variability_row)}

#### Strongest actual variability feature

The strongest variability-specific feature was **robust between-segment dispersion of
current Mid-wall peak strain** (`cur_var__mid_peak_abs_robust_sd`). For segment peak
magnitudes `p_s`, it is calculated as:

`1.4826 x median(|p_s - median(p)|)`

This robust standard deviation is less affected by one badly tracked segment than an
ordinary standard deviation. In this fitted sparse model, its coefficient was
negative, but its AP contribution was slightly negative and its confidence interval
crossed zero. It is therefore not a stable biological finding.

**Importance:** {result_text(variability_specific_row)}

### 4. Combined Extra Trees: overall normalized Endo–Mid curve-shape separation

**Technical name:** `cur_tm_mean_shape_gap_dct01`

This feature intentionally removes most amplitude information:

1. For every matched segment, divide the Endocardial curve by its own maximum absolute
   amplitude.
2. Do the same for the Mid-wall curve. Curves with maximum magnitude below 3% are
   treated as invalid.
3. Calculate the normalized Endo–Mid shape gap at each time point:
   `normalized Endo_s(t) - normalized Mid_s(t)`.
4. Average this gap across segments to obtain one mean shape-gap curve.
5. Apply the DCT and retain `DCT01`. This first coefficient is proportional to the
   average level of the shape-gap curve across the cardiac cycle.

It represents broad, systematic separation between Endocardial and Mid-wall curve
shapes, largely independent of their absolute strain amplitudes. Extra Trees can use
different thresholds and interactions, so this feature has no single global
"higher means higher risk" direction.

**Importance:** {result_text(trees_row)}

## Features with stable positive held-out importance

Only features whose patient-bootstrap AUC interval was entirely above zero are shown.

{markdown_table(stable_features, ['Plain-language feature', 'Technical name', 'Model', 'Family', 'AUC decrease', 'AUC 95% CI', 'AP decrease'])}

## Top five features within each model

The technical name is retained so every row can be traced back to the source table.

{chr(10).join(ranking_sections)}

## Feature-family permutation

Here all features from one family are shuffled together. This measures how strongly a
model depends on the family as a whole, including correlated features that can replace
one another.

{markdown_table(group_table, ['model', 'family', 'AUC drop', 'AUC CI low', 'AUC CI high', 'AP drop'])}

The Endo–Mid family caused a larger AUC decrease than the variability family in the
combined Extra Trees model. This is the clearest family-level support for the Endo–Mid
hypothesis. However, the family-level confidence intervals still included zero.

## Main interpretation

1. **Endo–Mid curve relationships are the most promising engineered signal.** Both the
   sparse model and Extra Trees relied on time-dependent differences between the
   Endocardial and Mid-wall curves.
2. **Inter-segment variability is weaker.** A few amplitude and timing-dispersion
   features contributed modestly, but most were not stable and added little AP.
3. **Baseline GLS is important but partly label-coupled.** This is especially relevant
   for baseline Mid-GLS because the 15% outcome is calculated from it.
4. **DCT features are compact mathematical descriptors, not direct clinical
   measurements.** They should be validated using curve visualizations and an external
   cohort before receiving physiological labels.
5. **Permutation importance is model-specific, not causal.** Correlated features can
   substitute for one another, and Extra Trees can use nonlinear interactions.

## Reproducibility check

The reconstructed models exactly reproduced the saved out-of-fold results:

{markdown_table(base_metrics, ['model', 'n', 'events', 'roc_auc_reproduced', 'average_precision_reproduced', 'prediction_repeats_min', 'prediction_repeats_max'])}

## Output files

- `noncnn_feature_importance_top.csv`: detailed model-specific permutation results.
- `noncnn_feature_family_importance.csv`: family-level permutation results.
- `noncnn_feature_consensus.csv`: cross-model prioritization.
- `noncnn_feature_importance_all.csv`: model-native screening values.
- `top_feature_permutation_importance.png`: top-feature figure.
- `feature_family_permutation_importance.png`: feature-family figure.

The feature calculations are implemented in `cardiotoxicity_early_detection.py` and
`cardiotoxicity_next_visit_gpu.py`.
"""
    (output / "noncnn_feature_importance_report.md").write_text(
        report, encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    results = args.results.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    saved_metrics = pd.read_csv(results / "model_metrics.csv")
    if args.report_only:
        required = {
            "base_metrics": output / "noncnn_base_metric_reproduction.csv",
            "top_features": output / "noncnn_feature_importance_top.csv",
            "group_importance": output / "noncnn_feature_family_importance.csv",
            "consensus": output / "noncnn_feature_consensus.csv",
            "screen_summary": output / "noncnn_feature_importance_all.csv",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Report-only mode is missing required result files: "
                + ", ".join(missing)
            )
        write_report(
            output,
            pd.read_csv(required["base_metrics"]),
            pd.read_csv(required["top_features"]),
            pd.read_csv(required["group_importance"]),
            pd.read_csv(required["consensus"]),
            pd.read_csv(required["screen_summary"]),
            saved_metrics,
        )
        print(output / "noncnn_feature_importance_report.md")
        return 0

    if args.plots_only:
        top_path = output / "noncnn_feature_importance_top.csv"
        family_path = output / "noncnn_feature_family_importance.csv"
        missing = [str(path) for path in (top_path, family_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Plots-only mode is missing required result files: "
                + ", ".join(missing)
            )
        make_plots(output, pd.read_csv(top_path), pd.read_csv(family_path))
        print(output / "top_feature_permutation_importance.png")
        print(output / "feature_family_permutation_importance.png")
        return 0

    transitions = pd.read_parquet(results / "next_visit_transitions.parquet")
    transitions["patient_id"] = transitions["patient_id"].astype(str)
    manifest = pd.read_csv(results / "feature_manifest.csv")
    assignments = pd.read_csv(
        results / "patient_fold_assignments.csv",
        dtype={"patient_id": str},
    )
    saved_predictions = pd.read_parquet(results / "oof_predictions.parquet")
    metadata = json.loads((results / "run_metadata.json").read_text(encoding="utf-8"))
    feature_sets = load_feature_sets(manifest)
    splits = load_splits(assignments, set(transitions["patient_id"]))
    active_task_index = metadata["active_task_names"].index(TASK)

    screen, artifacts, base_metrics = screen_models(
        transitions,
        feature_sets,
        splits,
        active_task_index,
        args.seed,
        args.screen_repeats,
        args.n_jobs,
    )
    screen_summary = summarize_screen(screen)

    saved_primary = saved_predictions[
        saved_predictions["task"].eq(TASK)
        & saved_predictions["model"].isin(MODEL_LABELS)
    ][["model", "transition_id", "score"]]
    reproduction = []
    for model_name, _, _ in MODEL_SPECS:
        saved = saved_primary[saved_primary["model"].eq(model_name)]
        saved_auc = saved_metrics[
            saved_metrics["task"].eq(TASK)
            & saved_metrics["model"].eq(model_name)
        ]["roc_auc"].iloc[0]
        reproduced_auc = base_metrics[
            base_metrics["model"].eq(model_name)
        ]["roc_auc_reproduced"].iloc[0]
        reproduction.append(abs(saved_auc - reproduced_auc))
    if max(reproduction) > 1e-10:
        raise RuntimeError(
            f"Base-model reconstruction did not match saved AUCs: {reproduction}"
        )

    candidates = candidate_features(screen_summary, args.top_candidates)
    refined, group_importance = refined_permutation(
        transitions,
        artifacts,
        candidates,
        args.refined_repeats,
        args.bootstraps,
        args.seed,
    )
    refined = refined.merge(
        screen_summary,
        left_on=["model", "item"],
        right_on=["model", "feature"],
        how="left",
        suffixes=("", "_screen"),
    )
    refined["rank_within_model"] = refined.groupby("model")[
        "auc_drop_mean"
    ].rank(method="min", ascending=False)
    refined = refined.sort_values(["model", "rank_within_model"])
    consensus = add_consensus(refined)

    screen_summary.sort_values(
        ["model", "auc_rank_screen"]
    ).to_csv(output / "noncnn_feature_importance_all.csv", index=False)
    refined.to_csv(output / "noncnn_feature_importance_top.csv", index=False)
    group_importance.to_csv(
        output / "noncnn_feature_family_importance.csv", index=False
    )
    consensus.to_csv(output / "noncnn_feature_consensus.csv", index=False)
    base_metrics.to_csv(output / "noncnn_base_metric_reproduction.csv", index=False)
    pd.DataFrame(
        [
            {
                "task": TASK,
                "eligible_transitions": int(
                    transitions[f"mask__{TASK}"].astype(bool).sum()
                ),
                "events": int(
                    transitions.loc[
                        transitions[f"mask__{TASK}"].astype(bool),
                        f"label__{TASK}",
                    ].sum()
                ),
                "patients": int(transitions["patient_id"].nunique()),
                "screen_repeats": args.screen_repeats,
                "refined_repeats": args.refined_repeats,
                "patient_cluster_bootstraps": args.bootstraps,
                "seed": args.seed,
                "importance_definition": (
                    "decrease in patient-held-out metric after feature permutation"
                ),
            }
        ]
    ).to_json(
        output / "noncnn_feature_importance_metadata.json",
        orient="records",
        indent=2,
    )
    make_plots(output, refined, group_importance)
    write_report(
        output,
        base_metrics,
        refined,
        group_importance,
        consensus,
        screen_summary,
        saved_metrics,
    )
    print(base_metrics.to_string(index=False))
    print("\nTop features by model:")
    print(
        refined.groupby("model", sort=False)
        .head(10)[
            [
                "model",
                "item",
                "feature_family",
                "auc_drop_mean",
                "ap_drop_mean",
                "auc_drop_boot_low",
                "auc_drop_boot_high",
            ]
        ]
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
