from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import dct
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import cardiotoxicity_next_visit_gpu as core


DEFAULT_CURVES = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_dataset.parquet")
DEFAULT_VISITS = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet")
DEFAULT_BASE = Path(r"D:\us\cardiotoxicity_next_visit_gpu_results")
DEFAULT_OUTPUT = Path(r"D:\us\cardiotoxicity_nonapical_qc_results")
POLICIES = ("noapex", "noapex_fixed_qc", "noapex_shape_qc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Non-apical and curve-QC next-visit ablation study.")
    parser.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--visits", type=Path, default=DEFAULT_VISITS)
    parser.add_argument("--base-results", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return np.nan
    aa = a[valid] - np.mean(a[valid])
    bb = b[valid] - np.mean(b[valid])
    denominator = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / denominator) if denominator > 1e-12 else np.nan


def robust_sd(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def add_stats(row: dict[str, object], prefix: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        for suffix in ("mean", "median", "std", "robust_sd", "iqr"):
            row[f"{prefix}_{suffix}"] = np.nan
        return
    row[f"{prefix}_mean"] = float(np.mean(values))
    row[f"{prefix}_median"] = float(np.median(values))
    row[f"{prefix}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    row[f"{prefix}_robust_sd"] = robust_sd(values)
    row[f"{prefix}_iqr"] = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))


def mean_pairwise_rmse(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    if len(matrix) < 2:
        return np.nan
    values = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            values.append(float(np.sqrt(np.mean((matrix[i] - matrix[j]) ** 2))))
    return float(np.mean(values))


def mean_pairwise_correlation(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    values = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            value = finite_corr(matrix[i], matrix[j])
            if np.isfinite(value):
                values.append(value)
    return float(np.mean(values)) if values else np.nan


def circular_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    radians = 2.0 * np.pi * values
    resultant = np.sqrt(np.mean(np.sin(radians)) ** 2 + np.mean(np.cos(radians)) ** 2)
    return float(np.sqrt(max(0.0, -2.0 * np.log(max(resultant, 1e-12)))) / (2.0 * np.pi))


def prepare_curves(path: Path, max_segment: int = 12) -> pd.DataFrame:
    columns = [
        "analysis_id",
        "visit_id",
        "patient_id",
        "layer",
        "curve_family",
        "segment_number",
        "segment_ring",
        "segment_view",
        "peak_abs",
        "time_to_peak_norm",
        "curve_roughness",
        "resampled_values",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame[
        (frame["curve_family"] == "longitudinal_strain")
        & frame["layer"].isin(["endo", "mid"])
        & frame["segment_number"].between(1, max_segment)
    ].copy().reset_index(drop=True)
    frame["segment_number"] = frame["segment_number"].astype(int)
    arrays = np.stack(frame["resampled_values"].map(lambda value: np.asarray(value, dtype=np.float32)))
    scale = np.max(np.abs(arrays), axis=1)
    shapes = np.divide(
        arrays,
        scale[:, None],
        out=np.zeros_like(arrays),
        where=scale[:, None] >= 3.0,
    )
    d2_rms = np.sqrt(np.mean(np.diff(shapes, n=2, axis=1) ** 2, axis=1))
    positive_fraction = np.mean(arrays > 1.0, axis=1)
    negative_fraction = np.mean(arrays < -1.0, axis=1)
    fixed = (
        frame["peak_abs"].between(3.0, 45.0).to_numpy()
        & frame["time_to_peak_norm"].between(0.20, 0.90).to_numpy()
        & (d2_rms <= 0.04)
        & (positive_fraction <= 0.55)
        & (negative_fraction >= 0.25)
    )
    consensus_correlation = np.full(len(frame), np.nan, dtype=float)
    for _, indices in frame.groupby(["analysis_id", "layer"]).groups.items():
        local = np.asarray(list(indices), dtype=int)
        consensus = np.median(shapes[local], axis=0)
        for index in local:
            consensus_correlation[index] = finite_corr(shapes[index], consensus)
    frame["curve_values"] = list(arrays)
    frame["shape_values"] = list(shapes)
    frame["shape_second_difference_rms"] = d2_rms
    frame["positive_fraction"] = positive_fraction
    frame["negative_fraction"] = negative_fraction
    frame["consensus_correlation"] = consensus_correlation
    frame["accept__noapex"] = True
    frame["accept__noapex_fixed_qc"] = fixed
    frame["accept__noapex_shape_qc"] = fixed & (consensus_correlation >= 0.75)
    return frame


def analysis_features(group: pd.DataFrame, policy: str) -> dict[str, object]:
    accept = f"accept__{policy}"
    accepted = group[group[accept]].copy()
    row: dict[str, object] = {
        "analysis_id": str(group["analysis_id"].iloc[0]),
        "visit_id": str(group["visit_id"].iloc[0]),
        "patient_id": str(group["patient_id"].iloc[0]),
        "q_total_curves": int(len(group)),
        "q_accepted_curves": int(len(accepted)),
        "q_accepted_fraction": float(len(accepted) / max(len(group), 1)),
        "q_consensus_correlation_median": float(np.nanmedian(accepted["consensus_correlation"]))
        if len(accepted)
        else np.nan,
    }
    layer_tables: dict[str, pd.DataFrame] = {}
    for layer in ("endo", "mid"):
        table = accepted[accepted["layer"] == layer].drop_duplicates("segment_number").set_index(
            "segment_number"
        )
        layer_tables[layer] = table
        prefix = f"var_{layer}"
        peaks = table["peak_abs"].to_numpy(float)
        timing = table["time_to_peak_norm"].to_numpy(float)
        curves = (
            np.stack(table["curve_values"].to_numpy()) if len(table) else np.empty((0, 96))
        )
        shapes = (
            np.stack(table["shape_values"].to_numpy()) if len(table) else np.empty((0, 96))
        )
        row[f"{prefix}_segments"] = int(len(table))
        row[f"{prefix}_accepted_fraction"] = float(len(table) / 12.0)
        add_stats(row, f"{prefix}_peak", peaks)
        row[f"{prefix}_peak_cv"] = (
            float(np.std(peaks, ddof=1) / np.mean(peaks))
            if len(peaks) > 1 and np.mean(peaks) > 1e-9
            else np.nan
        )
        add_stats(row, f"{prefix}_ttp", timing)
        row[f"{prefix}_ttp_circular_std"] = circular_std(timing)
        row[f"{prefix}_curve_dispersion_rms"] = (
            float(np.sqrt(np.mean((curves - np.mean(curves, axis=0)) ** 2)))
            if len(curves) > 1
            else np.nan
        )
        row[f"{prefix}_curve_pairwise_rmse"] = mean_pairwise_rmse(curves)
        row[f"{prefix}_shape_dispersion_rms"] = (
            float(np.sqrt(np.mean((shapes - np.mean(shapes, axis=0)) ** 2)))
            if len(shapes) > 1
            else np.nan
        )
        row[f"{prefix}_shape_pairwise_rmse"] = mean_pairwise_rmse(shapes)
        mean_correlation = mean_pairwise_correlation(shapes)
        row[f"{prefix}_shape_incoherence"] = (
            float(1.0 - mean_correlation) if np.isfinite(mean_correlation) else np.nan
        )
        row[f"{prefix}_roughness_median"] = float(np.nanmedian(table["curve_roughness"])) if len(table) else np.nan
        row[f"{prefix}_roughness_p90"] = float(np.nanquantile(table["curve_roughness"], 0.90)) if len(table) else np.nan
        row[f"{prefix}_impaired_fraction_lt15"] = float(np.mean(peaks < 15.0)) if len(peaks) else np.nan
        basal = table.loc[table.index.intersection(range(1, 7)), "peak_abs"].to_numpy(float)
        middle = table.loc[table.index.intersection(range(7, 13)), "peak_abs"].to_numpy(float)
        row[f"{prefix}_mid_basal_peak_gradient"] = (
            float(np.mean(middle) - np.mean(basal)) if len(basal) and len(middle) else np.nan
        )
        view_dispersion = []
        for segments in ([1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]):
            values = table.loc[table.index.intersection(segments), "peak_abs"].to_numpy(float)
            if len(values) >= 2:
                view_dispersion.append(robust_sd(values))
        row[f"{prefix}_within_view_peak_robust_sd_mean"] = (
            float(np.mean(view_dispersion)) if view_dispersion else np.nan
        )
        peak_edges = []
        timing_edges = []
        for segment in range(1, 7):
            if segment in table.index and segment + 6 in table.index:
                peak_edges.append(abs(float(table.loc[segment, "peak_abs"]) - float(table.loc[segment + 6, "peak_abs"])))
                timing_edges.append(abs(float(table.loc[segment, "time_to_peak_norm"]) - float(table.loc[segment + 6, "time_to_peak_norm"])))
        row[f"{prefix}_spatial_peak_graph_roughness"] = float(np.mean(peak_edges)) if peak_edges else np.nan
        row[f"{prefix}_spatial_timing_graph_roughness"] = float(np.mean(timing_edges)) if timing_edges else np.nan

    endo = layer_tables["endo"]
    mid = layer_tables["mid"]
    segments = endo.index.intersection(mid.index)
    row["tm_paired_segments"] = int(len(segments))
    row["q_paired_fraction"] = float(len(segments) / 12.0)
    if len(segments):
        endo = endo.loc[segments]
        mid = mid.loc[segments]
        endo_curve = np.stack(endo["curve_values"].to_numpy())
        mid_curve = np.stack(mid["curve_values"].to_numpy())
        endo_shape = np.stack(endo["shape_values"].to_numpy())
        mid_shape = np.stack(mid["shape_values"].to_numpy())
        endo_peak = endo["peak_abs"].to_numpy(float)
        mid_peak = mid["peak_abs"].to_numpy(float)
        peak_gap = endo_peak - mid_peak
        peak_ratio = np.divide(endo_peak, mid_peak, out=np.full_like(endo_peak, np.nan), where=mid_peak >= 3.0)
        ttp_gap = endo["time_to_peak_norm"].to_numpy(float) - mid["time_to_peak_norm"].to_numpy(float)
        difference = endo_curve - mid_curve
        shape_difference = endo_shape - mid_shape
        curve_rms = np.sqrt(np.mean(difference**2, axis=1))
        shape_rms = np.sqrt(np.mean(shape_difference**2, axis=1))
        correlations = np.asarray([finite_corr(a, b) for a, b in zip(endo_curve, mid_curve)])
        row["tm_peak_layer_correlation"] = finite_corr(endo_peak, mid_peak)
        row["tm_ttp_layer_correlation"] = finite_corr(
            endo["time_to_peak_norm"].to_numpy(float), mid["time_to_peak_norm"].to_numpy(float)
        )
        row["tm_fraction_endo_peak_gt_mid"] = float(np.mean(peak_gap > 0))
        row["tm_fraction_ttp_discordant_gt_5pct_cycle"] = float(np.mean(np.abs(ttp_gap) > 0.05))
        for name, values in (
            ("tm_peak_gap", peak_gap),
            ("tm_peak_ratio", peak_ratio),
            ("tm_ttp_gap", ttp_gap),
            ("tm_curve_rms", curve_rms),
            ("tm_shape_rms", shape_rms),
            ("tm_segment_curve_correlation", correlations),
        ):
            add_stats(row, name, values)
        for prefix, waveform in (
            ("tm_mean_gap", np.mean(difference, axis=0)),
            ("tm_sd_gap", np.std(difference, axis=0)),
            ("tm_mean_shape_gap", np.mean(shape_difference, axis=0)),
        ):
            for index, value in enumerate(dct(waveform, norm="ortho")[:6], start=1):
                row[f"{prefix}_dct{index:02d}"] = float(value)
        endo_consensus = np.mean(endo_curve, axis=0)
        mid_consensus = np.mean(mid_curve, axis=0)
        row["tm_consensus_curve_correlation"] = finite_corr(endo_consensus, mid_consensus)
        cross = np.correlate(endo_consensus - np.mean(endo_consensus), mid_consensus - np.mean(mid_consensus), mode="full")
        row["tm_consensus_phase_lag_fraction"] = float((np.argmax(cross) - 95) / 95.0)
    return row


def build_policy_data(curves: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    rows = [analysis_features(group, policy) for _, group in curves.groupby("analysis_id", sort=False)]
    analysis = pd.DataFrame(rows)
    numeric = [column for column in analysis.columns if column not in {"analysis_id", "visit_id", "patient_id"}]
    visits = analysis.groupby(["visit_id", "patient_id"], as_index=False)[numeric].mean()

    accept = f"accept__{policy}"
    tensors: dict[str, np.ndarray] = {}
    tensor_audit = []
    for visit_id, group in curves.groupby("visit_id", sort=False):
        tensor = np.empty((12, 2, 96), dtype=np.float32)
        imputed = 0
        for layer_index, layer in enumerate(("endo", "mid")):
            layer_group = group[group["layer"] == layer]
            accepted_layer = layer_group[layer_group[accept]]
            fallback_source = accepted_layer if len(accepted_layer) else layer_group
            fallback = np.median(np.stack(fallback_source["curve_values"].to_numpy()), axis=0)
            for segment in range(1, 13):
                segment_group = accepted_layer[accepted_layer["segment_number"] == segment]
                if len(segment_group):
                    tensor[segment - 1, layer_index] = np.mean(
                        np.stack(segment_group["curve_values"].to_numpy()), axis=0
                    )
                else:
                    tensor[segment - 1, layer_index] = fallback
                    imputed += 1
        tensors[str(visit_id)] = tensor
        tensor_audit.append(
            {
                "policy": policy,
                "visit_id": str(visit_id),
                "tensor_curves": 24,
                "imputed_curves": imputed,
                "imputed_fraction": imputed / 24.0,
            }
        )
    return visits, tensors, pd.DataFrame(tensor_audit)


def build_variant_transitions(
    base: pd.DataFrame,
    visits: pd.DataFrame,
    variant_visits: pd.DataFrame,
    tensors: dict[str, np.ndarray],
    clinical: list[str],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, list[str]]]:
    old_engineered = [
        column
        for column in base.columns
        if column.startswith(("cur_tm_", "d_tm_", "cur_var__", "d_var__", "cur_q_", "d_q_"))
    ]
    frame = base.drop(columns=old_engineered).reset_index(drop=True).copy()
    visit_order = {
        (str(row.patient_id), int(row.visit_order)): str(row.visit_id)
        for row in visits[["patient_id", "visit_order", "visit_id"]].itertuples(index=False)
    }
    variant = variant_visits.set_index("visit_id")
    tm_columns = [column for column in variant.columns if column.startswith("tm_")]
    var_columns = [column for column in variant.columns if column.startswith(("var_", "q_"))]
    curve_inputs = []
    feature_rows = []
    for row in frame.itertuples(index=False):
        current_id = str(row.current_visit_id)
        current_order = int(row.current_visit_order)
        previous_id = current_id if current_order == 1 else visit_order[(str(row.patient_id), current_order - 1)]
        current_features = variant.loc[current_id]
        previous_features = variant.loc[previous_id]
        values: dict[str, float] = {}
        for column in tm_columns + var_columns:
            current_value = float(current_features[column])
            previous_value = float(previous_features[column])
            values[f"cur_{column}"] = current_value
            values[f"d_{column}"] = 0.0 if current_order == 1 else current_value - previous_value
        feature_rows.append(values)
        current_tensor = tensors[current_id]
        previous_tensor = tensors[previous_id]
        endo = current_tensor[:, 0]
        mid = current_tensor[:, 1]
        delta_endo = endo - previous_tensor[:, 0] if current_order > 1 else np.zeros_like(endo)
        delta_mid = mid - previous_tensor[:, 1] if current_order > 1 else np.zeros_like(mid)
        tensor = np.stack(
            [endo, mid, endo - mid, delta_endo, delta_mid, delta_endo - delta_mid], axis=1
        ).astype(np.float32)
        curve_inputs.append(np.clip(tensor / 30.0, -2.0, 2.0))
    feature_frame = pd.DataFrame(feature_rows).reset_index(drop=True)
    frame = pd.concat([frame, feature_frame], axis=1).copy()
    tm_features = [column for column in frame.columns if column.startswith(("cur_tm_", "d_tm_"))]
    var_features = [column for column in frame.columns if column.startswith(("cur_var_", "d_var_", "cur_q_", "d_q_"))]
    feature_sets = {
        "clinical": clinical,
        "clinical_plus_transmural": clinical + tm_features,
        "clinical_plus_variability": clinical + var_features,
        "combined": clinical + tm_features + var_features,
        "gpu_scalars": clinical + var_features,
    }
    return frame, np.stack(curve_inputs), feature_sets


def rename_predictions(predictions: pd.DataFrame, prefix: str) -> pd.DataFrame:
    mapping = {
        "clinical_plus_transmural_sparse": f"{prefix}__transmural_sparse",
        "clinical_plus_variability_sparse": f"{prefix}__variability_sparse",
        "combined_extra_trees": f"{prefix}__combined_trees",
        "gpu_segment_curve_net": f"{prefix}__gpu_curve_net",
    }
    result = predictions.copy()
    result["model"] = result["model"].replace(mapping)
    return result


def cpu_variant_oof_predictions(
    transitions: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    active_tasks: list[core.TaskSpec],
    splits: list[dict[str, object]],
    seed: int,
) -> pd.DataFrame:
    model_specs = [
        ("clinical_plus_transmural_sparse", "clinical_plus_transmural", "l1"),
        ("clinical_plus_variability_sparse", "clinical_plus_variability", "l1"),
        ("combined_extra_trees", "combined", "trees"),
    ]
    accumulators = {
        (task.name, model_name): (
            np.zeros(len(transitions), dtype=float),
            np.zeros(len(transitions), dtype=int),
        )
        for task in active_tasks
        for model_name, _, _ in model_specs
    }
    for split in splits:
        train_patient = transitions["patient_id"].isin(split["train_patients"]).to_numpy()
        test_patient = transitions["patient_id"].isin(split["test_patients"]).to_numpy()
        for task_index, task in enumerate(active_tasks):
            task_mask = transitions[f"mask__{task.name}"].astype(bool).to_numpy()
            train_index = np.flatnonzero(train_patient & task_mask)
            test_index = np.flatnonzero(test_patient & task_mask)
            y_train = transitions.iloc[train_index][f"label__{task.name}"].to_numpy(int)
            if not len(test_index) or np.unique(y_train).size < 2:
                continue
            for model_index, (model_name, feature_set, kind) in enumerate(model_specs):
                columns = core.usable_features(
                    transitions.iloc[train_index], feature_sets[feature_set]
                )
                if kind == "trees":
                    model: object = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "model",
                                ExtraTreesClassifier(
                                    n_estimators=80,
                                    min_samples_leaf=5,
                                    max_features=0.5,
                                    class_weight="balanced_subsample",
                                    n_jobs=-1,
                                    random_state=seed
                                    + int(split["split_index"]) * 100
                                    + task_index * 10
                                    + model_index,
                                ),
                            ),
                        ]
                    )
                else:
                    model = core.robust_linear_pipeline(
                        kind,
                        seed
                        + int(split["split_index"]) * 100
                        + task_index * 10
                        + model_index,
                    )
                model.fit(transitions.iloc[train_index][columns].astype(float), y_train)
                score = model.predict_proba(
                    transitions.iloc[test_index][columns].astype(float)
                )[:, 1]
                score_sum, score_count = accumulators[(task.name, model_name)]
                score_sum[test_index] += score
                score_count[test_index] += 1
    rows = []
    for task in active_tasks:
        mask = transitions[f"mask__{task.name}"].astype(bool).to_numpy()
        for model_name, _, _ in model_specs:
            score_sum, score_count = accumulators[(task.name, model_name)]
            valid = mask & (score_count > 0)
            for index in np.flatnonzero(valid):
                rows.append(
                    {
                        "task": task.name,
                        "model": model_name,
                        "transition_id": transitions.iloc[index]["transition_id"],
                        "patient_id": transitions.iloc[index]["patient_id"],
                        "label": int(transitions.iloc[index][f"label__{task.name}"]),
                        "score": float(score_sum[index] / score_count[index]),
                        "prediction_repeats": int(score_count[index]),
                    }
                )
    return pd.DataFrame(rows)


def paired_variant_deltas(
    predictions: pd.DataFrame,
    comparisons: list[tuple[str, str, str]],
    bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for task_index, task in enumerate(sorted(predictions["task"].unique())):
        task_frame = predictions[predictions["task"] == task]
        for comparison_index, (name, reference, candidate) in enumerate(comparisons):
            ref = task_frame[task_frame["model"] == reference][
                ["transition_id", "patient_id", "label", "score"]
            ].rename(columns={"score": "reference_score"})
            cand = task_frame[task_frame["model"] == candidate][
                ["transition_id", "patient_id", "label", "score"]
            ]
            merged = cand.merge(ref, on=["transition_id", "patient_id", "label"], how="inner")
            if merged.empty or merged["label"].nunique() < 2:
                continue
            y = merged["label"].to_numpy(int)
            score = merged["score"].to_numpy(float)
            reference_score = merged["reference_score"].to_numpy(float)
            observed = core.metric_values(y, score)
            observed_reference = core.metric_values(y, reference_score)
            rng = np.random.default_rng(seed + task_index * 1000 + comparison_index)
            samples = {"roc_auc": [], "average_precision": []}
            patient_ids = merged["patient_id"].to_numpy(str)
            for _ in range(bootstraps):
                index = core.cluster_sample_indices(patient_ids, rng)
                if np.unique(y[index]).size < 2:
                    continue
                current = core.metric_values(y[index], score[index])
                prior = core.metric_values(y[index], reference_score[index])
                for metric in samples:
                    samples[metric].append(current[metric] - prior[metric])
            row: dict[str, object] = {
                "task": task,
                "comparison": name,
                "reference_model": reference,
                "candidate_model": candidate,
                "n": len(merged),
            }
            for metric, values in samples.items():
                row[f"delta_{metric}"] = observed[metric] - observed_reference[metric]
                row[f"delta_{metric}_ci_low"] = float(np.quantile(values, 0.025))
                row[f"delta_{metric}_ci_high"] = float(np.quantile(values, 0.975))
            rows.append(row)
    return pd.DataFrame(rows)


def make_figure(output: Path, metrics: pd.DataFrame) -> None:
    primary = metrics[metrics["task"] == "mid_first_rel15"].copy()
    families = {
        "transmural": [model for model in primary["model"] if "transmural" in model],
        "variability": [model for model in primary["model"] if "variability" in model],
        "trees": [model for model in primary["model"] if "combined_trees" in model],
        "GPU curves": [model for model in primary["model"] if "gpu_curve_net" in model],
    }
    order = ["original18", *POLICIES]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    for color, (family, models) in zip(colors, families.items()):
        subset = primary[primary["model"].isin(models)].copy()
        subset["policy"] = subset["model"].str.split("__").str[0]
        subset = subset.set_index("policy").reindex(order)
        axes[0].plot(order, subset["roc_auc"], marker="o", label=family, color=color)
        axes[1].plot(order, subset["average_precision"], marker="o", label=family, color=color)
    clinical = primary[primary["model"] == "clinical_ridge"].iloc[0]
    axes[0].axhline(clinical["roc_auc"], color="black", linestyle="--", label="clinical ridge")
    axes[1].axhline(clinical["average_precision"], color="black", linestyle="--", label="clinical ridge")
    axes[0].axhline(0.5, color="gray", linestyle=":", label="random AUC")
    axes[1].axhline(clinical["prevalence"], color="gray", linestyle=":", label="random AP")
    axes[0].set_ylabel("ROC AUC")
    axes[1].set_ylabel("Average precision")
    for axis in axes:
        axis.tick_params(axis="x", rotation=22)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_title("Discrimination")
    axes[1].set_title("Event ranking")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Primary next-visit task: 15% relative Mid-GLS decline")
    fig.tight_layout()
    fig.savefig(output / "nonapical_qc_primary_comparison.png", dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[columns].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in display.to_numpy()],
        ]
    )


def write_report(output: Path, metrics: pd.DataFrame, deltas: pd.DataFrame, qc: pd.DataFrame) -> None:
    primary = metrics[metrics["task"] == "mid_first_rel15"].copy()
    wanted = [
        "clinical_ridge",
        "original18__transmural_sparse",
        "noapex__transmural_sparse",
        "noapex_fixed_qc__transmural_sparse",
        "noapex_shape_qc__transmural_sparse",
        "original18__variability_sparse",
        "noapex__variability_sparse",
        "noapex_fixed_qc__variability_sparse",
        "noapex_shape_qc__variability_sparse",
        "original18__combined_trees",
        "noapex__combined_trees",
        "noapex_fixed_qc__combined_trees",
        "noapex_shape_qc__combined_trees",
        "original18__gpu_curve_net",
        "noapex__gpu_curve_net",
        "noapex_fixed_qc__gpu_curve_net",
        "noapex_shape_qc__gpu_curve_net",
    ]
    primary["sort"] = primary["model"].map({model: index for index, model in enumerate(wanted)})
    primary = primary[primary["model"].isin(wanted)].sort_values("sort")
    primary_delta = deltas[deltas["task"] == "mid_first_rel15"].copy()
    retention = qc.groupby("policy", as_index=False).agg(
        raw_curves=("curves", "sum"), accepted_curves=("accepted_curves", "sum")
    )
    retention["retained_fraction"] = retention["accepted_curves"] / retention["raw_curves"]
    stable_auc = primary_delta[primary_delta["delta_roc_auc_ci_low"] > 0]
    stable_ap = primary_delta[primary_delta["delta_average_precision_ci_low"] > 0]
    conclusion = (
        "At least one removal/filtering comparison had a patient-bootstrap-stable gain."
        if len(stable_auc) or len(stable_ap)
        else "No apex-removal or filtering gain was patient-bootstrap stable on the primary task."
    )
    report = f"""# Non-apical and curve-quality ablation

## Bottom line

{conclusion}

Segments 13–18 (the apical ring) were removed from every variant. Labels, clinical trajectory features, patient splits, and relative thresholds were unchanged.

## Filters

- `noapex`: retain all basal/mid curves (segments 1–12).
- `noapex_fixed_qc`: reject near-flat/extreme curves, peak outside 3–45%, time-to-peak outside 0.20–0.90 cycle, excessive normalized second-difference roughness, or positive-dominant morphology.
- `noapex_shape_qc`: fixed QC plus correlation below 0.75 with the within-analysis/layer median shape.
- Rejected tensor curves were replaced by the same-visit/layer median. Engineered summaries used accepted curves only and included retained-fraction features.

## Curve retention

{markdown_table(retention, ['policy', 'raw_curves', 'accepted_curves', 'retained_fraction'])}

## Primary task: first baseline, 15% relative Mid-GLS decline

{markdown_table(primary, ['model', 'n', 'events', 'roc_auc', 'roc_auc_ci_low', 'roc_auc_ci_high', 'average_precision', 'average_precision_ci_low', 'average_precision_ci_high'])}

## Direct paired changes

Positive values favor the candidate. These are paired on identical patient-held-out predictions.

{markdown_table(primary_delta, ['comparison', 'delta_roc_auc', 'delta_roc_auc_ci_low', 'delta_roc_auc_ci_high', 'delta_average_precision', 'delta_average_precision_ci_low', 'delta_average_precision_ci_high'])}

## Notes

- The CNN used the NVIDIA GPU for all three variants.
- Shape-consensus filtering is deliberately a sensitivity analysis: a true regional abnormality can also look like a shape outlier.
- This is exploratory validation, not a clinical alert system.
"""
    (output / "nonapical_qc_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    core.set_seed(args.seed)

    base = pd.read_parquet(args.base_results / "next_visit_transitions.parquet")
    original_predictions = pd.read_parquet(args.base_results / "oof_predictions.parquet")
    visits = pd.read_parquet(args.visits)
    manifest = pd.read_csv(args.base_results / "feature_manifest.csv")
    clinical = manifest[manifest["feature_set"] == "clinical"]["feature"].tolist()
    tasks = core.task_specs()
    audit = core.label_audit(base, tasks)
    active_names = set(
        audit[
            (audit["eligible_transitions"] >= 30)
            & (audit["events"] >= 8)
            & ((audit["eligible_transitions"] - audit["events"]) >= 8)
        ]["task"]
    )
    active_all = [task for task in tasks if task.name in active_names]
    active_mid = [
        task
        for task in active_all
        if task.name.startswith("mid_") and "_roll3_" not in task.name
    ]
    mid_names = {task.name for task in active_mid}
    splits, assignments = core.patient_splits(base, args.cv_repeats, args.seed)

    curves = prepare_curves(args.curves.resolve())
    qc_rows = []
    prediction_parts = []
    gpu_logs = []
    feature_manifest = []

    original_mid = original_predictions[original_predictions["task"].isin(mid_names)].copy()
    original_mid = rename_predictions(original_mid, "original18")
    clinical_predictions = original_mid[original_mid["model"] == "clinical_ridge"].copy()
    original_nonclinical = original_mid[original_mid["model"] != "clinical_ridge"].copy()
    prediction_parts.extend([clinical_predictions, original_nonclinical])

    for policy in POLICIES:
        accepted = f"accept__{policy}"
        for (analysis_id, layer), group in curves.groupby(["analysis_id", "layer"]):
            qc_rows.append(
                {
                    "policy": policy,
                    "analysis_id": str(analysis_id),
                    "layer": layer,
                    "curves": int(len(group)),
                    "accepted_curves": int(group[accepted].sum()),
                    "accepted_fraction": float(group[accepted].mean()),
                }
            )
        variant_visits, tensors, tensor_audit = build_policy_data(curves, policy)
        transitions, curve_inputs, feature_sets = build_variant_transitions(
            base, visits, variant_visits, tensors, clinical
        )
        for family, columns in feature_sets.items():
            feature_manifest.extend(
                {"policy": policy, "feature_set": family, "feature": column} for column in columns
            )
        cpu = cpu_variant_oof_predictions(
            transitions, feature_sets, active_mid, splits, args.seed
        )
        prediction_parts.append(rename_predictions(cpu, policy))
        gpu, gpu_log, gpu_metadata = core.gpu_oof_predictions(
            transitions,
            curve_inputs,
            feature_sets,
            active_all,
            splits,
            args.epochs,
            args.patience,
            args.seed,
        )
        gpu = gpu[gpu["task"].isin(mid_names)]
        prediction_parts.append(rename_predictions(gpu, policy))
        gpu_log.insert(0, "policy", policy)
        gpu_logs.append(gpu_log)
        tensor_audit.to_csv(output / f"tensor_imputation_{policy}.csv", index=False)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics, clinical_deltas = core.evaluate_predictions(predictions, args.bootstraps, args.seed)
    families = ("transmural_sparse", "variability_sparse", "combined_trees", "gpu_curve_net")
    comparisons = []
    for family in families:
        comparisons.extend(
            [
                (f"remove_apex__{family}", f"original18__{family}", f"noapex__{family}"),
                (f"fixed_qc_vs_noapex__{family}", f"noapex__{family}", f"noapex_fixed_qc__{family}"),
                (f"shape_qc_vs_noapex__{family}", f"noapex__{family}", f"noapex_shape_qc__{family}"),
            ]
        )
    variant_deltas = paired_variant_deltas(
        predictions, comparisons, args.bootstraps, args.seed + 90000
    )
    qc = pd.DataFrame(qc_rows)

    predictions.to_parquet(output / "oof_predictions.parquet", index=False)
    metrics.to_csv(output / "model_metrics.csv", index=False)
    clinical_deltas.to_csv(output / "model_deltas_vs_clinical.csv", index=False)
    variant_deltas.to_csv(output / "variant_deltas.csv", index=False)
    qc.to_csv(output / "curve_qc_audit.csv", index=False)
    pd.concat(gpu_logs, ignore_index=True).to_csv(output / "gpu_training_log.csv", index=False)
    pd.DataFrame(feature_manifest).to_csv(output / "feature_manifest.csv", index=False)
    assignments.to_csv(output / "patient_fold_assignments.csv", index=False)
    make_figure(output, metrics)
    write_report(output, metrics, variant_deltas, qc)

    metadata = {
        "excluded_segments": [13, 14, 15, 16, 17, 18],
        "retained_segments": list(range(1, 13)),
        "policies": list(POLICIES),
        "patients": int(base["patient_id"].nunique()),
        "transitions": int(len(base)),
        "active_mid_tasks": [task.name for task in active_mid],
        "gpu_auxiliary_tasks": [task.name for task in active_all],
        "cv_repeats": args.cv_repeats,
        "folds_per_repeat": 5,
        "bootstraps": args.bootstraps,
        "seed": args.seed,
        "gpu": {
            "device": gpu_metadata["device"],
            "torch_version": gpu_metadata["torch_version"],
            "torch_cuda_version": gpu_metadata["torch_cuda_version"],
            "curve_input": "12 non-apical segments x 6 channels x 96 phase points",
        },
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(output / "nonapical_qc_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
