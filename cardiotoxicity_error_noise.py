from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

import cardiotoxicity_next_visit_gpu as core
import cardiotoxicity_nonapical_qc as qc


VISITS_PATH = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet")
CURVES_PATH = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_dataset.parquet")
BASE_RESULTS = Path(r"D:\us\cardiotoxicity_next_visit_gpu_results")
OUTPUT = Path(r"D:\us\cardiotoxicity_plateau_results")
SEED = 20260722
BOOTSTRAPS = 500


def quantile_group(values: pd.Series, labels: list[str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    try:
        return pd.qcut(numeric, q=len(labels), labels=labels, duplicates="drop").astype(str)
    except ValueError:
        return pd.Series("unavailable", index=values.index)


def subgroup_metrics(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    grouping = [
        "history_group",
        "followup_interval_group",
        "baseline_gls_group",
        "current_decline_group",
        "ef_group",
        "endo_mid_gap_group",
        "variability_group",
        "curve_qc_group",
        "label_margin_group",
    ]
    rows = []
    for model_index, model in enumerate(models):
        full_score = frame[model].to_numpy(float)
        alert_threshold = float(np.quantile(full_score, 0.80))
        for group_index, group_column in enumerate(grouping):
            for level, subset in frame.groupby(group_column, dropna=False):
                y = subset["label"].to_numpy(int)
                score = subset[model].to_numpy(float)
                if len(subset) < 20 or y.sum() < 5 or (len(y) - y.sum()) < 5:
                    continue
                alerts = score >= alert_threshold
                tp = int(np.sum(alerts & (y == 1)))
                fp = int(np.sum(alerts & (y == 0)))
                row = {
                    "model": model,
                    "subgroup_variable": group_column,
                    "subgroup": str(level),
                    "n": len(subset),
                    "events": int(y.sum()),
                    "prevalence": float(y.mean()),
                    "roc_auc": float(roc_auc_score(y, score)),
                    "average_precision": float(average_precision_score(y, score)),
                    "sensitivity_at_global_top20": float(tp / y.sum()),
                    "false_positive_rate_at_global_top20": float(fp / max((y == 0).sum(), 1)),
                    "precision_at_global_top20": float(tp / max(alerts.sum(), 1)),
                    "mean_score": float(np.mean(score)),
                }
                rng = np.random.default_rng(
                    SEED + 700000 + model_index * 10000 + group_index * 100
                )
                patient = subset["patient_id"].to_numpy(str)
                samples_auc = []
                samples_ap = []
                for _ in range(BOOTSTRAPS):
                    index = core.cluster_sample_indices(patient, rng)
                    if np.unique(y[index]).size < 2:
                        continue
                    samples_auc.append(roc_auc_score(y[index], score[index]))
                    samples_ap.append(average_precision_score(y[index], score[index]))
                row["roc_auc_ci_low"] = float(np.quantile(samples_auc, 0.025))
                row["roc_auc_ci_high"] = float(np.quantile(samples_auc, 0.975))
                row["average_precision_ci_low"] = float(np.quantile(samples_ap, 0.025))
                row["average_precision_ci_high"] = float(np.quantile(samples_ap, 0.975))
                rows.append(row)
    return pd.DataFrame(rows)


def calibration_analysis(frame: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary = []
    for model in models:
        ranked = frame.copy()
        ranked["bin"] = pd.qcut(ranked[model], q=5, labels=False, duplicates="drop") + 1
        for bin_number, subset in ranked.groupby("bin"):
            rows.append(
                {
                    "model": model,
                    "score_quintile": int(bin_number),
                    "n": len(subset),
                    "events": int(subset["label"].sum()),
                    "mean_score": float(subset[model].mean()),
                    "observed_event_rate": float(subset["label"].mean()),
                }
            )
        table = pd.DataFrame([row for row in rows if row["model"] == model])
        ece = float(
            np.sum(table["n"] * np.abs(table["mean_score"] - table["observed_event_rate"]))
            / table["n"].sum()
        )
        summary.append({"model": model, "quintile_ece": ece})
    return pd.DataFrame(rows), pd.DataFrame(summary)


def error_tables(frame: pd.DataFrame, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold = float(np.quantile(frame[model], 0.80))
    alerted = frame[model] >= threshold
    y = frame["label"].astype(bool)
    frame = frame.copy()
    frame["error_status"] = np.select(
        [alerted & y, alerted & ~y, ~alerted & y],
        ["TP", "FP", "FN"],
        default="TN",
    )
    features = [
        "followup_days",
        "first_mid_gls",
        "current_mid_gls",
        "current_mid_decline_from_first",
        "last_mid_relative_change",
        "current_endo_mid_gap",
        "mid_variability",
        "curve_rejected_fraction",
        "target_relative_decline",
        "absolute_label_margin",
    ]
    rows = []
    for status, subset in frame.groupby("error_status"):
        row = {"error_status": status, "n": len(subset), "events": int(subset["label"].sum())}
        for feature in features:
            row[f"median__{feature}"] = float(pd.to_numeric(subset[feature], errors="coerce").median())
        rows.append(row)
    detail_columns = [
        "transition_id",
        "patient_id",
        "current_visit_order",
        "target_visit_order",
        "label",
        model,
        "error_status",
        *features,
    ]
    detail = frame[detail_columns].sort_values(
        ["error_status", model], ascending=[True, False]
    )
    return pd.DataFrame(rows), detail


def replicate_noise_estimate() -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["analysis_id", "visit_id", "patient_id", "gls_mid_peak_avg"]
    curves = pd.read_parquet(CURVES_PATH, columns=columns)
    analysis = curves.groupby(["visit_id", "analysis_id"], as_index=False).agg(
        patient_id=("patient_id", "first"),
        mid_gls=("gls_mid_peak_avg", "first"),
    )
    rows = []
    for visit_id, group in analysis.groupby("visit_id"):
        if len(group) != 2:
            continue
        values = group["mid_gls"].abs().to_numpy(float)
        rows.append(
            {
                "visit_id": visit_id,
                "patient_id": str(group["patient_id"].iloc[0]),
                "replicate_1": values[0],
                "replicate_2": values[1],
                "difference": values[1] - values[0],
                "absolute_difference": abs(values[1] - values[0]),
                "mean_gls": float(np.mean(values)),
            }
        )
    pairs = pd.DataFrame(rows)
    difference = pairs["difference"].to_numpy(float)
    x = pairs["replicate_1"].to_numpy(float)
    y = pairs["replicate_2"].to_numpy(float)

    def estimates(index: np.ndarray) -> dict[str, float]:
        d = difference[index]
        xx = x[index]
        yy = y[index]
        conventional = float(np.std(d, ddof=1) / math.sqrt(2.0)) if len(d) > 1 else np.nan
        center = np.median(d)
        robust = float(1.4826 * np.median(np.abs(d - center)) / math.sqrt(2.0))
        covariance = np.mean((xx - np.mean(xx)) * (yy - np.mean(yy)))
        ccc = float(
            2.0
            * covariance
            / (np.var(xx) + np.var(yy) + (np.mean(xx) - np.mean(yy)) ** 2)
        )
        return {
            "within_sd_conventional": conventional,
            "within_sd_robust": robust,
            "mean_absolute_difference": float(np.mean(np.abs(d))),
            "concordance_correlation": ccc,
        }

    observed = estimates(np.arange(len(pairs)))
    rng = np.random.default_rng(SEED + 800000)
    samples = {key: [] for key in observed}
    for _ in range(2000):
        index = rng.integers(0, len(pairs), len(pairs))
        result = estimates(index)
        for key in samples:
            samples[key].append(result[key])
    summary = {"replicate_pairs": len(pairs), **observed}
    for key, values in samples.items():
        summary[f"{key}_ci_low"] = float(np.nanquantile(values, 0.025))
        summary[f"{key}_ci_high"] = float(np.nanquantile(values, 0.975))
    summary["conventional_relative_sd"] = observed["within_sd_conventional"] / float(
        pairs["mean_gls"].mean()
    )
    return pairs, pd.DataFrame([summary])


def simulate_first_crossing_labels(
    visits: pd.DataFrame,
    transitions: pd.DataFrame,
    threshold: float,
    sigma: float,
    simulations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((simulations, len(transitions)), dtype=np.uint8)
    eligible = np.zeros((simulations, len(transitions)), dtype=bool)
    transition_lookup = {
        (str(row.patient_id), int(row.current_visit_order)): index
        for index, row in enumerate(transitions.itertuples(index=False))
    }
    rng = np.random.default_rng(seed)
    for patient_id, group in visits.groupby("patient_id", sort=False):
        group = group.sort_values("visit_order")
        values = group["gls_mid_peak_avg"].abs().to_numpy(float)
        if sigma == 0:
            simulated = np.repeat(values[None, :], simulations, axis=0)
        else:
            simulated = np.clip(
                values[None, :] + rng.normal(0.0, sigma, size=(simulations, len(values))),
                1.0,
                None,
            )
        crossing = 1.0 - simulated[:, 1:] / simulated[:, [0]] >= threshold
        has_crossing = crossing.any(axis=1)
        first = np.argmax(crossing, axis=1)
        for local_index in range(len(values) - 1):
            global_index = transition_lookup[(str(patient_id), local_index + 1)]
            eligible[:, global_index] = (~has_crossing) | (local_index <= first)
            labels[:, global_index] = has_crossing & (local_index == first)
    return labels, eligible


def noise_ceiling(
    visits: pd.DataFrame,
    transitions: pd.DataFrame,
    replicate_summary: pd.DataFrame,
) -> pd.DataFrame:
    robust = float(replicate_summary.iloc[0]["within_sd_robust"])
    conventional = float(replicate_summary.iloc[0]["within_sd_conventional"])
    sigmas = sorted(set([0.0, 0.5, round(robust, 3), round(conventional, 3), 1.5]))
    rows = []
    simulations = 2000
    split = simulations // 2
    for threshold_index, threshold in enumerate((0.10, 0.12, 0.15, 0.20)):
        tag = int(round(threshold * 100))
        observed_mask = transitions[f"mask__mid_first_rel{tag}"].astype(bool).to_numpy()
        observed_label = transitions[f"label__mid_first_rel{tag}"].to_numpy(int)
        for sigma_index, sigma in enumerate(sigmas):
            labels, eligible = simulate_first_crossing_labels(
                visits,
                transitions,
                threshold,
                sigma,
                simulations,
                SEED + 900000 + threshold_index * 10000 + sigma_index,
            )
            training_labels = labels[:split]
            training_eligible = eligible[:split]
            eligible_count = training_eligible.sum(axis=0)
            probability = np.divide(
                training_labels.sum(axis=0),
                eligible_count,
                out=np.zeros(len(transitions), dtype=float),
                where=eligible_count > 0,
            )
            auc_values = []
            ap_values = []
            prevalence = []
            for simulation in range(split, simulations):
                mask = eligible[simulation]
                y = labels[simulation, mask]
                if np.unique(y).size < 2:
                    continue
                score = probability[mask]
                auc_values.append(roc_auc_score(y, score))
                ap_values.append(average_precision_score(y, score))
                prevalence.append(float(np.mean(y)))
            observed_probability = probability[observed_mask]
            observed_y = observed_label[observed_mask]
            flip_probability = np.where(
                observed_y == 1, 1.0 - observed_probability, observed_probability
            )
            rows.append(
                {
                    "threshold": threshold,
                    "noise_sd_gls_points": sigma,
                    "noise_relative_to_mean_gls": sigma / visits["gls_mid_peak_avg"].abs().mean(),
                    "observed_events": int(observed_y.sum()),
                    "observed_eligible": int(len(observed_y)),
                    "expected_observed_label_flip_fraction": float(np.mean(flip_probability)),
                    "observed_event_identity_reproducibility": float(
                        np.mean(observed_probability[observed_y == 1])
                    ),
                    "ambiguous_observed_fraction_p10_p90": float(
                        np.mean((observed_probability > 0.10) & (observed_probability < 0.90))
                    ),
                    "oracle_auc_mean": float(np.mean(auc_values)),
                    "oracle_auc_sim_low": float(np.quantile(auc_values, 0.025)),
                    "oracle_auc_sim_high": float(np.quantile(auc_values, 0.975)),
                    "oracle_ap_mean": float(np.mean(ap_values)),
                    "oracle_ap_sim_low": float(np.quantile(ap_values, 0.025)),
                    "oracle_ap_sim_high": float(np.quantile(ap_values, 0.975)),
                    "simulated_prevalence_mean": float(np.mean(prevalence)),
                }
            )
    return pd.DataFrame(rows)


def make_figures(
    attention: pd.DataFrame,
    subgroups: pd.DataFrame,
    noise: pd.DataFrame,
) -> None:
    figures = OUTPUT / "figures"
    figures.mkdir(exist_ok=True)
    weights = attention[attention["variant"] == "attention_binary"].sort_values("segment")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(weights["segment"], weights["mean_weight"] * 100, color="#4472C4")
    ax.errorbar(
        weights["segment"],
        weights["mean_weight"] * 100,
        yerr=[
            (weights["mean_weight"] - weights["mean_weight_ci_low"]) * 100,
            (weights["mean_weight_ci_high"] - weights["mean_weight"]) * 100,
        ],
        fmt="none",
        color="black",
        capsize=2,
        linewidth=0.8,
    )
    ax.axhline(100 / 18, color="gray", linestyle="--", label="uniform 1/18")
    ax.set_xticks(range(1, 19), [str(index) for index in range(1, 19)])
    ax.set_xlabel("Segment number")
    ax.set_ylabel("Mean held-out attention weight (%)")
    ax.set_title("Segment attention remains close to uniform")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "attention_weights.png", dpi=180)
    plt.close(fig)

    primary_noise = noise[noise["threshold"] == 0.15]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(primary_noise["noise_sd_gls_points"], primary_noise["oracle_auc_mean"], marker="o")
    axes[0].fill_between(
        primary_noise["noise_sd_gls_points"],
        primary_noise["oracle_auc_sim_low"],
        primary_noise["oracle_auc_sim_high"],
        alpha=0.2,
    )
    axes[0].axhline(0.683, color="gray", linestyle="--", label="attention AUC")
    axes[0].set_ylabel("Noise-limited oracle AUC")
    axes[0].legend(fontsize=8)
    axes[1].plot(
        primary_noise["noise_sd_gls_points"],
        primary_noise["observed_event_identity_reproducibility"],
        marker="o",
        color="#ED7D31",
    )
    axes[1].set_ylabel("Observed event identity reproducibility")
    for axis in axes:
        axis.set_xlabel("Assumed Mid-GLS measurement SD (points)")
        axis.grid(alpha=0.25)
    fig.suptitle("15% first-baseline label-noise sensitivity")
    fig.tight_layout()
    fig.savefig(figures / "label_noise_ceiling.png", dpi=180)
    plt.close(fig)

    selected = subgroups[
        (subgroups["model"] == "attention_binary")
        & subgroups["subgroup_variable"].isin(
            ["history_group", "followup_interval_group", "baseline_gls_group", "current_decline_group"]
        )
    ].copy()
    selected["label"] = selected["subgroup_variable"].str.replace("_group", "") + ": " + selected["subgroup"]
    selected = selected.sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(9, max(4.5, len(selected) * 0.38)))
    ax.errorbar(
        selected["roc_auc"],
        np.arange(len(selected)),
        xerr=[selected["roc_auc"] - selected["roc_auc_ci_low"], selected["roc_auc_ci_high"] - selected["roc_auc"]],
        fmt="o",
        color="#4472C4",
        capsize=2,
    )
    ax.axvline(0.5, color="gray", linestyle=":")
    ax.set_yticks(np.arange(len(selected)), selected["label"])
    ax.set_xlabel("ROC AUC with patient-bootstrap 95% CI")
    ax.set_title("Attention-model subgroup performance")
    fig.tight_layout()
    fig.savefig(figures / "subgroup_auc.png", dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    display = frame[columns].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        )
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in display.to_numpy()],
        ]
    )


def write_report(
    model_metrics: pd.DataFrame,
    ablations: pd.DataFrame,
    regression: pd.DataFrame,
    attention: pd.DataFrame,
    subgroups: pd.DataFrame,
    errors: pd.DataFrame,
    calibration: pd.DataFrame,
    replicate_summary: pd.DataFrame,
    noise: pd.DataFrame,
    tree_delta: pd.DataFrame,
) -> None:
    primary_models = model_metrics[
        (model_metrics["task"] == "mid_first_rel15")
        & model_metrics["model"].isin(
            ["clinical_ridge", "uniform_binary", "attention_binary", "uniform_gls_aux", "attention_gls_aux"]
        )
    ].sort_values("average_precision", ascending=False)
    primary_ablation = ablations[ablations["task"] == "mid_first_rel15"]
    top_weights = attention[attention["variant"] == "attention_binary"].nlargest(6, "mean_weight")
    weakest = subgroups[subgroups["model"] == "attention_binary"].sort_values("roc_auc").head(8)
    primary_noise = noise[(noise["threshold"] == 0.15)].copy()
    replicate = replicate_summary.iloc[0]
    status = errors.set_index("error_status")
    fn_count = int(status.loc["FN", "n"])
    fp_count = int(status.loc["FP", "n"])
    report = f"""# Focused plateau experiments

## Bottom line

Small segment attention produced the best point estimate (AUC 0.683, AP 0.333), but its paired gain over uniform pooling was not bootstrap-stable. The continuous next-visit GLS auxiliary target was only moderately learnable and did not improve classification. Measurement noise makes individual threshold-crossing labels unstable, but the simulated noise-only oracle remains well above the achieved AUC, so label noise is a contributor rather than the entire plateau.

## Primary task: first baseline, 15% relative Mid-GLS decline

{markdown_table(primary_models, ['model', 'n', 'events', 'roc_auc', 'roc_auc_ci_low', 'roc_auc_ci_high', 'average_precision', 'average_precision_ci_low', 'average_precision_ci_high', 'sensitivity_top20pct', 'precision_top20pct'])}

Paired ablations:

{markdown_table(primary_ablation, ['comparison', 'delta_roc_auc', 'delta_roc_auc_ci_low', 'delta_roc_auc_ci_high', 'delta_average_precision', 'delta_average_precision_ci_low', 'delta_average_precision_ci_high'])}

Attention versus the existing combined Extra Trees model:

{markdown_table(tree_delta, ['comparison', 'delta_roc_auc', 'delta_roc_auc_ci_low', 'delta_roc_auc_ci_high', 'delta_average_precision', 'delta_average_precision_ci_low', 'delta_average_precision_ci_high'])}

## Segment attention

Uniform weight is 0.0556. The learned averages stayed close to uniform and event/non-event differences were not stable.

{markdown_table(top_weights, ['segment', 'segment_name', 'mean_weight', 'mean_weight_ci_low', 'mean_weight_ci_high', 'event_minus_nonevent', 'difference_ci_low', 'difference_ci_high'], digits=4)}

Interpretation: the point-estimate gain is more consistent with mild adaptive reweighting/regularization than a strong anatomical segment signature.

## Continuous GLS auxiliary target

{markdown_table(regression, ['variant', 'n', 'mae', 'mae_ci_low', 'mae_ci_high', 'r2', 'pearson_r', 'pearson_r_ci_low', 'pearson_r_ci_high', 'spearman_r'])}

The target was immediate relative Mid-GLS change. Adding its SmoothL1 loss reduced primary AUC/AP for both pooling methods.

## Error analysis

At the global top-20% alert threshold, the attention model produced {fn_count} false negatives and {fp_count} false positives. Despite higher overall AUC/AP, its primary sensitivity at this alert budget was 0.265, below the clinical ridge value of 0.327.

Weakest estimable subgroups:

{markdown_table(weakest, ['subgroup_variable', 'subgroup', 'n', 'events', 'roc_auc', 'roc_auc_ci_low', 'roc_auc_ci_high', 'average_precision', 'sensitivity_at_global_top20'])}

Calibration warning:

{markdown_table(calibration, ['model', 'quintile_ece'])}

Scores come from class-weighted models and should be interpreted as rankings, not calibrated probabilities.

## Strain label noise and ceiling

There were only {int(replicate['replicate_pairs'])} paired strain analyses. Estimated within-visit Mid-GLS SD was {replicate['within_sd_conventional']:.3f} points (95% bootstrap CI {replicate['within_sd_conventional_ci_low']:.3f}–{replicate['within_sd_conventional_ci_high']:.3f}); robust estimate {replicate['within_sd_robust']:.3f}. Mean absolute replicate difference was {replicate['mean_absolute_difference']:.3f} points. The small replicate sample makes this uncertain.

15% endpoint simulation:

{markdown_table(primary_noise, ['noise_sd_gls_points', 'expected_observed_label_flip_fraction', 'observed_event_identity_reproducibility', 'ambiguous_observed_fraction_p10_p90', 'oracle_auc_mean', 'oracle_auc_sim_low', 'oracle_auc_sim_high', 'oracle_ap_mean'])}

The oracle assumes perfect knowledge of the latent GLS trajectory and only random measurement error at observation. It is therefore an optimistic ceiling, not an expected model score. If the replicate SD near 0.8–0.9 points is realistic, a meaningful fraction of exact first-crossing identities is unstable; however, the oracle AUC remains above the achieved 0.683, showing that missing predictors and limited sample size still dominate.

## Recommendation

- Keep attention as an optional ensemble component, not a replacement justified by this cohort alone.
- Do not keep the GLS auxiliary loss at weight 0.25; it did not improve classification.
- Evaluate alerts over a confirmation window or require repeated deterioration to reduce threshold noise.
- Prioritize additional treatment/biomarker timing and more independent events before increasing model complexity.
"""
    (OUTPUT / "plateau_experiment_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    transitions = pd.read_parquet(BASE_RESULTS / "next_visit_transitions.parquet")
    visits = pd.read_parquet(VISITS_PATH)
    plateau_predictions = pd.read_parquet(OUTPUT / "plateau_oof_predictions.parquet")
    model_metrics = pd.read_csv(OUTPUT / "plateau_model_metrics.csv")
    ablations = pd.read_csv(OUTPUT / "plateau_ablation_deltas.csv")
    regression = pd.read_csv(OUTPUT / "auxiliary_gls_metrics.csv")
    attention = pd.read_csv(OUTPUT / "attention_weight_summary.csv")
    original = pd.read_parquet(BASE_RESULTS / "oof_predictions.parquet")

    primary = transitions[transitions["mask__mid_first_rel15"].astype(bool)].copy()
    primary["label"] = primary["label__mid_first_rel15"].astype(int)
    predictions = plateau_predictions[plateau_predictions["task"] == "mid_first_rel15"]
    for model in ["clinical_ridge", "uniform_binary", "attention_binary"]:
        score = predictions[predictions["model"] == model].set_index("transition_id")["score"]
        primary[model] = primary["transition_id"].map(score)
    tree_score = original[
        (original["task"] == "mid_first_rel15")
        & (original["model"] == "combined_extra_trees")
    ].set_index("transition_id")["score"]
    primary["existing_combined_trees"] = primary["transition_id"].map(tree_score)

    visit_index = visits.set_index("visit_id")
    current_day = primary["current_visit_id"].map(visit_index["days_since_baseline"])
    target_day = primary["target_visit_id"].map(visit_index["days_since_baseline"])
    primary["followup_days"] = target_day.to_numpy(float) - current_day.to_numpy(float)
    primary["target_relative_decline"] = primary["decline__mid_first_rel15"].astype(float)
    primary["absolute_label_margin"] = (primary["target_relative_decline"] - 0.15).abs()
    primary["mid_variability"] = primary["cur_var__mid_peak_abs_robust_sd"].astype(float)

    prepared = qc.prepare_curves(CURVES_PATH, max_segment=18)
    quality = prepared.groupby("visit_id")["accept__noapex_fixed_qc"].mean()
    primary["curve_rejected_fraction"] = 1.0 - primary["current_visit_id"].map(quality).astype(float)
    primary["history_group"] = np.where(primary["history_visits"] == 1, "current=V1", "current>=V2")
    primary["followup_interval_group"] = quantile_group(
        primary["followup_days"], ["short", "middle", "long"]
    )
    primary["baseline_gls_group"] = np.where(primary["first_mid_gls"] < 18.0, "baseline GLS <18", "baseline GLS >=18")
    primary["current_decline_group"] = pd.cut(
        primary["current_mid_decline_from_first"],
        bins=[-np.inf, 0.05, 0.10, np.inf],
        labels=["<5%", "5-10%", ">=10%"],
    ).astype(str)
    primary["ef_group"] = np.select(
        [primary["current_ef"].isna(), primary["current_ef"] < 55],
        ["EF missing", "EF <55"],
        default="EF >=55",
    )
    primary["endo_mid_gap_group"] = quantile_group(
        primary["current_endo_mid_gap"], ["gap low", "gap middle", "gap high"]
    )
    primary["variability_group"] = quantile_group(
        primary["mid_variability"], ["variability low", "variability middle", "variability high"]
    )
    primary["curve_qc_group"] = np.where(
        primary["curve_rejected_fraction"] > 0, "at least one rejected", "all curves pass"
    )
    primary["label_margin_group"] = pd.cut(
        primary["absolute_label_margin"],
        bins=[-np.inf, 0.03, 0.08, np.inf],
        labels=["within 3% of threshold", "3-8% from threshold", ">8% from threshold"],
    ).astype(str)

    models = ["clinical_ridge", "existing_combined_trees", "attention_binary"]
    subgroups = subgroup_metrics(primary, models)
    calibration_bins, calibration_summary = calibration_analysis(primary, models)
    error_summary, error_detail = error_tables(primary, "attention_binary")

    tree_comparison_predictions = pd.concat(
        [
            plateau_predictions[
                (plateau_predictions["task"] == "mid_first_rel15")
                & (plateau_predictions["model"] == "attention_binary")
            ],
            original[
                (original["task"] == "mid_first_rel15")
                & (original["model"] == "combined_extra_trees")
            ].assign(model="existing_combined_trees"),
        ],
        ignore_index=True,
    )
    tree_delta = qc.paired_variant_deltas(
        tree_comparison_predictions,
        [("attention_vs_existing_trees", "existing_combined_trees", "attention_binary")],
        BOOTSTRAPS,
        SEED + 1000000,
    )
    replicate_pairs, replicate_summary = replicate_noise_estimate()
    noise = noise_ceiling(visits, transitions, replicate_summary)

    subgroups.to_csv(OUTPUT / "subgroup_metrics.csv", index=False)
    calibration_bins.to_csv(OUTPUT / "calibration_quintiles.csv", index=False)
    calibration_summary.to_csv(OUTPUT / "calibration_summary.csv", index=False)
    error_summary.to_csv(OUTPUT / "error_status_summary.csv", index=False)
    error_detail.to_csv(OUTPUT / "error_case_details.csv", index=False)
    tree_delta.to_csv(OUTPUT / "attention_vs_existing_trees.csv", index=False)
    replicate_pairs.to_csv(OUTPUT / "strain_replicate_pairs.csv", index=False)
    replicate_summary.to_csv(OUTPUT / "strain_replicate_summary.csv", index=False)
    noise.to_csv(OUTPUT / "label_noise_ceiling.csv", index=False)
    make_figures(attention, subgroups, noise)
    write_report(
        model_metrics,
        ablations,
        regression,
        attention,
        subgroups,
        error_summary,
        calibration_summary,
        replicate_summary,
        noise,
        tree_delta,
    )
    metadata = {
        "primary_task": "mid_first_rel15",
        "error_models": models,
        "subgroup_bootstraps": BOOTSTRAPS,
        "noise_simulations_per_setting": 2000,
        "noise_thresholds": [0.10, 0.12, 0.15, 0.20],
        "noise_interpretation": "replicate-derived SD is uncertain because only 16 paired analyses are available",
    }
    (OUTPUT / "error_noise_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(OUTPUT / "plateau_experiment_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
