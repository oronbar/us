from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

import cardiotoxicity_next_visit_gpu as core
from cardiotoxicity_early_detection import VARIABILITY_SUFFIXES, curve_feature_table


TASK = "mid_first_rel15"
SEED = 20260807


PLAIN_NAMES = {
    "history_visits": "Number of visits observed",
    "days_since_first": "Days since baseline",
    "days_since_previous": "Days since previous visit",
    "has_previous_visit": "Previous visit available",
    "current_mid_gls": "Current Mid-wall GLS magnitude",
    "current_endo_gls": "Current Endocardial GLS magnitude",
    "current_ef": "Current ejection fraction",
    "first_mid_gls": "Baseline Mid-wall GLS magnitude",
    "first_endo_gls": "Baseline Endocardial GLS magnitude",
    "first_ef": "Baseline ejection fraction",
    "current_mid_decline_from_first": "Current relative Mid-GLS decline from baseline",
    "current_endo_decline_from_first": "Current relative Endocardial GLS decline from baseline",
    "current_ef_decline_from_first": "Current relative EF decline from baseline",
    "last_mid_relative_change": "Most recent relative Mid-GLS change",
    "last_endo_relative_change": "Most recent relative Endocardial GLS change",
    "last_ef_relative_change": "Most recent relative EF change",
    "mid_decline_slope_per_100d": "Mid-GLS decline slope per 100 days",
    "endo_decline_slope_per_100d": "Endocardial GLS decline slope per 100 days",
    "ef_decline_slope_per_100d": "EF decline slope per 100 days",
    "current_endo_mid_gap": "Current Endocardial-minus-Mid GLS gap",
    "last_endo_mid_gap_change": "Most recent change in Endocardial-minus-Mid GLS gap",
    "d_tm_sd_gap_dct04": "Change in phase pattern of segmental layer-gap variability",
    "cur_tm_segment_curve_correlation_std": "Variation across segments in Endocardial/Mid curve similarity",
    "d_tm_peak_gap_mean": "Change in mean Endocardial-minus-Mid peak-strain gap",
    "cur_tm_sd_gap_dct06": "Current finer phase pattern of segmental layer-gap variability",
    "cur_tm_mean_gap_dct02": "Broad temporal trend in the current Endocardial-minus-Mid curve gap",
    "cur_tm_mean_shape_gap_dct01": "Overall normalized Endocardial-minus-Mid curve-shape separation",
    "cur_tm_mean_shape_gap_dct04": "Intermediate phase pattern of normalized layer-shape separation",
    "cur_tm_mean_shape_gap_dct07": "Finer temporal pattern of normalized layer-shape separation",
    "cur_var__mid_peak_abs_robust_sd": "Between-segment dispersion of Mid-wall peak strain",
    "cur_var__endo_vendor_peak_systolic_abs_robust_sd": "Between-segment dispersion of vendor Endocardial peak-systolic strain",
    "cur_var__endo_vendor_time_to_peak_norm_circular_std": "Endocardial segment time-to-peak dispersion",
    "cur_var__endo_peak_abs_robust_sd": "Between-segment dispersion of Endocardial peak strain",
    "d_tm_vendor_peak_gap_mean": "Change in mean vendor Endocardial-minus-Mid peak gap",
    "cur_var__mid_vendor_peak_systolic_abs_robust_sd": "Between-segment dispersion of vendor Mid-wall peak-systolic strain",
    "cur_var__endo_impaired_segment_fraction_lt15": "Fraction of Endocardial segments with strain magnitude below 15",
    "cur_var__mid_impaired_segment_fraction_lt15": "Fraction of Mid-wall segments with strain magnitude below 15",
    "cur_var__mid_within_ring_peak_robust_sd_mean": "Mean within-ring dispersion of Mid-wall peak strain",
    "cur_var__mid_within_view_peak_robust_sd_mean": "Mean within-view dispersion of Mid-wall peak strain",
    "cur_var__endo_within_view_peak_robust_sd_mean": "Mean within-view dispersion of Endocardial peak strain",
    "cur_tm_sd_gap_dct04": "Current phase pattern of segmental layer-gap variability (DCT 4)",
    "cur_tm_peak_ratio_std": "Across-segment variation in Endocardial/Mid peak-strain ratio",
    "cur_tm_shape_rms_std": "Across-segment variation in Endocardial/Mid curve-shape separation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transitions",
        type=Path,
        default=Path("cardiotoxicity_next_visit_gpu_results/next_visit_transitions.parquet"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("cardiotoxicity_next_visit_gpu_results/feature_manifest.csv"),
    )
    parser.add_argument(
        "--visits",
        type=Path,
        default=Path("amber_full_105_preprocessed/Ichilov_july_visits.parquet"),
    )
    parser.add_argument(
        "--curves",
        type=Path,
        default=Path("amber_full_105_preprocessed/Ichilov_july_dataset.parquet"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("cardiotoxicity_pre_event_feature_results")
    )
    parser.add_argument("--bootstraps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--reuse-rankings", action="store_true")
    return parser.parse_args()


def family(feature: str) -> str:
    if feature.startswith(("cur_tm_", "d_tm_")):
        return "Endo-Mid"
    if feature.startswith(("cur_var__", "d_var__")):
        return "Segment variability"
    return "Clinical/trajectory"


def plain_name(feature: str) -> str:
    if feature in PLAIN_NAMES:
        return PLAIN_NAMES[feature]
    value = feature
    timing = ""
    if value.startswith("cur_var__"):
        timing, value = "Current: ", value[len("cur_var__") :]
    elif value.startswith("d_var__"):
        timing, value = "Change since previous visit: ", value[len("d_var__") :]
    elif value.startswith("cur_tm_"):
        timing, value = "Current Endocardial-Mid: ", value[len("cur_tm_") :]
    elif value.startswith("d_tm_"):
        timing, value = "Change in Endocardial-Mid: ", value[len("d_tm_") :]
    if value.startswith("endo_"):
        value = "Endocardial " + value[len("endo_") :]
    elif value.startswith("mid_"):
        value = "Mid-wall " + value[len("mid_") :]
    replacements = {
        "ttp": "time-to-peak",
        "dct": "DCT coefficient ",
        "rms": "RMS",
        "rmse": "RMSE",
        "robust_sd": "robust dispersion",
        "circular_std": "circular dispersion",
    }
    phrase = value
    for old, new in replacements.items():
        phrase = phrase.replace(old, new)
    return timing + phrase.replace("_", " ").strip().capitalize()


def visit_order_stratified_auc(frame: pd.DataFrame, feature: str) -> tuple[float, int, int, int]:
    data = frame[["current_visit_order", "label", feature]].replace([np.inf, -np.inf], np.nan).dropna()
    aucs: list[float] = []
    weights: list[int] = []
    n_case = 0
    n_control = 0
    strata = 0
    for _, group in data.groupby("current_visit_order"):
        y = group["label"].to_numpy(int)
        if np.unique(y).size < 2:
            continue
        score = group[feature].to_numpy(float)
        aucs.append(float(roc_auc_score(y, score)))
        cases = int(y.sum())
        weights.append(cases)
        n_case += cases
        n_control += int(len(y) - cases)
        strata += 1
    if not aucs:
        return np.nan, n_case, n_control, strata
    return float(np.average(aucs, weights=weights)), n_case, n_control, strata


def bootstrap_auc_ci(
    frame: pd.DataFrame,
    feature: str,
    direction: int,
    bootstraps: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    data = frame[["patient_id", "current_visit_order", "label", feature]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    patient_ids = data["patient_id"].astype(str).unique()
    patient_code = pd.Categorical(data["patient_id"].astype(str), categories=patient_ids).codes
    visit_order = data["current_visit_order"].to_numpy(int)
    label = data["label"].to_numpy(int)
    score = data[feature].to_numpy(float)
    orders = np.unique(visit_order)
    values: list[float] = []
    for _ in range(bootstraps):
        multiplicity = rng.multinomial(len(patient_ids), np.full(len(patient_ids), 1.0 / len(patient_ids)))
        row_weight = multiplicity[patient_code].astype(float)
        aucs: list[float] = []
        case_weights: list[float] = []
        for order in orders:
            use = (visit_order == order) & (row_weight > 0)
            if not np.any(use):
                continue
            y = label[use]
            weights = row_weight[use]
            positive_weight = float(weights[y == 1].sum())
            negative_weight = float(weights[y == 0].sum())
            if positive_weight <= 0 or negative_weight <= 0:
                continue
            aucs.append(float(roc_auc_score(y, score[use], sample_weight=weights)))
            case_weights.append(positive_weight)
        if aucs:
            auc = float(np.average(aucs, weights=case_weights))
            values.append(auc if direction > 0 else 1.0 - auc)
    if len(values) < max(30, bootstraps // 3):
        return np.nan, np.nan
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def rank_features(
    cohort: pd.DataFrame,
    features: list[str],
    bootstraps: int,
    seed: int,
    ci_top: int = 50,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, feature in enumerate(features):
        if feature not in cohort.columns:
            continue
        auc, n_case, n_control, strata = visit_order_stratified_auc(cohort, feature)
        if not np.isfinite(auc) or n_case < 8 or n_control < 8:
            continue
        direction = 1 if auc >= 0.5 else -1
        oriented = auc if direction > 0 else 1.0 - auc
        rows.append(
            {
                "feature": feature,
                "feature_name": plain_name(feature),
                "family": family(feature),
                "direction_in_pre_event_visit": "higher" if direction > 0 else "lower",
                "visit_order_stratified_auc": oriented,
                "auc_ci_low": np.nan,
                "auc_ci_high": np.nan,
                "case_rows_used": n_case,
                "control_rows_used": n_control,
                "visit_order_strata": strata,
                "outcome_coupled": feature
                in {
                    "current_mid_gls",
                    "first_mid_gls",
                    "current_mid_decline_from_first",
                    "last_mid_relative_change",
                    "mid_decline_slope_per_100d",
                    "current_mid_decline_from_roll2",
                    "current_mid_decline_from_roll3",
                },
            }
        )
    result = pd.DataFrame(rows).sort_values("visit_order_stratified_auc", ascending=False).reset_index(drop=True)
    # Bootstrap the strongest candidates only. Screening every feature is fast;
    # resampling all 269 features adds computation without changing which ones
    # need intervals for interpretation.
    for rank_index in range(min(ci_top, len(result))):
        feature = str(result.loc[rank_index, "feature"])
        direction = 1 if result.loc[rank_index, "direction_in_pre_event_visit"] == "higher" else -1
        low, high = bootstrap_auc_ci(
            cohort,
            feature,
            direction,
            bootstraps,
            seed + 1009 * (features.index(feature) + 1),
        )
        result.loc[rank_index, "auc_ci_low"] = low
        result.loc[rank_index, "auc_ci_high"] = high
    result = result.sort_values(
        ["visit_order_stratified_auc", "auc_ci_low"], ascending=False
    )
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def build_all_visit_features(visits: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    visits = visits.sort_values(["patient_id", "visit_order"]).copy()
    tm = tm.set_index("visit_id")
    tm_columns = [column for column in tm.columns if column.startswith("tm_")]
    variability_columns = [
        f"{layer}_{suffix}"
        for layer in ["endo", "mid"]
        for suffix in VARIABILITY_SUFFIXES
        if f"{layer}_{suffix}" in visits.columns
    ]
    rows: list[dict[str, object]] = []
    for patient_id, patient in visits.groupby("patient_id", sort=False):
        patient = patient.sort_values("visit_order").reset_index(drop=True)
        first = patient.iloc[0]
        first_values = {metric: core.metric_value(first, metric) for metric in ["mid", "endo", "ef"]}
        for current_index in range(len(patient)):
            current = patient.iloc[current_index]
            previous = patient.iloc[max(0, current_index - 1)]
            current_id = str(current["visit_id"])
            previous_id = str(previous["visit_id"])
            if current_id not in tm.index or previous_id not in tm.index:
                continue
            current_values = {metric: core.metric_value(current, metric) for metric in ["mid", "endo", "ef"]}
            previous_values = {metric: core.metric_value(previous, metric) for metric in ["mid", "endo", "ef"]}
            dt = max(
                core.safe_float(current["days_since_baseline"])
                - core.safe_float(previous["days_since_baseline"]),
                1.0,
            )
            row: dict[str, object] = {
                "patient_id": str(patient_id),
                "visit_id": current_id,
                "visit_order": int(current["visit_order"]),
                "history_visits": int(current_index + 1),
                "days_since_first": core.safe_float(current["days_since_baseline"]),
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
                "current_ef_decline_from_first": (
                    1.0 - current_values["ef"] / first_values["ef"]
                    if np.isfinite(current_values["ef"]) and np.isfinite(first_values["ef"])
                    else np.nan
                ),
                "last_mid_relative_change": 0.0 if current_index == 0 else 1.0 - current_values["mid"] / previous_values["mid"],
                "last_endo_relative_change": 0.0 if current_index == 0 else 1.0 - current_values["endo"] / previous_values["endo"],
                "last_ef_relative_change": (
                    0.0
                    if current_index == 0
                    or not np.isfinite(current_values["ef"])
                    or not np.isfinite(previous_values["ef"])
                    else 1.0 - current_values["ef"] / previous_values["ef"]
                ),
                "mid_decline_slope_per_100d": 0.0 if current_index == 0 else (1.0 - current_values["mid"] / previous_values["mid"]) * 100.0 / dt,
                "endo_decline_slope_per_100d": 0.0 if current_index == 0 else (1.0 - current_values["endo"] / previous_values["endo"]) * 100.0 / dt,
                "ef_decline_slope_per_100d": (
                    0.0
                    if current_index == 0
                    or not np.isfinite(current_values["ef"])
                    or not np.isfinite(previous_values["ef"])
                    else (1.0 - current_values["ef"] / previous_values["ef"]) * 100.0 / dt
                ),
                "current_endo_mid_gap": current_values["endo"] - current_values["mid"],
                "last_endo_mid_gap_change": (
                    0.0
                    if current_index == 0
                    else (current_values["endo"] - current_values["mid"])
                    - (previous_values["endo"] - previous_values["mid"])
                ),
            }
            for n in [2, 3]:
                for metric in ["mid", "endo", "ef"]:
                    name = f"current_{metric}_decline_from_roll{n}"
                    if current_index + 1 >= n:
                        history = patient.iloc[current_index - n + 1 : current_index + 1]
                        values = np.asarray(
                            [core.metric_value(history_row, metric) for _, history_row in history.iterrows()],
                            dtype=float,
                        )
                        baseline = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
                        row[name] = (
                            1.0 - current_values[metric] / baseline
                            if np.isfinite(current_values[metric]) and np.isfinite(baseline)
                            else np.nan
                        )
                    else:
                        row[name] = np.nan
            for column in variability_columns:
                current_value = core.safe_float(current[column])
                previous_value = core.safe_float(previous[column])
                row[f"cur_var__{column}"] = current_value
                row[f"d_var__{column}"] = 0.0 if current_index == 0 else current_value - previous_value
            for column in tm_columns:
                current_value = tm.loc[current_id, column]
                previous_value = tm.loc[previous_id, column]
                if isinstance(current_value, pd.Series):
                    current_value = current_value.mean()
                if isinstance(previous_value, pd.Series):
                    previous_value = previous_value.mean()
                current_value = core.safe_float(current_value)
                previous_value = core.safe_float(previous_value)
                row[f"cur_{column}"] = current_value
                row[f"d_{column}"] = 0.0 if current_index == 0 else current_value - previous_value
            rows.append(row)
    return pd.DataFrame(rows)


def aligned_event_values(
    visit_features: pd.DataFrame,
    events: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    event_lookup = events.set_index("patient_id")[["target_visit_id", "target_visit_order"]]
    rows = []
    for patient_id, patient in visit_features.groupby("patient_id"):
        if patient_id not in event_lookup.index:
            continue
        event_order = int(event_lookup.loc[patient_id, "target_visit_order"])
        event_visit_id = str(event_lookup.loc[patient_id, "target_visit_id"])
        patient = patient[patient["visit_order"] <= event_order]
        for _, visit in patient.iterrows():
            relative = int(visit["visit_order"] - event_order)
            if relative < -3:
                continue
            row = {
                "patient_id": patient_id,
                "visit_id": visit["visit_id"],
                "visit_order": int(visit["visit_order"]),
                "relative_visit": relative,
                "event_visit_id": event_visit_id,
                "event_visit_order": event_order,
            }
            for feature in selected_features:
                row[feature] = visit.get(feature, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def within_case_change_ranking(
    aligned: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    before_two = aligned[aligned["relative_visit"] == -2].set_index("patient_id")
    before_one = aligned[aligned["relative_visit"] == -1].set_index("patient_id")
    common = before_two.index.intersection(before_one.index)
    rows = []
    for feature in features:
        if feature not in before_one.columns or not feature.startswith("cur_"):
            continue
        paired = pd.DataFrame(
            {"minus2": before_two.loc[common, feature], "minus1": before_one.loc[common, feature]}
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(paired) < 10:
            continue
        delta = paired["minus1"] - paired["minus2"]
        pooled = pd.concat([paired["minus2"], paired["minus1"]], ignore_index=True)
        scale = float(pooled.quantile(0.75) - pooled.quantile(0.25))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(pooled.std(ddof=1))
        if not np.isfinite(scale) or scale <= 1e-12:
            continue
        try:
            p_value = float(wilcoxon(delta).pvalue) if np.any(np.abs(delta) > 0) else 1.0
        except ValueError:
            p_value = np.nan
        rows.append(
            {
                "feature": feature,
                "feature_name": plain_name(feature),
                "family": family(feature),
                "patients": len(paired),
                "median_at_minus2": float(paired["minus2"].median()),
                "median_at_minus1": float(paired["minus1"].median()),
                "median_change": float(delta.median()),
                "median_absolute_change": float(delta.abs().median()),
                "median_absolute_change_over_pooled_iqr": float(delta.abs().median() / scale),
                "wilcoxon_p_value": p_value,
            }
        )
    result = pd.DataFrame(rows).sort_values("median_absolute_change_over_pooled_iqr", ascending=False)
    p = result["wilcoxon_p_value"].to_numpy(float)
    valid = np.isfinite(p)
    q = np.full(len(result), np.nan)
    if valid.any():
        valid_indices = np.flatnonzero(valid)
        order = valid_indices[np.argsort(p[valid])]
        adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        q[order] = np.minimum(adjusted, 1.0)
    result["wilcoxon_fdr_q_value"] = q
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def plot_auc(ranking: pd.DataFrame, output: Path, title: str) -> None:
    engineered = ranking[~ranking["outcome_coupled"]].head(12).sort_values(
        "visit_order_stratified_auc"
    )
    fig, ax = plt.subplots(figsize=(11, 7.2))
    colors = engineered["family"].map(
        {"Clinical/trajectory": "#4776E6", "Endo-Mid": "#8E54E9", "Segment variability": "#00A896"}
    )
    y = np.arange(len(engineered))
    x = engineered["visit_order_stratified_auc"].to_numpy(float)
    low = engineered["auc_ci_low"].to_numpy(float)
    high = engineered["auc_ci_high"].to_numpy(float)
    ax.barh(y, x - 0.5, left=0.5, color=colors, alpha=0.9)
    ax.errorbar(x, y, xerr=np.vstack([x - low, high - x]), fmt="none", ecolor="#263238", capsize=3)
    ax.axvline(0.5, color="#555555", linestyle="--", linewidth=1.2)
    ax.set_yticks(y, engineered["feature_name"])
    ax.set_xlim(0.48, max(0.76, float(np.nanmax(high)) + 0.02))
    ax.set_xlabel("Visit-order-stratified univariate AUC (95% patient-bootstrap CI)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(aligned: pd.DataFrame, features: list[str], output: Path) -> None:
    ncols = 2
    nrows = math.ceil(len(features) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.1 * nrows), squeeze=False)
    x_values = [-3, -2, -1, 0]
    for ax, feature in zip(axes.ravel(), features):
        summary = []
        for relative in x_values:
            values = aligned.loc[aligned["relative_visit"] == relative, feature].dropna().to_numpy(float)
            if len(values):
                summary.append(
                    (relative, len(values), np.median(values), np.quantile(values, 0.25), np.quantile(values, 0.75))
                )
        if summary:
            table = pd.DataFrame(summary, columns=["x", "n", "median", "q1", "q3"])
            ax.plot(table["x"], table["median"], marker="o", color="#7B2CBF", linewidth=2.4)
            ax.fill_between(table["x"], table["q1"], table["q3"], color="#CDB4DB", alpha=0.45)
            for _, row in table.iterrows():
                ax.annotate(f"n={int(row['n'])}", (row["x"], row["median"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        ax.axvline(-1, color="#F77F00", linestyle="--", linewidth=1.2)
        ax.axvline(0, color="#D62828", linestyle="--", linewidth=1.2)
        ax.set_title(plain_name(feature), fontsize=10.5)
        ax.set_xticks(x_values, ["3 before", "2 before", "1 before", "Event"])
        ax.grid(alpha=0.18)
    for ax in axes.ravel()[len(features) :]:
        ax.axis("off")
    fig.suptitle(
        "Event-aligned trajectories for 23 later first-deterioration events (visit 3 or 4)\nMedian and interquartile range; the event visit is descriptive only",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_report(
    output: Path,
    transitions: pd.DataFrame,
    overall: pd.DataFrame,
    later: pd.DataFrame,
    within: pd.DataFrame,
    selected: list[str],
) -> None:
    events = transitions[transitions["label"] == 1]
    event_counts = events["target_visit_order"].value_counts().sort_index()

    def markdown_table(frame: pd.DataFrame) -> str:
        headers = [str(column) for column in frame.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in frame.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
        return "\n".join(lines)

    def table(frame: pd.DataFrame, n: int = 10) -> str:
        view = frame.head(n).copy()
        view["AUC (95% CI)"] = view.apply(
            lambda row: f"{row['visit_order_stratified_auc']:.3f} ({row['auc_ci_low']:.3f}-{row['auc_ci_high']:.3f})",
            axis=1,
        )
        view["Direction"] = view["direction_in_pre_event_visit"]
        view["Cases"] = view["case_rows_used"]
        return markdown_table(view[["rank", "feature_name", "family", "Direction", "AUC (95% CI)", "Cases"]])

    lines = [
        "# Feature behavior before first 15% relative Mid-GLS deterioration",
        "",
        "## Question and design",
        "",
        "The event is the first visit with at least a 15% relative drop in Mid-GLS magnitude from the first visit. "
        "For every event patient, the visit immediately before that event is the positive pre-event observation. "
        "Feature ranking never uses the event visit itself.",
        "",
        "Each feature was evaluated alone using ROC AUC. AUCs were calculated separately within current visit order and averaged, "
        "so ordinary differences between visit 1, visit 2, and visit 3 do not create a false signal. Confidence intervals use patient-level bootstrap resampling.",
        "",
        "## Cohort",
        "",
        f"- Eligible transitions: {len(transitions)}",
        f"- First-deterioration events: {int(events.shape[0])} in {events['patient_id'].nunique()} patients",
        f"- Non-event eligible transitions: {int((transitions['label'] == 0).sum())}",
        "- Event visit distribution: " + ", ".join(f"V{int(k)}: {int(v)}" for k, v in event_counts.items()),
        f"- Later events with a genuine V(t-2) to V(t-1) comparison: {int((events['current_visit_order'] >= 2).sum())}",
        "",
        "**Key limitation:** events at visit 2 have only baseline as the one-visit-early observation. They can test baseline risk markers, "
        "but they cannot show an evolving within-patient warning signal.",
        "",
        "## Strongest one-visit-early features: all 49 events",
        "",
        table(overall),
        "",
        "Outcome-coupled Mid-GLS features were flagged in the CSV because they are mathematically related to the label definition. "
        "The table above prioritizes the remaining features for biological interpretation.",
        "",
        "## Sensitivity analysis: 23 events at visit 3 or 4",
        "",
        table(later),
        "",
        "This smaller analysis is the better test of an evolving warning signal, but its confidence intervals are wider.",
        "",
        "## Largest within-patient changes from two visits before to one visit before",
        "",
    ]
    within_view = within.head(10).copy()
    if len(within_view):
        within_view["Median change"] = within_view["median_change"].map(lambda value: f"{value:.4g}")
        within_view["Absolute change / IQR"] = within_view[
            "median_absolute_change_over_pooled_iqr"
        ].map(lambda value: f"{value:.2f}")
        within_view["FDR q"] = within_view["wilcoxon_fdr_q_value"].map(
            lambda value: f"{value:.3f}" if np.isfinite(value) else "NA"
        )
        lines.append(markdown_table(within_view[
            ["rank", "feature_name", "family", "patients", "Median change", "Absolute change / IQR", "FDR q"]
        ]))
    lines.extend(
        [
            "",
            "This within-patient ranking measures amount of change, not predictive accuracy. A feature can change strongly yet still be noisy across patients.",
            "None of the paired changes survived false-discovery-rate correction "
            f"(minimum q={within['wilcoxon_fdr_q_value'].min():.3f}). Therefore, the apparent trajectories are descriptive, not confirmed group-wide trends.",
            "",
            "## Interpretation",
            "",
            "A feature is a plausible one-visit-early marker only when it has useful AUC, a reasonably stable bootstrap interval, "
            "and a coherent trajectory before the event. These results are exploratory because many features were screened on the same cohort; "
            "the strongest candidates should be tested in patient-held-out modeling or an external cohort.",
            "",
            "Trajectory figure features: " + "; ".join(plain_name(feature) for feature in selected) + ".",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)

    raw = pd.read_parquet(args.transitions)
    eligible = raw[raw[f"mask__{TASK}"].astype(bool)].copy()
    eligible["label"] = eligible[f"label__{TASK}"].astype(int)
    manifest = pd.read_csv(args.manifest)
    features = manifest.loc[manifest["feature_set"] == "combined", "feature"].drop_duplicates().tolist()
    features = [feature for feature in features if feature in eligible.columns]

    later_cohort = eligible[eligible["current_visit_order"] >= 2].copy()
    overall_path = args.output / "pre_event_feature_ranking_all_events.csv"
    later_path = args.output / "pre_event_feature_ranking_later_events.csv"
    if args.reuse_rankings and overall_path.exists() and later_path.exists():
        overall = pd.read_csv(overall_path)
        later = pd.read_csv(later_path)
    else:
        overall = rank_features(eligible, features, args.bootstraps, args.seed)
        later = rank_features(later_cohort, features, args.bootstraps, args.seed + 1_000_000)
        overall.to_csv(overall_path, index=False)
        later.to_csv(later_path, index=False)
    overall["feature_name"] = overall["feature"].map(plain_name)
    later["feature_name"] = later["feature"].map(plain_name)
    overall.to_csv(overall_path, index=False)
    later.to_csv(later_path, index=False)

    tm = curve_feature_table(args.curves)
    visits = pd.read_parquet(args.visits)
    visit_features = build_all_visit_features(visits, tm)

    # Verify the reconstructed values against the saved transition table.
    overlap = eligible.merge(
        visit_features,
        left_on=["patient_id", "current_visit_id"],
        right_on=["patient_id", "visit_id"],
        suffixes=("_saved", "_rebuilt"),
    )
    checks = []
    for feature in features:
        saved = f"{feature}_saved"
        rebuilt = f"{feature}_rebuilt"
        if saved in overlap and rebuilt in overlap:
            delta = np.abs(overlap[saved].to_numpy(float) - overlap[rebuilt].to_numpy(float))
            checks.append({"feature": feature, "max_absolute_difference": float(np.nanmax(delta))})
    pd.DataFrame(checks).to_csv(args.output / "visit_feature_reconstruction_check.csv", index=False)

    events = eligible[eligible["label"] == 1][
        ["patient_id", "target_visit_id", "target_visit_order"]
    ].copy()
    engineered = later[(~later["outcome_coupled"]) & (later["family"] != "Clinical/trajectory")]
    selected = engineered.head(6)["feature"].tolist()
    if len(selected) < 6:
        selected += [
            feature
            for feature in overall.loc[~overall["outcome_coupled"], "feature"]
            if feature not in selected
        ][: 6 - len(selected)]
    all_aligned = aligned_event_values(visit_features, events, features)
    all_aligned.to_csv(args.output / "event_aligned_all_feature_values.csv", index=False)
    trajectory_aligned = all_aligned[all_aligned["event_visit_order"] >= 3].copy()
    selected_aligned = trajectory_aligned[
        ["patient_id", "visit_id", "visit_order", "relative_visit", "event_visit_id", "event_visit_order"]
        + selected
    ]
    selected_aligned.to_csv(args.output / "event_aligned_selected_feature_values.csv", index=False)
    summary_rows = []
    for feature in selected:
        for relative, group in selected_aligned.groupby("relative_visit"):
            values = group[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values):
                summary_rows.append(
                    {
                        "feature": feature,
                        "feature_name": plain_name(feature),
                        "relative_visit": int(relative),
                        "patients": len(values),
                        "median": float(values.median()),
                        "q1": float(values.quantile(0.25)),
                        "q3": float(values.quantile(0.75)),
                    }
                )
    pd.DataFrame(summary_rows).to_csv(args.output / "event_aligned_selected_feature_summary.csv", index=False)

    within = within_case_change_ranking(all_aligned, features)
    within.to_csv(args.output / "within_case_change_ranking.csv", index=False)

    plot_auc(
        overall[overall["case_rows_used"] >= 39],
        figures / "pre_event_feature_auc_all_events.png",
        "One-visit-early features across all 49 first-deterioration events",
    )
    plot_auc(
        later,
        figures / "pre_event_feature_auc_later_events.png",
        "One-visit-early features in 23 later events (visit 3 or 4)",
    )
    plot_trajectories(selected_aligned, selected, figures / "event_aligned_feature_trajectories.png")
    make_report(
        args.output / "pre_event_feature_report.md",
        eligible,
        overall[(~overall["outcome_coupled"]) & (overall["case_rows_used"] >= 39)],
        later[~later["outcome_coupled"]],
        within,
        selected,
    )
    print(f"Saved analysis to {args.output.resolve()}")
    print(overall[~overall["outcome_coupled"]].head(12).to_string(index=False))
    print("\nLater-event sensitivity:")
    print(later[~later["outcome_coupled"]].head(12).to_string(index=False))
    print("\nSelected trajectory features:", selected)


if __name__ == "__main__":
    main()
