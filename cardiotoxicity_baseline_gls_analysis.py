from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


INPUT = Path(r"D:\us\cardiotoxicity_next_visit_gpu_results\next_visit_transitions.parquet")
OUTPUT = Path(r"D:\us\cardiotoxicity_feature_importance_results")
TASK = "mid_first_rel15"
SEED = 20260805
BOOTSTRAPS = 2000


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


def estimates(frame: pd.DataFrame, x_col: str) -> dict[str, float]:
    clean = frame[[x_col, "decline", "patient_id"]].dropna()
    x = clean[x_col].to_numpy(float)
    y = clean["decline"].to_numpy(float)
    return {
        "n": int(len(clean)),
        "patients": int(clean["patient_id"].nunique()),
        "pearson_r": float(pearsonr(x, y).statistic),
        "spearman_rho": float(spearmanr(x, y).statistic),
        "slope_fraction_per_gls_point": linear_slope(x, y),
        "intercept": float(np.polyfit(x, y, 1)[1]),
    }


def bootstrap_interval(
    frame: pd.DataFrame,
    x_col: str,
    cluster: bool,
) -> dict[str, list[float]]:
    clean = frame[[x_col, "decline", "patient_id"]].dropna().reset_index(drop=True)
    rng = np.random.default_rng(SEED + (0 if x_col == "first_mid_gls" else 1) + (10 if cluster else 20))
    patients = clean["patient_id"].unique()
    values = {"pearson_r": [], "spearman_rho": [], "slope_fraction_per_gls_point": []}
    patient_rows = {patient: np.flatnonzero(clean["patient_id"].to_numpy() == patient) for patient in patients}
    for _ in range(BOOTSTRAPS):
        if cluster:
            sampled = rng.choice(patients, size=len(patients), replace=True)
            index = np.concatenate([patient_rows[patient] for patient in sampled])
        else:
            index = rng.integers(0, len(clean), size=len(clean))
        x = clean.iloc[index][x_col].to_numpy(float)
        y = clean.iloc[index]["decline"].to_numpy(float)
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        values["pearson_r"].append(float(pearsonr(x, y).statistic))
        values["spearman_rho"].append(float(spearmanr(x, y).statistic))
        values["slope_fraction_per_gls_point"].append(linear_slope(x, y))
    return {
        key: [float(np.quantile(series, 0.025)), float(np.quantile(series, 0.975))]
        for key, series in values.items()
    }


def summarize(frame: pd.DataFrame, x_col: str, cluster: bool) -> dict[str, object]:
    result = estimates(frame, x_col)
    result["bootstrap_95_ci"] = bootstrap_interval(frame, x_col, cluster=cluster)
    return result


def point_records(frame: pd.DataFrame, scope: str) -> list[dict[str, object]]:
    columns = [
        "patient_id",
        "transition_id",
        "target_visit_order",
        "first_mid_gls",
        "first_endo_gls",
        "decline",
        "event",
    ]
    records = []
    for row in frame[columns].itertuples(index=False):
        records.append(
            {
                "scope": scope,
                "patient": str(row.patient_id),
                "transition": str(row.transition_id),
                "targetVisit": int(row.target_visit_order),
                "baselineMid": round(float(row.first_mid_gls), 3),
                "baselineEndo": round(float(row.first_endo_gls), 3),
                "declinePct": round(100.0 * float(row.decline), 3),
                "event": bool(row.event),
            }
        )
    return records


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(INPUT)
    eligible = data.loc[data[f"mask__{TASK}"].astype(bool)].copy()
    eligible["decline"] = eligible[f"decline__{TASK}"].astype(float)
    eligible["event"] = eligible[f"label__{TASK}"].astype(bool)
    eligible = eligible.dropna(subset=["first_mid_gls", "first_endo_gls", "decline"])

    final = (
        eligible.sort_values(["patient_id", "target_visit_order"])
        .groupby("patient_id", as_index=False, sort=False)
        .tail(1)
        .copy()
    )

    summary: dict[str, object] = {
        "task": "Next-visit relative Mid-GLS decline from first-visit baseline",
        "transition_rows": int(len(eligible)),
        "patients": int(eligible["patient_id"].nunique()),
        "events": int(eligible["event"].sum()),
        "final_visit_rows": int(len(final)),
        "baseline_mid_endo_patient_pearson_r": float(
            pearsonr(final["first_mid_gls"], final["first_endo_gls"]).statistic
        ),
        "transition_level": {
            "baseline_mid": summarize(eligible, "first_mid_gls", cluster=True),
            "baseline_endo": summarize(eligible, "first_endo_gls", cluster=True),
        },
        "final_followup_patient_level": {
            "baseline_mid": summarize(final, "first_mid_gls", cluster=False),
            "baseline_endo": summarize(final, "first_endo_gls", cluster=False),
        },
        "single_feature_event_prediction": {},
        "transition_event_rate_by_baseline_quartile": {},
    }

    labels = eligible["event"].astype(int).to_numpy()
    for column, key in [("first_mid_gls", "baseline_mid"), ("first_endo_gls", "baseline_endo")]:
        scores = eligible[column].to_numpy(float)
        summary["single_feature_event_prediction"][key] = {
            "auc": float(roc_auc_score(labels, scores)),
            "ap": float(average_precision_score(labels, scores)),
        }
        quartile = pd.qcut(eligible[column], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        rates = eligible.assign(quartile=quartile).groupby("quartile", observed=True).agg(
            n=("event", "size"), event_rate=("event", "mean"), baseline_min=(column, "min"), baseline_max=(column, "max")
        )
        summary["transition_event_rate_by_baseline_quartile"][key] = [
            {
                "quartile": str(index),
                "n": int(row.n),
                "event_rate": float(row.event_rate),
                "baseline_min": float(row.baseline_min),
                "baseline_max": float(row.baseline_max),
            }
            for index, row in rates.iterrows()
        ]

    points = point_records(eligible, "All eligible next visits") + point_records(final, "Final follow-up per patient")
    (OUTPUT / "baseline_gls_vs_relative_drop_points.json").write_text(json.dumps(points, separators=(",", ":")), encoding="utf-8")
    (OUTPUT / "baseline_gls_vs_relative_drop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
