"""Paired held-out comparison of R-wave and HR-estimated PanEcho sampling."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score


ROOT = Path(r"D:\DS\ichilov3_rwave_panecho_20261005")
OLD = Path(r"D:\DS\ichilov3_temporal_trial_20260927")
SELECTED = Path(r"D:\DS\ichilov3_stage2_padded_20260926\full_manifest.json")
ALIGN = Path(r"D:\DS\ichilov3_aligned_20260927")


def bootstrap_delta(frame, actual, old, new, metric, seed=20261005, draws=2000):
    y = frame[actual].to_numpy()
    p = frame[old].to_numpy()
    q = frame[new].to_numpy()
    groups = frame.patient_id.to_numpy(str)
    pts = np.unique(groups)
    indices = {pt: np.flatnonzero(groups == pt) for pt in pts}
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        take = np.concatenate([indices[pt] for pt in rng.choice(pts, len(pts), replace=True)])
        if metric is roc_auc_score and len(np.unique(y[take])) < 2:
            continue
        differences.append(metric(y[take], q[take]) - metric(y[take], p[take]))
    return dict(reference=float(metric(y, p)), rwave=float(metric(y, q)),
                difference=float(metric(y, q) - metric(y, p)),
                ci95=[float(v) for v in np.quantile(differences, [.025, .975])])


def corr(y, pred):
    return np.corrcoef(y, pred)[0, 1]


def main():
    if not (ROOT / "run_complete.json").exists():
        raise RuntimeError("Trial has not completed")
    records = json.loads(SELECTED.read_text(encoding="utf-8"))
    by_visit = {}
    for record in records:
        key = (record["patient"], record["visit_date"])
        by_visit.setdefault(key, set()).add(record["manufacturer"])
    ge_visits = {key for key, vendors in by_visit.items()
                 if vendors and all("GE" in v.upper() for v in vendors)}
    audit = pd.read_parquet(ROOT / "sampling_audit.parquet")
    all_r = audit.groupby(["patient", "visit_date"]).reason.apply(
        lambda values: values.eq("rwave_first_to_last").all())
    all_r_visits = set(all_r[all_r].index)

    old = pd.read_parquet(OLD / "physiology" / "oof_predictions.parquet")
    new = pd.read_parquet(ROOT / "physiology" / "oof_predictions.parquet")
    key = ["target", "model", "visit_id", "patient_id", "value"]
    physiology = old[key + ["prediction"]].merge(
        new[key + ["prediction"]], on=key, suffixes=("_old", "_rwave"),
        validate="one_to_one")
    matched = pd.read_parquet(ALIGN / "visit_alignment.parquet")[
        ["visit_id", "patient_id", "visit_date"]]
    physiology = physiology.merge(matched, on=["visit_id", "patient_id"],
                                  validate="many_to_one")
    physiology = physiology[physiology.model.eq("panecho")].copy()
    physiology["ge"] = [(p, str(d)) in ge_visits for p, d in zip(
        physiology.patient_id, physiology.visit_date)]
    physiology["all_three_rwave"] = [(p, str(d)) in all_r_visits for p, d in zip(
        physiology.patient_id, physiology.visit_date)]
    gls_rows = []
    for target in ("mid_gls", "endo_gls", "ef"):
        source = physiology[physiology.target.eq(target)]
        for subgroup, frame in (("all", source), ("GE", source.loc[source["ge"]]),
                                ("all_three_R", source[source.all_three_rwave])):
            if len(frame) < 3:
                continue
            row = dict(target=target, subgroup=subgroup, visits=len(frame),
                       patients=frame.patient_id.nunique())
            row["mae"] = bootstrap_delta(frame, "value", "prediction_old",
                                         "prediction_rwave", mean_absolute_error)
            row["pearson_r"] = bootstrap_delta(frame, "value", "prediction_old",
                                               "prediction_rwave", corr)
            gls_rows.append(row)

    old = pd.read_parquet(OLD / "late_fusion" / "paired_oof_predictions.parquet")
    new = pd.read_parquet(ROOT / "late_fusion" / "paired_oof_predictions.parquet")
    key = ["model", "transition_id", "patient_id", "label"]
    future = old[key + ["score"]].merge(new[key + ["score"]], on=key,
                                       suffixes=("_old", "_rwave"), validate="one_to_one")
    cohort = pd.read_parquet(ALIGN / "cohort.parquet")
    future = future.merge(cohort[["transition_id", "current_visit_date"]],
                          on="transition_id", validate="many_to_one")
    future["ge"] = [(p, str(d)) in ge_visits for p, d in zip(
        future.patient_id, future.current_visit_date)]
    future["all_three_rwave"] = [(p, str(d)) in all_r_visits for p, d in zip(
        future.patient_id, future.current_visit_date)]
    future_rows = []
    for model in ("cnn_moment_baseline", "video_panecho", "late_panecho_video25"):
        source = future[future.model.eq(model)]
        for subgroup, frame in (("all", source), ("GE", source.loc[source["ge"]]),
                                ("all_three_R", source[source.all_three_rwave])):
            if len(frame) < 3 or frame.label.nunique() < 2:
                continue
            row = dict(model=model, subgroup=subgroup, transitions=len(frame),
                       events=int(frame.label.sum()), patients=frame.patient_id.nunique())
            row["auroc"] = bootstrap_delta(frame, "label", "score_old",
                                           "score_rwave", roc_auc_score)
            row["ap"] = bootstrap_delta(frame, "label", "score_old",
                                        "score_rwave", average_precision_score)
            future_rows.append(row)

    result = {"same_visit": gls_rows, "next_visit": future_rows,
              "interpretation_limit": "R-wave arm uses one window while the prior HR-estimated arm averages up to four; both timing and aggregation change."}
    (ROOT / "paired_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# R-wave-anchored PanEcho experiment", "",
             "Frozen PanEcho, native color and unchanged crops/checkpoint. Sixteen native frames from first to last R-wave marker in each eligible cine. No-R-wave clips retain the exact previous HR-estimated PanEcho embedding.",
             "", "Paired comparisons use the same patient-held-out folds and visits/transitions as the previous estimated-heartbeat trial. Negative MAE and positive AUROC/AP differences favor R-wave sampling.",
             "", "## Current-visit physiology", "",
             "| Target | Subgroup | Visits | MAE: old → R-wave | Pearson r: old → R-wave |",
             "|---|---|---:|---:|---:|"]
    for row in gls_rows:
        m, r = row["mae"], row["pearson_r"]
        lines.append(f"| {row['target']} | {row['subgroup']} | {row['visits']} | "
                     f"{m['reference']:.3f} → {m['rwave']:.3f} (Δ {m['difference']:+.3f}; 95% CI {m['ci95'][0]:+.3f} to {m['ci95'][1]:+.3f}) | "
                     f"{r['reference']:.3f} → {r['rwave']:.3f} |")
    lines += ["", "## Deterioration at next visit", "",
              "| Model | Subgroup | Transitions/events | AUROC: old → R-wave | AP: old → R-wave |",
              "|---|---|---:|---:|---:|"]
    for row in future_rows:
        a, p = row["auroc"], row["ap"]
        lines.append(f"| {row['model']} | {row['subgroup']} | {row['transitions']}/{row['events']} | "
                     f"{a['reference']:.3f} → {a['rwave']:.3f} (Δ {a['difference']:+.3f}; 95% CI {a['ci95'][0]:+.3f} to {a['ci95'][1]:+.3f}) | "
                     f"{p['reference']:.3f} → {p['rwave']:.3f} (Δ {p['difference']:+.3f}; 95% CI {p['ci95'][0]:+.3f} to {p['ci95'][1]:+.3f}) |")
    lines += ["", "The R-wave arm changes both temporal anchoring and number of windows (one instead of up to four). Fourteen cines have more than two R-wave markers; first-to-last spans two or three cycles. Exploratory internal comparison, not independent validation."]
    (ROOT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparison saved to {ROOT / 'results.md'}")


if __name__ == "__main__":
    main()
