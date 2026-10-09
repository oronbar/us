"""Patient-held-out GLS and deterioration evaluation of adapted PanEcho."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from threadpoolctl import threadpool_limits

import ichilov_ge_ssl_evaluate as prior_gls
from ichilov_video_experiment_eval import (ALIGNED, NONPHIL, fit_classifier,
                                           make_future_arrays, ppv_top_fifth)
from ichilov3_compare_rwave_panecho import bootstrap_delta, corr


ROOT = Path(r"D:\DS\ichilov3_panecho_rwave_ssl_20261005")
SOURCE = Path(r"D:\DS\ichilov3_rwave_first_cycle_panecho_20261005")
LABELS = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet")


def visit_table(path):
    with np.load(path) as f:
        if list(f["views"]) != ["A2C", "A3C", "A4C"]:
            raise ValueError("View order changed")
        return {(str(p), str(d)): v.copy() for p, d, v in zip(
            f["patient_ids"], f["visit_dates"], f["embeddings"])}


def future():
    out = ROOT / "evaluation" / "future"
    (out / "folds").mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(NONPHIL / "cohort.parquet")
    visits = pd.read_parquet(ALIGNED / "visit_alignment.parquet")
    labels = pd.read_parquet(LABELS)
    folds = pd.read_parquet(NONPHIL / "patient_folds.parquet")
    frozen = visit_table(SOURCE / "panecho_visits.npz")
    groups = cohort.patient_id.to_numpy(str)
    y = cohort.label.to_numpy(int)
    with threadpool_limits(limits=4):
        for (repeat, fold), assignment in folds.groupby(["repeat", "fold"]):
            repeat, fold = int(repeat), int(fold)
            held = np.flatnonzero(np.isin(groups, assignment.patient_id))
            train = np.flatnonzero(~np.isin(groups, assignment.patient_id))
            if set(groups[held]) & set(groups[train]):
                raise ValueError("Patient leakage")
            adapted = visit_table(ROOT / "evaluation_embeddings" /
                                  f"r{repeat}_f{fold}" / "visit_embeddings.npz")
            arrays = make_future_arrays({"adapted": adapted, "frozen": frozen},
                                        visits, labels, cohort)
            for model in ("adapted", "frozen"):
                for mode in ("current", "history"):
                    dest = out / "folds" / f"{model}_{mode}_r{repeat}_f{fold}.parquet"
                    if dest.exists():
                        continue
                    score, tuning = fit_classifier(arrays[(model, mode)], y, groups,
                                                   train, held, 20260930+repeat*5+fold)
                    result = cohort.iloc[held][["transition_id", "patient_id", "label"]].copy()
                    result = result.assign(model=model, mode=mode, repeat=repeat,
                                           fold=fold, score=score)
                    temp = dest.with_suffix(".tmp")
                    result.to_parquet(temp, index=False)
                    temp.replace(dest)
                    dest.with_suffix(".json").write_text(json.dumps(tuning), encoding="utf-8")
                    print(f"Future {model}/{mode} r{repeat} f{fold}", flush=True)
    files = sorted((out / "folds").glob("*.parquet"))
    if len(files) != 60:
        raise ValueError(f"Expected 60 future fold predictions, found {len(files)}")
    raw = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    if raw.groupby(["model", "mode", "transition_id"]).size().ne(3).any():
        raise ValueError("Incomplete repeated predictions")
    raw.to_parquet(out / "fold_predictions.parquet", index=False)
    pred = raw.groupby(["model", "mode", "transition_id", "patient_id", "label"],
                       as_index=False).score.mean()
    baseline = pd.read_parquet(NONPHIL / "retained_baseline_oof.parquet")[
        ["transition_id", "patient_id", "label", "score"]].rename(
            columns={"score": "baseline_score"})
    combined = pred.merge(baseline, on=["transition_id", "patient_id", "label"],
                          validate="many_to_one")
    combined["late25"] = .25*combined.score + .75*combined.baseline_score
    combined.to_parquet(out / "oof_predictions.parquet", index=False)
    metrics = []
    for (model, mode), group in combined.groupby(["model", "mode"]):
        for arm in ("score", "late25"):
            metrics.append({"model": model, "mode": mode, "arm": arm,
                            "n": len(group), "events": int(group.label.sum()),
                            "auroc": roc_auc_score(group.label, group[arm]),
                            "ap": average_precision_score(group.label, group[arm]),
                            "ppv_top20": ppv_top_fifth(group.label.to_numpy(int),
                                                        group[arm].to_numpy(float))})
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    paired = {}
    for mode in ("current", "history"):
        for arm in ("score", "late25"):
            wide = combined[combined["mode"].eq(mode)].pivot(
                index=["transition_id", "patient_id", "label"],
                columns="model", values=arm).reset_index()
            paired[f"{mode}/{arm}"] = {
                "auroc": bootstrap_delta(wide, "label", "frozen", "adapted", roc_auc_score),
                "ap": bootstrap_delta(wide, "label", "frozen", "adapted",
                                      average_precision_score)}
    return metrics, paired


def main():
    if not all((ROOT / "evaluation_embeddings" / f"r{r}_f{f}" /
                "run_complete.json").exists() for r in range(3) for f in range(5)):
        raise RuntimeError("All 15 fold embeddings required")
    prior_gls.ROOT = ROOT
    prior_gls.visit_table = lambda config, repeat, fold: visit_table(
        ROOT / "evaluation_embeddings" / f"r{repeat}_f{fold}" /
        "visit_embeddings.npz")
    gls_out = ROOT / "evaluation" / "gls"
    gls_out.mkdir(parents=True, exist_ok=True)
    visits = pd.read_parquet(ALIGNED / "visit_alignment.parquet")
    labels = pd.read_parquet(LABELS)
    # Reuse the established nested patient-fold Ridge evaluation verbatim.
    adapted_gls = prior_gls.gls("all", gls_out, visits, labels)
    frozen_gls = pd.read_parquet(SOURCE / "ge_only" / "gls" / "oof_predictions.parquet")
    frozen_gls = frozen_gls[frozen_gls.model.eq("panecho_rwave")][
        ["visit_id", "patient_id", "value", "prediction"]]
    new_gls = pd.read_parquet(gls_out / "gls" / "oof_predictions.parquet")
    wide = frozen_gls.merge(new_gls, on=["visit_id", "patient_id", "value"],
                            suffixes=("_frozen", "_adapted"), validate="one_to_one")
    if len(wide) != 298:
        raise ValueError(f"Expected 298 matched GE GLS visits, got {len(wide)}")
    gls_paired = {
        "mae": bootstrap_delta(wide, "value", "prediction_frozen",
                               "prediction_adapted", mean_absolute_error),
        "pearson_r": bootstrap_delta(wide, "value", "prediction_frozen",
                                     "prediction_adapted", corr)}
    future_metrics, future_paired = future()
    result = {"protocol": "3x5 patient-held-out folds, three SSL epochs per fold",
              "same_visit_mid_gls": {"adapted": adapted_gls, "paired": gls_paired},
              "next_visit": {"metrics": future_metrics, "paired": future_paired},
              "frozen_reference": str(SOURCE / "panecho_visits.npz"),
              "clinical_strain_baseline": "AUROC 0.7255, AP 0.3792 on 182 GE transitions"}
    out = ROOT / "evaluation"
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "run_complete.json").write_text(json.dumps({"results": str(out / "results.json")}),
                                           encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
