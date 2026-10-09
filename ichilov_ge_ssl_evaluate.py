"""Held-out next-visit and GLS probes for fold-isolated GE EchoPrime adapters."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from ichilov_video_experiment_eval import (ALIGNED, NONPHIL, REPRESENTATIONS,
                                           RIDGE_ALPHAS, fit_classifier,
                                           make_future_arrays, ppv_top_fifth,
                                           transform_video, vendor_visits)

ROOT = Path(r"D:\DS\ichilov_ge_ssl_20260930")
LABELS = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet")
REFERENCE = Path(r"D:\us\output\ichilov_video_experiments_20260930\stage2\future")


def visit_table(config: str, repeat: int, fold: int) -> dict:
    path = ROOT / "evaluation_embeddings" / config / f"r{repeat}_f{fold}" / "visit_embeddings.npz"
    with np.load(path) as data:
        if list(data["views"]) != ["A2C", "A3C", "A4C"]:
            raise ValueError("Changed view order")
        return dict(zip(zip(data["patient_ids"].astype(str), data["visit_dates"].astype(str)),
                        data["embeddings"].astype(np.float32)))


def future(config: str, out: Path, cohort, visits, labels):
    groups = cohort.patient_id.to_numpy(str)
    y = cohort.label.to_numpy(int)
    folds = pd.read_parquet(NONPHIL / "patient_folds.parquet")
    (out / "folds").mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        for (repeat, fold), assignment in folds.groupby(["repeat", "fold"]):
            repeat, fold = int(repeat), int(fold)
            held = np.flatnonzero(np.isin(groups, assignment.patient_id))
            train = np.flatnonzero(~np.isin(groups, assignment.patient_id))
            models = {"adapter": visit_table(config, repeat, fold)}
            arrays = make_future_arrays(models, visits, labels, cohort)
            for mode in ("current", "history"):
                dest = out / "folds" / f"{mode}_r{repeat}_f{fold}.parquet"
                if dest.exists():
                    continue
                score, selection = fit_classifier(arrays[("adapter", mode)], y, groups,
                                                  train, held, 20260930 + repeat*5 + fold)
                result = cohort.iloc[held][["transition_id", "patient_id", "label"]].copy()
                result = result.assign(config=config, mode=mode, repeat=repeat,
                                       fold=fold, score=score)
                temp = dest.with_suffix(".tmp")
                result.to_parquet(temp, index=False)
                temp.replace(dest)
                dest.with_suffix(".json").write_text(json.dumps(selection), encoding="utf-8")
                print(f"Future adapter {config} {mode} r{repeat} f{fold}", flush=True)
    files = sorted((out / "folds").glob("*.parquet"))
    if len(files) != 30:
        raise ValueError(f"Incomplete adapter future evaluation: {len(files)} folds")
    raw = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    raw.to_parquet(out / "fold_predictions.parquet", index=False)
    if raw.groupby(["mode", "transition_id"]).size().ne(3).any():
        raise ValueError("Missing repeat for an adapter prediction")
    pred = raw.groupby(["mode", "transition_id", "patient_id", "label"],
                       as_index=False).score.mean()
    pred.to_parquet(out / "oof_predictions.parquet", index=False)
    reference = pd.read_parquet(NONPHIL / "retained_baseline_oof.parquet")
    baseline = reference[["transition_id", "patient_id", "label", "score"]].rename(
        columns={"score": "reference_score"})
    frames = []
    for mode, group in pred.groupby("mode"):
        joined = group.merge(baseline, on=["transition_id", "patient_id", "label"],
                             validate="one_to_one")
        joined["late25"] = .25*joined.score + .75*joined.reference_score
        frames.append(joined)
    combined = pd.concat(frames, ignore_index=True)
    rows = []
    for mode, group in combined.groupby("mode"):
        for arm, column in (("video", "score"), ("late25", "late25")):
            truth = group.label.to_numpy(int)
            score = group[column].to_numpy(float)
            rows.append({"config": config, "mode": mode, "arm": arm,
                         "n": len(truth), "events": int(truth.sum()),
                         "auc": roc_auc_score(truth, score),
                         "ap": average_precision_score(truth, score),
                         "ppv_top20": ppv_top_fifth(truth, score)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    combined.to_parquet(out / "paired_predictions.parquet", index=False)
    original = pd.read_parquet(REFERENCE / "oof_predictions.parquet")
    bootstrap_rows = []
    rng = np.random.default_rng(20260930)
    for mode, group in combined.groupby("mode"):
        group = group.sort_values("transition_id").reset_index(drop=True)
        key = ["transition_id", "patient_id", "label"]
        clinical = group.reference_score.to_numpy(float)
        frozen_name = f"echoprime_{mode}_late25"
        frozen = original[original.model.eq(frozen_name)][key + ["score"]]
        group = group.merge(frozen.rename(columns={"score": "frozen_score"}),
                            on=key, validate="one_to_one")
        if len(group) != len(cohort):
            raise ValueError("Adapter/frozen OOF cohort mismatch")
        truth = group.label.to_numpy(int)
        patient = group.patient_id.to_numpy(str)
        adapted = group.late25.to_numpy(float)
        frozen = group.frozen_score.to_numpy(float)
        clinical = group.reference_score.to_numpy(float)
        people = np.unique(patient)
        by_patient = {p: np.flatnonzero(patient == p) for p in people}
        for reference_name, reference_score in (("strain_clinical", clinical),
                                                ("frozen_echoprime", frozen)):
            deltas = {metric: [] for metric in ("auc", "ap", "ppv_top20")}
            for _ in range(1000):
                take = np.concatenate([by_patient[p] for p in
                                       rng.choice(people, len(people), replace=True)])
                yy = truth[take]
                if len(np.unique(yy)) < 2:
                    continue
                x, z = adapted[take], reference_score[take]
                deltas["auc"].append(roc_auc_score(yy, x) - roc_auc_score(yy, z))
                deltas["ap"].append(average_precision_score(yy, x) -
                                    average_precision_score(yy, z))
                deltas["ppv_top20"].append(ppv_top_fifth(yy, x) - ppv_top_fifth(yy, z))
            row = {"config": config, "mode": mode, "arm": "late25",
                   "reference": reference_name}
            point = {
                "auc": roc_auc_score(truth, adapted) - roc_auc_score(truth, reference_score),
                "ap": average_precision_score(truth, adapted) -
                      average_precision_score(truth, reference_score),
                "ppv_top20": ppv_top_fifth(truth, adapted) -
                             ppv_top_fifth(truth, reference_score),
            }
            for metric, values in deltas.items():
                low, high = np.quantile(values, [.025, .975])
                row[f"delta_{metric}"] = float(point[metric])
                row[f"delta_{metric}_lo"] = float(low)
                row[f"delta_{metric}_hi"] = float(high)
            bootstrap_rows.append(row)
    pd.DataFrame(bootstrap_rows).to_csv(out / "paired_deltas.csv", index=False)
    return metrics


def gls(config: str, out: Path, visits, labels):
    folder = out / "gls"
    folder.mkdir(exist_ok=True)
    valid = vendor_visits()
    matched = visits[visits.match_status.eq("exact_patient_date_study_uid")].copy()
    matched = matched[[(p, d) in valid for p, d in zip(matched.patient_id, matched.visit_date)]]
    cohort = matched[["visit_id", "patient_id", "visit_date"]].merge(
        labels[["visit_id", "gls_mid_magnitude"]], on="visit_id", validate="one_to_one")
    cohort = cohort[np.isfinite(cohort.gls_mid_magnitude)].sort_values(["patient_id", "visit_date"])
    patients = cohort.patient_id.to_numpy(str)
    y = cohort.gls_mid_magnitude.to_numpy(float)
    folds = pd.read_parquet(ALIGNED / "patient_folds.parquet")
    with threadpool_limits(limits=4):
        for (repeat, fold), assignment in folds.groupby(["repeat", "fold"]):
            repeat, fold = int(repeat), int(fold)
            dest = folder / f"r{repeat}_f{fold}.parquet"
            if dest.exists():
                continue
            held = np.flatnonzero(np.isin(patients, assignment.patient_id))
            train = np.flatnonzero(~np.isin(patients, assignment.patient_id))
            table = visit_table(config, repeat, fold)
            x = np.stack([table[(p, d)] for p, d in zip(cohort.patient_id,
                                                       cohort.visit_date)])
            seed = 20260930 + repeat*5 + fold
            inner = list(GroupKFold(n_splits=3, shuffle=True, random_state=seed)
                         .split(np.zeros(len(train)), y[train], patients[train]))
            candidates = []
            for representation in REPRESENTATIONS:
                for alpha in RIDGE_ALPHAS:
                    errors = []
                    for a, b in inner:
                        fit, val = train[a], train[b]
                        u, v = transform_video(x, fit, val, representation, seed)
                        reg = Ridge(alpha=alpha, solver="lsqr", tol=1e-5).fit(u, y[fit])
                        errors.append(mean_absolute_error(y[val], reg.predict(v)))
                    candidates.append((float(np.mean(errors)), representation, alpha))
            _, representation, alpha = min(candidates)
            u, v = transform_video(x, train, held, representation, seed)
            prediction = Ridge(alpha=alpha, solver="lsqr", tol=1e-5).fit(u, y[train]).predict(v)
            result = cohort.iloc[held][["visit_id", "patient_id", "visit_date"]].copy()
            result = result.assign(repeat=repeat, fold=fold, value=y[held], prediction=prediction)
            temp = dest.with_suffix(".tmp")
            result.to_parquet(temp, index=False)
            temp.replace(dest)
            print(f"GLS adapter {config} r{repeat} f{fold}", flush=True)
    files = sorted(folder.glob("r*_f*.parquet"))
    if len(files) != 15:
        raise ValueError(f"Incomplete adapter GLS folds: {len(files)}")
    raw = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    pred = raw.groupby(["visit_id", "patient_id", "value"], as_index=False).prediction.mean()
    if len(pred) != len(cohort) or raw.groupby("visit_id").size().ne(3).any():
        raise ValueError("Incomplete GLS predictions")
    pred.to_parquet(folder / "oof_predictions.parquet", index=False)
    result = {"config": config, "visits": len(pred), "patients": pred.patient_id.nunique(),
              "mae": mean_absolute_error(pred.value, pred.prediction),
              "pearson_r": pearsonr(pred.value, pred.prediction).statistic}
    pd.DataFrame([result]).to_csv(folder / "metrics.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", choices=("all", "specific"), required=True)
    args = parser.parse_args()
    out = ROOT / "evaluation" / args.config
    out.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(NONPHIL / "cohort.parquet")
    visits = pd.read_parquet(ALIGNED / "visit_alignment.parquet")
    labels = pd.read_parquet(LABELS)
    f = future(args.config, out, cohort, visits, labels)
    g = gls(args.config, out, visits, labels)
    report = f"# {args.config} GE SSL adapter\n\nFuture deterioration:\n\n{f.to_string(index=False)}\n\nGLS:\n\n{g}\n"
    (out / "results.md").write_text(report, encoding="utf-8")
    (out / "run_complete.json").write_text(json.dumps({"config": args.config,
        "future_metrics": str(out / "metrics.csv"), "gls_metrics": str(out / "gls" / "metrics.csv")},
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
