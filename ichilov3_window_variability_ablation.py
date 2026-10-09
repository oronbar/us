"""Paired, patient-held-out ablation of per-cine mean versus mean+SD embeddings.

Reads the saved 16-frame window embeddings; never touches source DICOMs or crops.
The mean-only arm is the unchanged, previously completed EchoPrime evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from ichilov3_train_fusion import atomic, hashfile
from ichilov_video_experiment_eval import (
    ALIGNED, NONPHIL, RIDGE_ALPHAS, REPRESENTATIONS, ROOT as PREVIOUS,
    fit_classifier, make_future_arrays, ppv_top_fifth, transform_video,
    vendor_visits,
)

ROOT = Path(r"D:\DS\ichilov3_window_variability_20261004")
MANIFEST = Path(r"D:\DS\ichilov3_temporal_trial_20260927\echoprime_manifest.json")
MEAN_VISITS = Path(r"D:\DS\ichilov3_temporal_trial_20260927\echoprime_visits.npz")
LABELS = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet")
VIEWS = ("A2C", "A3C", "A4C")
SEED = 20261004


def prepare() -> dict[tuple[str, str], np.ndarray]:
    ROOT.mkdir(parents=True, exist_ok=True)
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if len(records) != 1233:
        raise ValueError(f"Expected 1233 selected clips, got {len(records)}")
    protocol = {
        "question": "Does between-window variation add signal beyond mean pooling?",
        "model": "frozen EchoPrime, same saved 16-frame windows as prior heartbeat trial",
        "arms": ["prior_mean_only", "mean_plus_population_SD"],
        "future": "current and current-minus-prior video; same 182 transitions, patient folds, inner CV and 25% late fusion as stage2",
        "gls": "same-visit GE Mid-GLS; same patients, folds, Ridge selection as stage1",
        "primary": "paired change in next-visit AUROC and AP for history and fixed 25% late fusion",
        "selection": "No outer-fold target used for representation or regularization selection",
        "manifest_sha256": hashfile(MANIFEST),
        "mean_visits_sha256": hashfile(MEAN_VISITS),
        "prior_protocol_sha256": hashfile(PREVIOUS / "stage2" / "protocol.json"),
    }
    path = ROOT / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Changed source or protocol; use a new output directory")
    atomic(path, protocol)

    with np.load(MEAN_VISITS) as source:
        keys = list(zip(source["patient_ids"].astype(str), source["visit_dates"].astype(str)))
        old = source["embeddings"].astype(np.float32)
        if list(source["views"]) != list(VIEWS):
            raise ValueError("Unexpected view order")
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate visit")
    index = {key: i for i, key in enumerate(keys)}
    values = np.full((len(keys), 3, old.shape[-1] * 2), np.nan, np.float32)
    seen: set[tuple[str, str, str]] = set()
    counts: dict[int, int] = {}
    max_mean_error = 0.0
    for record in records:
        key = (str(record["patient"]), str(record["visit_date"]))
        view = str(record["view"])
        if key not in index or view not in VIEWS or (*key, view) in seen:
            raise ValueError(f"Unexpected or duplicate clip {key} {view}")
        seen.add((*key, view))
        with np.load(record["embedding_path"]) as source:
            windows = source["window_embeddings"].astype(np.float32)
            stored_mean = source["view_embedding"].astype(np.float32)
        if windows.ndim != 2 or windows.shape[1] != old.shape[-1]:
            raise ValueError(f"Bad window embedding shape: {record['file_id']}")
        mean = windows.mean(axis=0)
        sd = windows.std(axis=0, ddof=0)
        error = max(float(np.max(np.abs(mean - stored_mean))),
                    float(np.max(np.abs(mean - old[index[key], VIEWS.index(view)]))))
        max_mean_error = max(max_mean_error, error)
        if error > 2e-5:
            raise ValueError(f"Prior mean mismatch {record['file_id']}: {error}")
        values[index[key], VIEWS.index(view)] = np.concatenate([mean, sd])
        counts[len(windows)] = counts.get(len(windows), 0) + 1
    if len(seen) != len(keys) * 3 or not np.isfinite(values).all():
        raise ValueError("Incomplete or nonfinite visit representation")
    destination = ROOT / "echoprime_mean_sd_visits.npz"
    if not destination.exists():
        temporary = ROOT / "echoprime_mean_sd_visits.partial.npz"
        np.savez_compressed(temporary, patient_ids=np.array([key[0] for key in keys]),
                            visit_dates=np.array([key[1] for key in keys]),
                            views=np.array(VIEWS), embeddings=values)
        temporary.replace(destination)
    else:
        with np.load(destination) as saved:
            if not np.array_equal(saved["embeddings"], values):
                raise ValueError("Saved representations differ from sources")
    audit = {"visits": len(keys), "clips": len(seen), "windows_per_clip": counts,
             "maximum_reconstructed_mean_error": max_mean_error,
             "feature_shape": list(values.shape)}
    atomic(ROOT / "preparation_audit.json", audit)
    print(f"Prepared {len(keys)} visits, {len(seen)} clips; windows {counts}; "
          f"max mean error {max_mean_error:.2g}", flush=True)
    return dict(zip(keys, values))


def score_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model"):
        y = group.label.to_numpy(int)
        score = group.score.to_numpy(float)
        rows.append({"model": model, "n": len(group), "patients": group.patient_id.nunique(),
                     "events": int(y.sum()), "auc": roc_auc_score(y, score),
                     "ap": average_precision_score(y, score),
                     "ppv_top20": ppv_top_fifth(y, score)})
    return pd.DataFrame(rows).sort_values("model")


def paired_ci(predictions: pd.DataFrame, metrics: pd.DataFrame,
              pairs: list[tuple[str, str]], seed: int) -> pd.DataFrame:
    wide = predictions.pivot(index=["transition_id", "patient_id", "label"],
                             columns="model", values="score")
    if wide.isna().any().any():
        raise ValueError("Unequal compared transitions")
    y = wide.index.get_level_values("label").to_numpy(int)
    groups = wide.index.get_level_values("patient_id").to_numpy(str)
    patients = np.unique(groups)
    by_patient = {p: np.flatnonzero(groups == p) for p in patients}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(1000):
        take = np.concatenate([by_patient[p] for p in rng.choice(patients, len(patients), True)])
        if np.unique(y[take]).size < 2:
            continue
        draws.append(take)
    table = metrics.set_index("model")
    rows = []
    for candidate, reference in pairs:
        row = {"candidate": candidate, "reference": reference}
        for metric, fn in (("auc", roc_auc_score), ("ap", average_precision_score),
                           ("ppv_top20", ppv_top_fifth)):
            diffs = np.array([fn(y[idx], wide[candidate].to_numpy()[idx]) -
                              fn(y[idx], wide[reference].to_numpy()[idx]) for idx in draws])
            row[f"delta_{metric}"] = float(table.loc[candidate, metric] - table.loc[reference, metric])
            row[f"delta_{metric}_lo"], row[f"delta_{metric}_hi"] = np.quantile(diffs, [.025, .975])
        rows.append(row)
    return pd.DataFrame(rows)


def future(models: dict[tuple[str, str], np.ndarray], max_folds: int | None) -> bool:
    out = ROOT / "future"
    fold_dir = out / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(NONPHIL / "cohort.parquet")
    visits = pd.read_parquet(ALIGNED / "visit_alignment.parquet")
    labels = pd.read_parquet(LABELS)
    arrays = make_future_arrays({"mean_sd": models}, visits, labels, cohort)
    folds = pd.read_parquet(NONPHIL / "patient_folds.parquet")
    groups = cohort.patient_id.to_numpy(str)
    y = cohort.label.to_numpy(int)
    if len(cohort) != 182 or int(y.sum()) != 38:
        raise ValueError("Future evaluation cohort changed")
    done = 0
    with threadpool_limits(limits=4):
        for (repeat, fold), assignment in folds.groupby(["repeat", "fold"]):
            held = np.flatnonzero(np.isin(groups, assignment.patient_id))
            train = np.flatnonzero(~np.isin(groups, assignment.patient_id))
            if len(held) == 0 or set(groups[held]) & set(groups[train]):
                raise ValueError("Bad outer patient fold")
            seed = 20260930 + 5 * int(repeat) + int(fold)
            for mode in ("current", "history"):
                destination = fold_dir / f"echoprime_mean_sd_{mode}_r{repeat}_f{fold}.parquet"
                if not destination.exists():
                    score, selection = fit_classifier(arrays[("mean_sd", mode)], y, groups,
                                                      train, held, seed)
                    prediction = cohort.iloc[held][["transition_id", "patient_id", "label"]].copy()
                    prediction = prediction.assign(model=f"echoprime_mean_sd_{mode}",
                                                   repeat=int(repeat), fold=int(fold), score=score)
                    temporary = destination.with_suffix(".tmp")
                    prediction.to_parquet(temporary, index=False)
                    temporary.replace(destination)
                    atomic(destination.with_suffix(".json"), selection)
                done += 1
                atomic(out / "status.json", {"status": "running", "completed": done,
                                             "total": 30, "pid": os.getpid()})
                if max_folds and done >= max_folds:
                    return False
                print(f"Future {done}/30: repeat {repeat} fold {fold} {mode}", flush=True)
    files = sorted(fold_dir.glob("echoprime_mean_sd_*.parquet"))
    if len(files) != 30:
        raise ValueError("Incomplete future folds")
    raw = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    if raw.groupby(["model", "transition_id"]).size().ne(3).any():
        raise ValueError("Missing repeated prediction")
    raw.to_parquet(out / "fold_predictions.parquet", index=False)
    pred = raw.groupby(["model", "transition_id", "patient_id", "label"],
                       as_index=False).score.mean()
    old = pd.read_parquet(PREVIOUS / "stage2" / "future" / "oof_predictions.parquet")
    old = old[old.model.isin(("echoprime_current", "echoprime_history",
                             "echoprime_current_late25", "echoprime_history_late25",
                             "strain_clinical_reference"))].copy()
    baseline = old[old.model.eq("strain_clinical_reference")]
    for name in pred.model.unique():
        paired = pred[pred.model.eq(name)].merge(baseline, on=["transition_id", "patient_id", "label"],
                                                suffixes=("_video", "_baseline"), validate="one_to_one")
        fused = paired[["transition_id", "patient_id", "label"]].copy()
        fused["model"] = name + "_late25"
        fused["score"] = .25 * paired.score_video + .75 * paired.score_baseline
        old = pd.concat([old, fused], ignore_index=True)
    all_predictions = pd.concat([old, pred], ignore_index=True)
    if all_predictions.groupby("model").size().nunique() != 1:
        raise ValueError("Model cohorts differ")
    all_predictions.to_parquet(out / "oof_predictions.parquet", index=False)
    metrics = score_table(all_predictions)
    metrics.to_csv(out / "metrics.csv", index=False)
    pairs = [(f"echoprime_mean_sd_{mode}{suffix}", f"echoprime_{mode}{suffix}")
             for mode in ("current", "history") for suffix in ("", "_late25")]
    pairs += [("echoprime_mean_sd_history_late25", "strain_clinical_reference")]
    delta = paired_ci(all_predictions, metrics, pairs, SEED)
    delta.to_csv(out / "paired_deltas.csv", index=False)
    atomic(out / "status.json", {"status": "complete", "completed": 30, "total": 30})
    return True


def gls(models: dict[tuple[str, str], np.ndarray], max_folds: int | None) -> bool:
    out = ROOT / "gls"
    fold_dir = out / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    visits = pd.read_parquet(ALIGNED / "visit_alignment.parquet")
    labels = pd.read_parquet(LABELS)
    valid = vendor_visits()
    matched = visits[visits.match_status.eq("exact_patient_date_study_uid")].copy()
    matched = matched[[(p, d) in valid for p, d in zip(matched.patient_id, matched.visit_date)]]
    cohort = matched[["visit_id", "patient_id", "visit_date"]].merge(
        labels[["visit_id", "gls_mid_magnitude"]], on="visit_id", validate="one_to_one")
    cohort = cohort[np.isfinite(cohort.gls_mid_magnitude)].sort_values(["patient_id", "visit_date"])
    x = np.stack([models[(str(p), str(d))] for p, d in zip(cohort.patient_id, cohort.visit_date)])
    y = cohort.gls_mid_magnitude.to_numpy(float)
    patients = cohort.patient_id.to_numpy(str)
    folds = pd.read_parquet(ALIGNED / "patient_folds.parquet")
    if len(cohort) != 298 or cohort.patient_id.nunique() != 89:
        raise ValueError("GLS evaluation cohort changed")
    done = 0
    with threadpool_limits(limits=4):
        for (repeat, fold), assignment in folds.groupby(["repeat", "fold"]):
            held = np.flatnonzero(np.isin(patients, assignment.patient_id))
            train = np.flatnonzero(~np.isin(patients, assignment.patient_id))
            if len(held) == 0 or set(patients[held]) & set(patients[train]):
                raise ValueError("Bad GLS outer patient fold")
            destination = fold_dir / f"echoprime_mean_sd_r{repeat}_f{fold}.parquet"
            if not destination.exists():
                seed = 20260930 + int(repeat) * 5 + int(fold)
                inner = list(GroupKFold(n_splits=3, shuffle=True, random_state=seed)
                             .split(np.zeros(len(train)), y[train], patients[train]))
                candidates = []
                for representation in REPRESENTATIONS:
                    transformed = [(transform_video(x, train[a], train[b], representation, seed),
                                    train[a], train[b]) for a, b in inner]
                    for alpha in RIDGE_ALPHAS:
                        errors = []
                        for (u, v), fit, val in transformed:
                            regressor = Ridge(alpha=alpha, solver="lsqr", tol=1e-5).fit(u, y[fit])
                            errors.append(mean_absolute_error(y[val], regressor.predict(v)))
                        candidates.append((float(np.mean(errors)), representation, alpha))
                _, representation, alpha = min(candidates)
                u, v = transform_video(x, train, held, representation, seed)
                prediction = Ridge(alpha=alpha, solver="lsqr", tol=1e-5).fit(u, y[train]).predict(v)
                result = cohort.iloc[held][["visit_id", "patient_id", "visit_date"]].copy()
                result = result.assign(model="echoprime_mean_sd", repeat=int(repeat), fold=int(fold),
                                       value=y[held], prediction=prediction)
                temporary = destination.with_suffix(".tmp")
                result.to_parquet(temporary, index=False)
                temporary.replace(destination)
                atomic(destination.with_suffix(".json"),
                       {"representation": representation, "alpha": alpha})
            done += 1
            atomic(out / "status.json", {"status": "running", "completed": done,
                                         "total": 15, "pid": os.getpid()})
            if max_folds and done >= max_folds:
                return False
            print(f"GLS {done}/15: repeat {repeat} fold {fold}", flush=True)
    files = sorted(fold_dir.glob("echoprime_mean_sd_*.parquet"))
    if len(files) != 15:
        raise ValueError("Incomplete GLS folds")
    raw = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    if raw.groupby("visit_id").size().ne(3).any():
        raise ValueError("Missing repeated GLS prediction")
    raw.to_parquet(out / "fold_predictions.parquet", index=False)
    pred = raw.groupby(["visit_id", "patient_id", "value"], as_index=False).prediction.mean()
    old = pd.read_parquet(PREVIOUS / "stage1" / "gls" / "oof_predictions.parquet")
    old = old[old.model.eq("echoprime")][["visit_id", "patient_id", "value", "prediction"]]
    paired = pred.merge(old, on=["visit_id", "patient_id", "value"],
                        suffixes=("_mean_sd", "_mean"), validate="one_to_one")
    if len(paired) != len(pred):
        raise ValueError("Unequal GLS cohorts")
    paired.to_parquet(out / "paired_oof_predictions.parquet", index=False)
    result = {}
    for name, col in (("mean", "prediction_mean"), ("mean_sd", "prediction_mean_sd")):
        result[name] = {"visits": len(paired), "patients": paired.patient_id.nunique(),
                        "mae": float(mean_absolute_error(paired.value, paired[col])),
                        "pearson_r": float(pearsonr(paired.value, paired[col]).statistic)}
    atomic(out / "metrics.json", result)
    patient_ids = paired.patient_id.unique()
    by_patient = {p: paired[paired.patient_id.eq(p)] for p in patient_ids}
    rng = np.random.default_rng(SEED)
    ci = {"delta_pearson_r": [], "delta_mae": []}
    for _ in range(1000):
        sample = pd.concat((by_patient[p] for p in rng.choice(patient_ids, len(patient_ids), True)),
                           ignore_index=True)
        if sample.value.nunique() < 2:
            continue
        ci["delta_pearson_r"].append(pearsonr(sample.value, sample.prediction_mean_sd).statistic -
                                     pearsonr(sample.value, sample.prediction_mean).statistic)
        ci["delta_mae"].append(mean_absolute_error(sample.value, sample.prediction_mean_sd) -
                               mean_absolute_error(sample.value, sample.prediction_mean))
    atomic(out / "paired_deltas.json", {
        metric: {"delta": result["mean_sd"][metric[6:]] - result["mean"][metric[6:]],
                 "95pct_patient_bootstrap": np.quantile(samples, [.025, .975]).tolist()}
        for metric, samples in ci.items()})
    atomic(out / "status.json", {"status": "complete", "completed": 15, "total": 15})
    return True


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("prepare", "future", "gls", "all"), default="all")
    parser.add_argument("--max-folds", type=int)
    args = parser.parse_args()
    models = prepare()
    if args.phase in ("future", "all") and not future(models, args.max_folds):
        return
    if args.phase in ("gls", "all") and not gls(models, args.max_folds):
        return
    if args.phase == "all":
        future_metrics = pd.read_csv(ROOT / "future" / "metrics.csv")
        deltas = pd.read_csv(ROOT / "future" / "paired_deltas.csv")
        gls_metrics = json.loads((ROOT / "gls" / "metrics.json").read_text())
        lines = ["# EchoPrime window-variation ablation", "",
                 "Prespecified comparison: unchanged mean-only OOF predictions versus mean+SD",
                 "from the same saved 16-frame windows and identical patient-held-out folds.", "",
                 "## Next-visit deterioration", "", future_metrics.to_string(index=False), "",
                 "## Paired differences (mean+SD minus mean-only)", "",
                 deltas.to_string(index=False), "", "## Same-visit Mid-GLS", "",
                 json.dumps(gls_metrics, indent=2), "",
                 "Exploratory reuse of the development cohort; confirm any gain on unused patients."]
        (ROOT / "results.md").write_text("\n".join(lines), encoding="utf-8")
        atomic(ROOT / "run_complete.json", {"future_folds": 30, "gls_folds": 15,
                                             "results": str(ROOT / "results.md")})


if __name__ == "__main__":
    main()
