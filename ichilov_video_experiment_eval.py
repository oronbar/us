"""Patient-held-out comparison of frozen video encoders and video history."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ichilov3_train_fusion import atomic, hashfile

ROOT = Path(r"D:\us\output\ichilov_video_experiments_20260930")
ALIGNED = Path(r"D:\DS\ichilov3_aligned_20260927")
NONPHIL = Path(r"D:\DS\ichilov3_nonphilips_future_20260930\aligned")
CROPS = Path(r"D:\DS\ichilov3_stage2_padded_20260926\full_manifest.json")
PATHS = {
    "echoprime": Path(r"D:\DS\ichilov3_temporal_trial_20260927\echoprime_visits.npz"),
    "panecho": Path(r"D:\DS\ichilov3_temporal_trial_20260927\panecho_visits.npz"),
    "echofm": Path(r"D:\DS\ichilov3_echofm_20260930\echofm_visits.npz"),
}
REPRESENTATIONS = ("full", "pca8")
REGULARIZATION = (1e-5, 1e-4, 1e-3, .01, .1, 1.)
RIDGE_ALPHAS = (.1, 1., 10., 100., 1000.)


def load_visits() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = {}
    for name, path in PATHS.items():
        with np.load(path) as f:
            if list(f["views"]) != ["A2C", "A3C", "A4C"]:
                raise ValueError(f"Unexpected view order: {name}")
            keys = list(zip(f["patient_ids"].astype(str), f["visit_dates"].astype(str)))
            values = f["embeddings"].astype(np.float32)
            if len(set(keys)) != len(keys) or not np.isfinite(values).all():
                raise ValueError(f"Invalid visit array: {name}")
            models[name] = dict(zip(keys, values))
    visits = pd.read_parquet(ALIGNED / "visit_alignment.parquet")
    labels = pd.read_parquet(Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet"))
    nonphil = pd.read_parquet(NONPHIL / "cohort.parquet")
    return models, visits, labels, nonphil


def vendor_visits() -> set[tuple[str, str]]:
    records = json.loads(CROPS.read_text(encoding="utf-8"))
    groups = {}
    for r in records:
        groups.setdefault((r["patient"], r["visit_date"]), set()).add(r["manufacturer"])
    return {key for key, manufacturers in groups.items()
            if manufacturers and all("GE" in vendor.upper() for vendor in manufacturers)}


def make_future_arrays(models, visits, labels, cohort):
    by_id = labels.set_index("visit_id")
    lookup = visits.set_index("visit_id")
    identifiers = {}
    for r in cohort.itertuples():
        current = lookup.loc[r.current_visit_id]
        key = (str(current.patient_id), str(current.visit_date))
        same = labels[labels.patient_id.eq(current.patient_id)].sort_values("visit_order")
        current_order = int(by_id.loc[r.current_visit_id].visit_order)
        earlier = same[same.visit_order.lt(current_order)]
        prior_key = None
        if not earlier.empty:
            earlier_id = earlier.iloc[-1].visit_id
            prior = lookup.loc[earlier_id]
            prior_key = (str(prior.patient_id), str(prior.visit_date))
        identifiers[r.transition_id] = (key, prior_key)
    arrays = {}
    for model, embeddings in models.items():
        current_rows, delta_rows, availability = [], [], []
        for transition in cohort.transition_id:
            key, prior_key = identifiers[transition]
            current = embeddings[key]
            prior = embeddings.get(prior_key) if prior_key else None
            current_rows.append(current)
            delta_rows.append(current - prior if prior is not None else np.zeros_like(current))
            availability.append(np.ones((3, 1), np.float32) if prior is not None else np.zeros((3, 1), np.float32))
        current = np.stack(current_rows)
        delta = np.stack(delta_rows)
        available = np.stack(availability)
        arrays[(model, "current")] = current
        arrays[(model, "history")] = np.concatenate([current, delta, available], axis=2)
    return arrays


def transform_video(x, train, test, representation, seed):
    fit, held = [], []
    for view in range(3):
        steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                 ("scale", StandardScaler())]
        if representation == "pca8":
            steps.extend([("pca", PCA(n_components=min(8, len(train)-1, x.shape[2]),
                                            svd_solver="randomized", random_state=seed)),
                          ("component_scale", StandardScaler())])
        elif representation != "full":
            raise ValueError(representation)
        pipeline = Pipeline(steps)
        fit.append(pipeline.fit_transform(x[train, view]))
        held.append(pipeline.transform(x[test, view]))
    return np.concatenate(fit, axis=1), np.concatenate(held, axis=1)


def fit_classifier(x, y, groups, train, test, seed):
    split = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                 .split(np.zeros(len(train)), y[train], groups[train]))
    candidates = []
    for representation in REPRESENTATIONS:
        scores = {c: [] for c in REGULARIZATION}
        for a, b in split:
            fit, val = train[a], train[b]
            if set(groups[fit]) & set(groups[val]):
                raise ValueError("Inner patient overlap")
            u, v = transform_video(x, fit, val, representation, seed)
            for c in REGULARIZATION:
                model = LogisticRegression(C=c, max_iter=3000).fit(u, y[fit])
                scores[c].append(average_precision_score(y[val], model.predict_proba(v)[:, 1]))
        candidates.extend((float(np.mean(values)), representation, c)
                          for c, values in scores.items())
    _, representation, c = max(candidates)
    u, v = transform_video(x, train, test, representation, seed)
    model = LogisticRegression(C=c, max_iter=3000).fit(u, y[train])
    return model.predict_proba(v)[:, 1], {"representation": representation, "C": c}


def ppv_top_fifth(y, score):
    count = max(1, math.ceil(len(y) * .2))
    return float(np.mean(y[np.argsort(score)[-count:]]))


def future_evaluation(out: Path, mode: str, models, visits, labels, cohort):
    out.mkdir(parents=True, exist_ok=True)
    (out / "folds").mkdir(exist_ok=True)
    arrays = make_future_arrays(models, visits, labels, cohort)
    arms = [(m, "current") for m in models]
    if mode == "stage2":
        arms += [(m, "history") for m in models]
    folds = pd.read_parquet(NONPHIL / "patient_folds.parquet")
    groups = cohort.patient_id.to_numpy(str)
    y = cohort.label.to_numpy(int)
    expected = len(arms) * 15
    completed = 0
    with threadpool_limits(limits=4):
        for (repeat, fold), assignment in folds.groupby(["repeat", "fold"]):
            held = np.flatnonzero(np.isin(groups, assignment.patient_id))
            train = np.flatnonzero(~np.isin(groups, assignment.patient_id))
            if set(groups[train]) & set(groups[held]):
                raise ValueError("Outer patient overlap")
            seed = 20260930 + 5 * int(repeat) + int(fold)
            for model, representation in arms:
                name = f"{model}_{representation}"
                dest = out / "folds" / f"{name}_r{repeat}_f{fold}.parquet"
                if dest.exists():
                    completed += 1
                    continue
                score, selection = fit_classifier(arrays[(model, representation)], y, groups,
                                                  train, held, seed)
                result = cohort.iloc[held][["transition_id", "patient_id", "label"]].copy()
                result = result.assign(model=name, repeat=int(repeat), fold=int(fold), score=score,
                                       train_prevalence=float(y[train].mean()))
                temp = dest.with_suffix(".tmp")
                result.to_parquet(temp, index=False)
                temp.replace(dest)
                atomic(dest.with_suffix(".json"), selection)
                completed += 1
                atomic(out / "status.json", {"status": "training", "completed": completed,
                                             "total": expected, "model": name, "pid": os.getpid()})
                print(f"Future {completed}/{expected}: {name} r{repeat} f{fold}", flush=True)
    files = sorted((out / "folds").glob("*.parquet"))
    if len(files) != expected:
        raise ValueError("Incomplete future folds")
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if raw.groupby(["model", "transition_id"]).size().ne(3).any():
        raise ValueError("Missing repeated predictions")
    raw.to_parquet(out / "fold_predictions.parquet", index=False)
    pred = raw.groupby(["model", "transition_id", "patient_id", "label"], as_index=False).score.mean()
    baseline = pd.read_parquet(NONPHIL / "retained_baseline_oof.parquet")
    base = baseline[["transition_id", "patient_id", "label", "score"]].copy()
    base["model"] = "strain_clinical_reference"
    frames = [pred, base]
    for name in pred.model.unique():
        t = pred[pred.model.eq(name)].merge(base, on=["transition_id", "patient_id", "label"],
                                           suffixes=("_video", "_baseline"), validate="one_to_one")
        fused = t[["transition_id", "patient_id", "label"]].copy()
        fused["model"] = name + "_late25"
        fused["score"] = .25 * t.score_video + .75 * t.score_baseline
        frames.append(fused)
    all_predictions = pd.concat(frames, ignore_index=True)
    all_predictions.to_parquet(out / "oof_predictions.parquet", index=False)
    wide = all_predictions.pivot(index=["transition_id", "patient_id", "label"],
                                 columns="model", values="score")
    if wide.isna().any().any() or len(wide) != len(cohort):
        raise ValueError("Unequal held-out cohorts")
    yy = wide.index.get_level_values("label").to_numpy()
    patient = wide.index.get_level_values("patient_id").to_numpy()
    rows = []
    for name in wide.columns:
        score = wide[name].to_numpy()
        rows.append({"model": name, "n": len(yy), "events": int(yy.sum()),
                     "auc": float(roc_auc_score(yy, score)),
                     "ap": float(average_precision_score(yy, score)),
                     "ppv_top20": ppv_top_fifth(yy, score)})
    metrics = pd.DataFrame(rows).sort_values("model")
    # Paired draws use the same resampled patients for every model.
    patients = np.unique(patient)
    indices = {p: np.flatnonzero(patient == p) for p in patients}
    rng = np.random.default_rng(20260930)
    samples = {name: {"auc": [], "ap": []} for name in wide.columns}
    for _ in range(1000):
        take = np.concatenate([indices[p] for p in rng.choice(patients, len(patients), replace=True)])
        if len(np.unique(yy[take])) < 2:
            continue
        for name in wide.columns:
            score = wide[name].to_numpy()[take]
            samples[name]["auc"].append(roc_auc_score(yy[take], score))
            samples[name]["ap"].append(average_precision_score(yy[take], score))
    deltas = []
    reference = "strain_clinical_reference"
    for name in wide.columns:
        if name == reference:
            continue
        row = {"model": name, "reference": reference}
        for metric in ("auc", "ap"):
            diff = np.array(samples[name][metric]) - np.array(samples[reference][metric])
            row[f"delta_{metric}"] = float(metrics.set_index("model").loc[name, metric] -
                                         metrics.set_index("model").loc[reference, metric])
            row[f"delta_{metric}_lo"], row[f"delta_{metric}_hi"] = np.quantile(diff, [.025, .975])
        deltas.append(row)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(deltas).to_csv(out / "paired_deltas.csv", index=False)
    return metrics


def gls_evaluation(out: Path, models, visits, labels):
    folder = out / "gls"
    folder.mkdir(parents=True, exist_ok=True)
    valid = vendor_visits()
    matched = visits[visits.match_status.eq("exact_patient_date_study_uid")].copy()
    matched = matched[[(p, d) in valid for p, d in zip(matched.patient_id, matched.visit_date)]]
    cohort = matched[["visit_id", "patient_id", "visit_date"]].merge(
        labels[["visit_id", "gls_mid_magnitude"]], on="visit_id", validate="one_to_one")
    cohort = cohort[np.isfinite(cohort.gls_mid_magnitude)].sort_values(["patient_id", "visit_date"])
    if not all((p, d) in table for table in models.values()
               for p, d in zip(cohort.patient_id, cohort.visit_date)):
        raise ValueError("Missing GLS cine embedding")
    y = cohort.gls_mid_magnitude.to_numpy(float)
    patients = cohort.patient_id.to_numpy(str)
    splits = pd.read_parquet(ALIGNED / "patient_folds.parquet")
    embeddings = {model: np.stack([table[(p, d)] for p, d in zip(cohort.patient_id, cohort.visit_date)])
                  for model, table in models.items()}
    results = []
    with threadpool_limits(limits=4):
        for (repeat, fold), assignment in splits.groupby(["repeat", "fold"]):
            held = np.flatnonzero(np.isin(patients, assignment.patient_id))
            train = np.flatnonzero(~np.isin(patients, assignment.patient_id))
            seed = 20260930 + int(repeat) * 5 + int(fold)
            for name, x in embeddings.items():
                dest = folder / f"{name}_r{repeat}_f{fold}.parquet"
                if dest.exists():
                    continue
                inner = list(GroupKFold(n_splits=3, shuffle=True, random_state=seed)
                             .split(np.zeros(len(train)), y[train], patients[train]))
                candidates = []
                for representation in REPRESENTATIONS:
                    for alpha in RIDGE_ALPHAS:
                        errors = []
                        for a, b in inner:
                            fit, val = train[a], train[b]
                            u, v = transform_video(x, fit, val, representation, seed)
                            ridge = Ridge(alpha=alpha, solver="lsqr", tol=1e-5).fit(u, y[fit])
                            errors.append(mean_absolute_error(y[val], ridge.predict(v)))
                        candidates.append((float(np.mean(errors)), representation, alpha))
                _, representation, alpha = min(candidates)
                u, v = transform_video(x, train, held, representation, seed)
                prediction = Ridge(alpha=alpha, solver="lsqr", tol=1e-5).fit(u, y[train]).predict(v)
                result = cohort.iloc[held][["visit_id", "patient_id", "visit_date"]].copy()
                result = result.assign(model=name, repeat=int(repeat), fold=int(fold),
                                       value=y[held], prediction=prediction)
                temp = dest.with_suffix(".tmp")
                result.to_parquet(temp, index=False)
                temp.replace(dest)
                print(f"GLS {name} r{repeat} f{fold}", flush=True)
    files = sorted(folder.glob("*.parquet"))
    if len(files) != 15 * len(models):
        raise ValueError("Incomplete GLS folds")
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if raw.groupby(["model", "visit_id"]).size().ne(3).any():
        raise ValueError("Missing GLS predictions")
    pred = raw.groupby(["model", "visit_id", "patient_id", "value"], as_index=False).prediction.mean()
    pred.to_parquet(folder / "oof_predictions.parquet", index=False)
    rows = []
    for name, group in pred.groupby("model"):
        rows.append({"model": name, "visits": len(group), "patients": group.patient_id.nunique(),
                     "mae": mean_absolute_error(group.value, group.prediction),
                     "pearson_r": pearsonr(group.value, group.prediction).statistic})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(folder / "metrics.csv", index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    args = parser.parse_args()
    out = ROOT / args.stage
    out.mkdir(parents=True, exist_ok=True)
    models, visits, labels, cohort = load_visits()
    protocol = {"stage": args.stage, "models": list(models), "cohort": len(cohort),
                "patients": int(cohort.patient_id.nunique()), "events": int(cohort.label.sum()),
                "folds": "retained 3x5 patient-held-out", "inner": 3,
                "representations": REPRESENTATIONS, "regularization": REGULARIZATION,
                "history": args.stage == "stage2", "glm_target": "current Mid-GLS magnitude",
                "precision": "PPV among top 20% held-out risk scores",
                "embedding_hashes": {name: hashfile(path) for name, path in PATHS.items()}}
    path = out / "protocol.json"
    normalized = json.loads(json.dumps(protocol))
    if path.exists() and json.loads(path.read_text()) != normalized:
        raise ValueError("Protocol changed; use a fresh output")
    atomic(path, normalized)
    future = future_evaluation(out / "future", args.stage, models, visits, labels, cohort)
    if args.stage == "stage1":
        gls = gls_evaluation(out, models, visits, labels)
        gls_text = gls.to_string(index=False)
    else:
        gls_text = "Stage 1 GLS probes retained; Stage 2 changes history input only."
    text = f"# {args.stage} results\n\nFuture deterioration:\n\n{future.to_string(index=False)}\n\nGLS:\n\n{gls_text}\n"
    (out / "results.md").write_text(text, encoding="utf-8")
    atomic(out / "run_complete.json", {"stage": args.stage, "future_models": future.model.tolist(),
                                       "gls_evaluated": args.stage == "stage1"})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True, exist_ok=True)
        atomic(ROOT / "evaluation_error.json", {"error": traceback.format_exc()})
        raise
