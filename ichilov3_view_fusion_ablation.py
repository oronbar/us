"""Paired patient-held-out ablation of EchoPrime view subsets.

The previously selected three cines and saved embeddings are read only. Each
view subset gets the same inner-fold representation/C search as the original
video-only classifier. The all-view adapter arm must reproduce its old OOF
predictions before any subset result is accepted.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ichilov3_train_fusion import atomic, hashfile
from ichilov_ge_ssl_evaluate import visit_table
from ichilov_video_experiment_eval import (ALIGNED, NONPHIL, PATHS, REGULARIZATION,
                                           REPRESENTATIONS, make_future_arrays)
from ichilov3_window_variability_ablation import paired_ci, score_table

ROOT = Path(r"D:\DS\ichilov3_view_fusion_20261004")
PRIOR_ADAPTED = Path(r"D:\DS\ichilov_ge_ssl_20260930\evaluation\specific")
VIEWS = ("A2C", "A3C", "A4C")
SUBSETS = [tuple(combination) for count in (1, 2, 3)
           for combination in itertools.combinations(range(3), count)]
LABELS = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet")


def subset_name(subset: tuple[int, ...]) -> str:
    return "_".join(VIEWS[index] for index in subset)


def transform(x: np.ndarray, train: np.ndarray, test: np.ndarray,
              representation: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    fit, held = [], []
    for view in range(x.shape[1]):
        steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                 ("scale", StandardScaler())]
        if representation == "pca8":
            steps.extend([("pca", PCA(n_components=min(8, len(train)-1, x.shape[2]),
                                            svd_solver="randomized", random_state=seed)),
                          ("component_scale", StandardScaler())])
        elif representation != "full":
            raise ValueError(representation)
        pipe = Pipeline(steps)
        fit.append(pipe.fit_transform(x[train, view]))
        held.append(pipe.transform(x[test, view]))
    return np.concatenate(fit, axis=1), np.concatenate(held, axis=1)


def classify(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
             train: np.ndarray, held: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    splits = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                  .split(np.zeros(len(train)), y[train], groups[train]))
    candidates = []
    for representation in REPRESENTATIONS:
        scores = {c: [] for c in REGULARIZATION}
        for a, b in splits:
            fit, val = train[a], train[b]
            if set(groups[fit]) & set(groups[val]):
                raise ValueError("Inner patient overlap")
            u, v = transform(x, fit, val, representation, seed)
            for c in REGULARIZATION:
                model = LogisticRegression(C=c, max_iter=3000).fit(u, y[fit])
                scores[c].append(average_precision_score(y[val], model.predict_proba(v)[:, 1]))
        candidates.extend((float(np.mean(values)), representation, c)
                          for c, values in scores.items())
    _, representation, c = max(candidates)
    u, v = transform(x, train, held, representation, seed)
    model = LogisticRegression(C=c, max_iter=3000).fit(u, y[train])
    return model.predict_proba(v)[:, 1], {"representation": representation, "C": c}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    folds_dir = ROOT / "folds"
    folds_dir.mkdir(exist_ok=True)
    protocol = {"question": "Which EchoPrime view combinations contribute to next-visit risk?",
                "models": ["frozen", "adapted_specific"], "input": "current plus prior-visit difference",
                "subsets": [subset_name(s) for s in SUBSETS],
                "folds": "unchanged 3 repeats x 5 patient-held-out folds",
                "inner": "same PCA8-or-full and C search as previous evaluation",
                "video_weight": .25, "view_selection": "descriptive only; no selection on outer outcomes",
                "adapted_reference_sha256": hashfile(PRIOR_ADAPTED / "fold_predictions.parquet"),
                "frozen_reference_sha256": hashfile(PATHS["echoprime"])}
    path = ROOT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Changed protocol; use new output directory")
    atomic(path, protocol)
    cohort = pd.read_parquet(NONPHIL / "cohort.parquet")
    visits = pd.read_parquet(ALIGNED / "visit_alignment.parquet")
    labels = pd.read_parquet(LABELS)
    folds = pd.read_parquet(NONPHIL / "patient_folds.parquet")
    groups = cohort.patient_id.to_numpy(str)
    y = cohort.label.to_numpy(int)
    if len(cohort) != 182 or int(y.sum()) != 38:
        raise ValueError("Changed evaluation cohort")
    with np.load(PATHS["echoprime"]) as data:
        frozen = dict(zip(zip(data["patient_ids"].astype(str), data["visit_dates"].astype(str)),
                          data["embeddings"].astype(np.float32)))
    frozen_array = make_future_arrays({"frozen": frozen}, visits, labels, cohort)[("frozen", "history")]
    expected = len(SUBSETS) * 2 * 15
    done = 0
    with threadpool_limits(limits=4):
        for (repeat, fold), assignment in folds.groupby(["repeat", "fold"]):
            repeat, fold = int(repeat), int(fold)
            held = np.flatnonzero(np.isin(groups, assignment.patient_id))
            train = np.flatnonzero(~np.isin(groups, assignment.patient_id))
            if not len(held) or set(groups[held]) & set(groups[train]):
                raise ValueError("Bad patient fold")
            adapted = visit_table("specific", repeat, fold)
            adapted_array = make_future_arrays({"adapted": adapted}, visits, labels, cohort)[("adapted", "history")]
            seed = 20260930 + 5 * repeat + fold
            for model, full in (("frozen", frozen_array), ("adapted_specific", adapted_array)):
                for subset in SUBSETS:
                    name = f"{model}_{subset_name(subset)}"
                    destination = folds_dir / f"{name}_r{repeat}_f{fold}.parquet"
                    if not destination.exists():
                        score, selection = classify(full[:, subset, :], y, groups, train, held, seed)
                        result = cohort.iloc[held][["transition_id", "patient_id", "label"]].copy()
                        result = result.assign(model=name, repeat=repeat, fold=fold, score=score)
                        temporary = destination.with_suffix(".tmp")
                        result.to_parquet(temporary, index=False)
                        temporary.replace(destination)
                        atomic(destination.with_suffix(".json"), selection)
                    done += 1
                    atomic(ROOT / "status.json", {"status": "running", "completed": done,
                                                  "total": expected, "pid": __import__("os").getpid()})
            print(f"View subsets: {done}/{expected} after repeat {repeat} fold {fold}", flush=True)
    files = sorted(folds_dir.glob("*.parquet"))
    if len(files) != expected:
        raise ValueError(f"Expected {expected} subset folds, got {len(files)}")
    raw = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    if raw.groupby(["model", "transition_id"]).size().ne(3).any():
        raise ValueError("Incomplete repeated predictions")
    raw.to_parquet(ROOT / "fold_predictions.parquet", index=False)
    prior = pd.read_parquet(PRIOR_ADAPTED / "fold_predictions.parquet")
    prior = prior[prior["mode"].eq("history")]
    all_adapted = raw[raw.model.eq("adapted_specific_A2C_A3C_A4C")]
    parity = all_adapted.merge(prior, on=["transition_id", "patient_id", "label", "repeat", "fold"],
                               suffixes=("_new", "_prior"), validate="one_to_one")
    max_error = float(np.max(np.abs(parity.score_new - parity.score_prior)))
    if len(parity) != len(all_adapted) or max_error > 1e-6:
        raise ValueError(f"All-view adapter does not reproduce prior OOF: {max_error}")
    pred = raw.groupby(["model", "transition_id", "patient_id", "label"],
                       as_index=False).score.mean()
    baseline = pd.read_parquet(NONPHIL / "retained_baseline_oof.parquet")
    base = baseline[["transition_id", "patient_id", "label", "score"]].copy()
    base["model"] = "strain_clinical_reference"
    results = [pred, base]
    for name, group in pred.groupby("model"):
        joined = group.merge(base, on=["transition_id", "patient_id", "label"],
                             suffixes=("_video", "_baseline"), validate="one_to_one")
        fused = joined[["transition_id", "patient_id", "label"]].copy()
        fused["model"] = name + "_late25"
        fused["score"] = .25 * joined.score_video + .75 * joined.score_baseline
        results.append(fused)
    predictions = pd.concat(results, ignore_index=True)
    predictions.to_parquet(ROOT / "oof_predictions.parquet", index=False)
    metrics = score_table(predictions)
    metrics.to_csv(ROOT / "metrics.csv", index=False)
    pairs = []
    for model in ("frozen", "adapted_specific"):
        anchor = f"{model}_A2C_A3C_A4C"
        for subset in SUBSETS:
            candidate = f"{model}_{subset_name(subset)}"
            if candidate != anchor:
                pairs.extend([(candidate, anchor), (candidate + "_late25", anchor + "_late25")])
    paired_ci(predictions, metrics, pairs, 20261004).to_csv(ROOT / "paired_deltas.csv", index=False)
    atomic(ROOT / "parity.json", {"adapted_all_view_prior_max_error": max_error})
    atomic(ROOT / "run_complete.json", {"folds": expected, "models": metrics.model.tolist()})
    atomic(ROOT / "status.json", {"status": "complete", "completed": expected,
                                  "total": expected})


if __name__ == "__main__":
    main()
