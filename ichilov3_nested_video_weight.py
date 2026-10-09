"""Nested, patient-held-out fusion-weight test for adapted EchoPrime history.

The compact strain/clinical baseline is re-fitted inside every outer and inner
fold. This deliberately avoids learning a blend weight from the separately
retained CNN+MOMENT model's outer-fold OOF predictions, which would leak its
training labels across the proposed weight-selection split.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ichilov3_train_fusion import atomic, hashfile
from ichilov3_view_fusion_ablation import transform as transform_video
from ichilov3_window_variability_ablation import paired_ci, score_table
from ichilov_ge_ssl_evaluate import visit_table
from ichilov_video_experiment_eval import (ALIGNED, NONPHIL, REGULARIZATION,
                                           REPRESENTATIONS, make_future_arrays)

ROOT = Path(r"D:\DS\ichilov3_nested_video_weight_20261004")
LABELS = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet")
CS_BASE = (.001, .01, .1, 1.)
WEIGHTS = (0., .05, .1, .25, .5, 1.)


def baseline_transform(features: dict, fit: np.ndarray, val: np.ndarray,
                       seed: int) -> tuple[np.ndarray, np.ndarray]:
    blocks = (("clinical", features["clinical"], 0),
              ("strain_scalars", features["strain_scalars"], 8),
              ("strain_curves", features["strain_curves"].reshape(len(features["clinical"]), -1), 8))
    train_parts, val_parts = [], []
    for _, values, components in blocks:
        steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                 ("scale", StandardScaler())]
        if components:
            steps.extend([("pca", PCA(n_components=min(components, len(fit)-1,
                                                     values.shape[1]),
                                         svd_solver="randomized", random_state=seed)),
                          ("component_scale", StandardScaler())])
        pipe = Pipeline(steps)
        train_parts.append(pipe.fit_transform(values[fit]))
        val_parts.append(pipe.transform(values[val]))
    return np.concatenate(train_parts, axis=1), np.concatenate(val_parts, axis=1)


def calibrator(oof_score: np.ndarray, y: np.ndarray):
    x = logit(np.clip(oof_score, 1e-5, 1-1e-5)).reshape(-1, 1)
    return LogisticRegression(C=1., max_iter=1000).fit(x, y)


def apply_calibration(model, score: np.ndarray) -> np.ndarray:
    x = logit(np.clip(score, 1e-5, 1-1e-5)).reshape(-1, 1)
    return model.predict_proba(x)[:, 1]


def select_and_predict(video: np.ndarray, features: dict, y: np.ndarray,
                       groups: np.ndarray, train: np.ndarray,
                       held: np.ndarray, seed: int) -> tuple[dict, dict]:
    splits = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                  .split(np.zeros(len(train)), y[train], groups[train]))
    candidates_base = {}
    candidates_video = {}
    for c in CS_BASE:
        candidates_base[c] = np.full(len(train), np.nan)
    for representation in REPRESENTATIONS:
        for c in REGULARIZATION:
            candidates_video[(representation, c)] = np.full(len(train), np.nan)
    for a, b in splits:
        fit, val = train[a], train[b]
        if set(groups[fit]) & set(groups[val]):
            raise ValueError("Inner patient overlap")
        u_base, v_base = baseline_transform(features, fit, val, seed)
        for c, oof in candidates_base.items():
            model = LogisticRegression(C=c, max_iter=3000).fit(u_base, y[fit])
            oof[b] = model.predict_proba(v_base)[:, 1]
        for representation in REPRESENTATIONS:
            u_video, v_video = transform_video(video, fit, val, representation, seed)
            for c in REGULARIZATION:
                model = LogisticRegression(C=c, max_iter=3000).fit(u_video, y[fit])
                candidates_video[(representation, c)][b] = model.predict_proba(v_video)[:, 1]
    if any(not np.isfinite(a).all() for a in list(candidates_base.values()) +
           list(candidates_video.values())):
        raise ValueError("Incomplete inner OOF scores")
    base_c = max(CS_BASE, key=lambda c: (average_precision_score(y[train], candidates_base[c]), -c))
    video_rep, video_c = max(candidates_video, key=lambda key:
                             (average_precision_score(y[train], candidates_video[key]),
                              key[0], -key[1]))
    base_oof = candidates_base[base_c]
    video_oof = candidates_video[(video_rep, video_c)]
    base_cal = calibrator(base_oof, y[train])
    video_cal = calibrator(video_oof, y[train])
    base_inner = apply_calibration(base_cal, base_oof)
    video_inner = apply_calibration(video_cal, video_oof)
    by_weight = {w: float(average_precision_score(y[train], (1-w)*base_inner + w*video_inner))
                 for w in WEIGHTS}
    # A tie selects the smaller video contribution, including zero.
    weight = max(WEIGHTS, key=lambda w: (by_weight[w], -w))
    u_base, v_base = baseline_transform(features, train, held, seed)
    u_video, v_video = transform_video(video, train, held, video_rep, seed)
    base_raw = LogisticRegression(C=base_c, max_iter=3000).fit(u_base, y[train]).predict_proba(v_base)[:, 1]
    video_raw = LogisticRegression(C=video_c, max_iter=3000).fit(u_video, y[train]).predict_proba(v_video)[:, 1]
    base = apply_calibration(base_cal, base_raw)
    video_score = apply_calibration(video_cal, video_raw)
    scores = {"compact_baseline": base,
              "adapted_video": video_score,
              "fusion_fixed25": .75*base + .25*video_score,
              "fusion_nested_weight": (1-weight)*base + weight*video_score}
    selection = {"base_C": base_c, "video_representation": video_rep,
                 "video_C": video_c, "chosen_video_weight": weight,
                 "inner_AP_by_weight": by_weight,
                 "inner_AP_baseline": average_precision_score(y[train], base_inner),
                 "inner_AP_video": average_precision_score(y[train], video_inner),
                 "train_patients": len(set(groups[train])), "test_patients": len(set(groups[held]))}
    return scores, selection


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    folds_dir = ROOT / "folds"
    folds_dir.mkdir(exist_ok=True)
    protocol = {"question": "Can a train-selected video weight improve next-visit prediction?",
                "video": "GE adapted EchoPrime, three views, current plus prior visit change",
                "baseline": "refitted compact clinical + strain PCA/logistic model",
                "calibration": "Platt logistic on each modality's inner patient-fold OOF logit scores",
                "inner": "three StratifiedGroupKFold splits on each outer-training set",
                "weight_grid": WEIGHTS, "baseline_C_grid": CS_BASE,
                "video_representation_grid": REPRESENTATIONS,
                "video_C_grid": REGULARIZATION,
                "outer": "same 3x5 patient-held-out folds on 182 transitions",
                "retained_CNN_MOMENT": "reference only; not used to select weight because its OOF train scores are not nested",
                "features_sha256": hashfile(NONPHIL / "features.npz"),
                "folds_sha256": hashfile(NONPHIL / "patient_folds.parquet")}
    path = ROOT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError("Changed protocol; use new output directory")
    atomic(path, protocol)
    cohort = pd.read_parquet(NONPHIL / "cohort.parquet")
    folds = pd.read_parquet(NONPHIL / "patient_folds.parquet")
    visits = pd.read_parquet(ALIGNED / "visit_alignment.parquet")
    labels = pd.read_parquet(LABELS)
    with np.load(NONPHIL / "features.npz") as data:
        features = {name: data[name].copy() for name in ("clinical", "strain_scalars", "strain_curves")}
    y = cohort.label.to_numpy(int)
    groups = cohort.patient_id.to_numpy(str)
    if len(cohort) != 182 or int(y.sum()) != 38:
        raise ValueError("Changed cohort")
    done = 0
    with threadpool_limits(limits=4):
        for (repeat, fold), assignment in folds.groupby(["repeat", "fold"]):
            repeat, fold = int(repeat), int(fold)
            held = np.flatnonzero(np.isin(groups, assignment.patient_id))
            train = np.flatnonzero(~np.isin(groups, assignment.patient_id))
            if not len(held) or set(groups[held]) & set(groups[train]):
                raise ValueError("Bad patient fold")
            destination = folds_dir / f"r{repeat}_f{fold}.parquet"
            if not destination.exists():
                table = visit_table("specific", repeat, fold)
                video = make_future_arrays({"adapted": table}, visits, labels,
                                           cohort)[("adapted", "history")]
                scores, selection = select_and_predict(video, features, y, groups,
                                                       train, held, 20260930 + repeat*5 + fold)
                rows = []
                for name, score in scores.items():
                    result = cohort.iloc[held][["transition_id", "patient_id", "label"]].copy()
                    result = result.assign(model=name, repeat=repeat, fold=fold, score=score)
                    rows.append(result)
                temporary = destination.with_suffix(".tmp")
                pd.concat(rows, ignore_index=True).to_parquet(temporary, index=False)
                temporary.replace(destination)
                atomic(destination.with_suffix(".json"), selection)
            done += 1
            atomic(ROOT / "status.json", {"status": "running", "completed": done,
                                          "total": 15})
            print(f"Nested fusion {done}/15: repeat {repeat} fold {fold}", flush=True)
    files = sorted(folds_dir.glob("r*_f*.parquet"))
    if len(files) != 15:
        raise ValueError("Incomplete folds")
    raw = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    if raw.groupby(["model", "transition_id"]).size().ne(3).any():
        raise ValueError("Missing repeated prediction")
    raw.to_parquet(ROOT / "fold_predictions.parquet", index=False)
    pred = raw.groupby(["model", "transition_id", "patient_id", "label"],
                       as_index=False).score.mean()
    retained = pd.read_parquet(NONPHIL / "retained_baseline_oof.parquet")
    reference = retained[["transition_id", "patient_id", "label", "score"]].copy()
    reference["model"] = "retained_CNN_MOMENT_reference"
    predictions = pd.concat([pred, reference], ignore_index=True)
    if predictions.groupby("model").size().nunique() != 1:
        raise ValueError("Unequal model cohorts")
    predictions.to_parquet(ROOT / "oof_predictions.parquet", index=False)
    metrics = score_table(predictions)
    metrics["brier"] = [brier_score_loss(group.label, group.score)
                        for name in metrics.model
                        for group in [predictions[predictions.model.eq(name)]]]
    metrics.to_csv(ROOT / "metrics.csv", index=False)
    pairs = [("fusion_nested_weight", "compact_baseline"),
             ("fusion_fixed25", "compact_baseline"),
             ("fusion_nested_weight", "fusion_fixed25"),
             ("fusion_nested_weight", "retained_CNN_MOMENT_reference")]
    paired_ci(predictions, metrics, pairs, 20261004).to_csv(ROOT / "paired_deltas.csv", index=False)
    selections = [json.loads(path.read_text()) for path in sorted(folds_dir.glob("r*_f*.json"))]
    counts = {str(w): sum(item["chosen_video_weight"] == w for item in selections) for w in WEIGHTS}
    atomic(ROOT / "weight_selection.json", {"folds": len(selections), "counts": counts,
                                            "mean_weight": float(np.mean([item["chosen_video_weight"]
                                                                           for item in selections]))})
    atomic(ROOT / "run_complete.json", {"folds": 15, "models": metrics.model.tolist()})
    atomic(ROOT / "status.json", {"status": "complete", "completed": 15, "total": 15})


if __name__ == "__main__":
    main()
