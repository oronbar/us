"""Choose EchoPrime view subset within training patients, evaluate held-out folds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from ichilov3_train_fusion import atomic, hashfile
from ichilov3_view_fusion_ablation import ROOT as SOURCE, SUBSETS, subset_name, transform
from ichilov3_window_variability_ablation import paired_ci, score_table
from ichilov_ge_ssl_evaluate import visit_table
from ichilov_video_experiment_eval import (ALIGNED, NONPHIL, PATHS, REGULARIZATION,
                                           REPRESENTATIONS, make_future_arrays)

ROOT = Path(r"D:\DS\ichilov3_nested_view_selection_20261004")
LABELS = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet")


def inner_score(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                train: np.ndarray, seed: int) -> tuple[float, str, float]:
    split = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                 .split(np.zeros(len(train)), y[train], groups[train]))
    candidates = []
    for representation in REPRESENTATIONS:
        scores = {c: [] for c in REGULARIZATION}
        for a, b in split:
            fit, val = train[a], train[b]
            u, v = transform(x, fit, val, representation, seed)
            for c in REGULARIZATION:
                classifier = LogisticRegression(C=c, max_iter=3000).fit(u, y[fit])
                scores[c].append(average_precision_score(y[val],
                                                          classifier.predict_proba(v)[:, 1]))
        candidates.extend((float(np.mean(scores[c])), representation, c)
                          for c in REGULARIZATION)
    return max(candidates)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    decisions = ROOT / "decisions"
    decisions.mkdir(exist_ok=True)
    protocol = {"question": "Does train-only selection of the A2C/A3C/A4C subset improve future risk?",
                "models": ["frozen", "adapted_specific"],
                "input": "current and prior-visit difference",
                "candidate_subsets": [subset_name(s) for s in SUBSETS],
                "selection": "highest inner-fold AP; ties prefer all three views",
                "outer": "same 3x5 patient-held-out folds; reuse exact held-out subset predictions",
                "source_oof_sha256": hashfile(SOURCE / "fold_predictions.parquet")}
    path = ROOT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Changed protocol")
    atomic(path, protocol)
    cohort = pd.read_parquet(NONPHIL / "cohort.parquet")
    visits = pd.read_parquet(ALIGNED / "visit_alignment.parquet")
    labels = pd.read_parquet(LABELS)
    folds = pd.read_parquet(NONPHIL / "patient_folds.parquet")
    groups = cohort.patient_id.to_numpy(str)
    y = cohort.label.to_numpy(int)
    with np.load(PATHS["echoprime"]) as data:
        frozen = dict(zip(zip(data["patient_ids"].astype(str), data["visit_dates"].astype(str)),
                          data["embeddings"].astype(np.float32)))
    frozen_array = make_future_arrays({"frozen": frozen}, visits, labels, cohort)[("frozen", "history")]
    saved = pd.read_parquet(SOURCE / "fold_predictions.parquet")
    chosen = []
    with threadpool_limits(limits=4):
        for (repeat, fold), assignment in folds.groupby(["repeat", "fold"]):
            repeat, fold = int(repeat), int(fold)
            train = np.flatnonzero(~np.isin(groups, assignment.patient_id))
            seed = 20260930 + repeat*5 + fold
            adapted = visit_table("specific", repeat, fold)
            adapted_array = make_future_arrays({"adapted": adapted}, visits, labels,
                                               cohort)[("adapted", "history")]
            for model, full in (("frozen", frozen_array), ("adapted_specific", adapted_array)):
                decision = decisions / f"{model}_r{repeat}_f{fold}.json"
                if decision.exists():
                    selection = json.loads(decision.read_text())
                else:
                    scores = {subset_name(s): inner_score(full[:, s, :], y, groups, train, seed)[0]
                              for s in SUBSETS}
                    winner = max(scores, key=lambda name: (scores[name], name.count("_") + 1))
                    selection = {"model": model, "repeat": repeat, "fold": fold,
                                 "chosen_subset": winner, "inner_AP_by_subset": scores}
                    atomic(decision, selection)
                chosen.append(selection)
            print(f"Nested view selection repeat {repeat} fold {fold}", flush=True)
    selected_frames = []
    for choice in chosen:
        name = choice["model"] + "_" + choice["chosen_subset"]
        part = saved[(saved.model.eq(name)) & (saved.repeat.eq(choice["repeat"])) &
                     (saved.fold.eq(choice["fold"]))].copy()
        if part.empty:
            raise ValueError(f"Missing held-out predictions for {name}")
        part["model"] = choice["model"] + "_nested_view"
        selected_frames.append(part)
    raw = pd.concat(selected_frames, ignore_index=True)
    if raw.groupby(["model", "transition_id"]).size().ne(3).any():
        raise ValueError("Incomplete nested view OOF")
    raw.to_parquet(ROOT / "fold_predictions.parquet", index=False)
    pred = raw.groupby(["model", "transition_id", "patient_id", "label"],
                       as_index=False).score.mean()
    all_views = pd.read_parquet(SOURCE / "oof_predictions.parquet")
    anchors = all_views[all_views.model.isin(("frozen_A2C_A3C_A4C",
                                              "frozen_A2C_A3C_A4C_late25",
                                              "adapted_specific_A2C_A3C_A4C",
                                              "adapted_specific_A2C_A3C_A4C_late25",
                                              "strain_clinical_reference"))].copy()
    baseline = anchors[anchors.model.eq("strain_clinical_reference")]
    frames = [pred, anchors]
    for name, group in pred.groupby("model"):
        joined = group.merge(baseline, on=["transition_id", "patient_id", "label"],
                             suffixes=("_video", "_baseline"), validate="one_to_one")
        fused = joined[["transition_id", "patient_id", "label"]].copy()
        fused["model"] = name + "_late25"
        fused["score"] = .25*joined.score_video + .75*joined.score_baseline
        frames.append(fused)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_parquet(ROOT / "oof_predictions.parquet", index=False)
    metrics = score_table(predictions)
    metrics.to_csv(ROOT / "metrics.csv", index=False)
    pairs = []
    for model in ("frozen", "adapted_specific"):
        pairs.extend([(model+"_nested_view", model+"_A2C_A3C_A4C"),
                      (model+"_nested_view_late25", model+"_A2C_A3C_A4C_late25"),
                      (model+"_nested_view_late25", "strain_clinical_reference")])
    paired_ci(predictions, metrics, pairs, 20261004).to_csv(ROOT / "paired_deltas.csv", index=False)
    counts = {}
    for model in ("frozen", "adapted_specific"):
        counts[model] = {subset_name(s): sum(row["model"] == model and
                                              row["chosen_subset"] == subset_name(s)
                                              for row in chosen) for s in SUBSETS}
    atomic(ROOT / "selection_counts.json", counts)
    atomic(ROOT / "run_complete.json", {"outer_folds": 15, "models": metrics.model.tolist()})


if __name__ == "__main__":
    main()
