"""Paired comparison of first-to-second and first-to-last R-wave sampling."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

from ichilov3_compare_rwave_panecho import bootstrap_delta, corr


ROOT = Path(r"D:\DS\ichilov3_rwave_first_cycle_panecho_20261005")
PRIOR = Path(r"D:\DS\ichilov3_rwave_panecho_20261005")


def compare(table_a, table_b, key, prediction, filter_models, metrics):
    a = pd.read_parquet(table_a)
    b = pd.read_parquet(table_b)
    a = a[a.model.isin(filter_models)]
    b = b[b.model.isin(filter_models)]
    merged = a[key + [prediction]].merge(b[key + [prediction]], on=key,
                                            suffixes=("_first_last", "_first_cycle"),
                                            validate="one_to_one")
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError("Prediction rows did not match")
    result = {}
    grouping = ["model", "target"] if "target" in key else ["model"]
    for group_key, group in merged.groupby(grouping):
        name = "/".join(group_key) if isinstance(group_key, tuple) else group_key
        actual = "value" if "value" in key else "label"
        result[name] = {
            "rows": len(group), "patients": group.patient_id.nunique(),
            "events": int(group.label.sum()) if actual == "label" else None,
            **{name: bootstrap_delta(group, actual, prediction + "_first_last",
                                     prediction + "_first_cycle", fn)
               for name, fn in metrics.items()},
        }
    return result


def main():
    if not (ROOT / "run_complete.json").exists():
        raise RuntimeError("First-cycle trial is incomplete")
    same_visit = compare(
        PRIOR / "physiology" / "oof_predictions.parquet",
        ROOT / "physiology" / "oof_predictions.parquet",
        ["target", "model", "visit_id", "patient_id", "value"],
        "prediction", ["panecho"], {"mae": mean_absolute_error, "pearson_r": corr})
    future = compare(
        PRIOR / "late_fusion" / "paired_oof_predictions.parquet",
        ROOT / "late_fusion" / "paired_oof_predictions.parquet",
        ["model", "transition_id", "patient_id", "label"], "score",
        ["video_panecho", "late_panecho_video25", "cnn_moment_baseline"],
        {"auroc": roc_auc_score, "ap": average_precision_score})
    result = {"same_visit": same_visit, "next_visit": future,
              "changed_cines": 14,
              "reference": "first-to-last R marker",
              "new": "first-to-second R marker (one cycle)",
              "patient_held_out": True}
    (ROOT / "first_cycle_vs_first_last.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
