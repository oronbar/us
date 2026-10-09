"""Paired GE-only differences between HR-estimated and R-anchored PanEcho."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

from ichilov3_compare_rwave_panecho import bootstrap_delta, corr


ROOT = Path(r"D:\DS\ichilov3_rwave_panecho_20261005\ge_only")


def main():
    gls = pd.read_parquet(ROOT / "gls" / "oof_predictions.parquet")
    wide = gls.pivot(index=["visit_id", "patient_id", "value"],
                     columns="model", values="prediction").reset_index()
    gls_comparison = {
        "mae": bootstrap_delta(wide, "value", "panecho_hr_estimated",
                               "panecho_rwave", mean_absolute_error),
        "pearson_r": bootstrap_delta(wide, "value", "panecho_hr_estimated",
                                     "panecho_rwave", corr),
    }
    future = pd.read_parquet(ROOT / "future" / "oof_predictions.parquet")
    wide = future.pivot(index=["transition_id", "patient_id", "label"],
                        columns="model", values="score").reset_index()
    future_comparison = {}
    for arm, old, new in (
        ("video_only", "panecho_hr_estimated_current", "panecho_rwave_current"),
        ("fixed_25_percent_fusion", "panecho_hr_estimated_current_late25",
         "panecho_rwave_current_late25"),
    ):
        future_comparison[arm] = {
            "auroc": bootstrap_delta(wide, "label", old, new, roc_auc_score),
            "ap": bootstrap_delta(wide, "label", old, new, average_precision_score),
        }
    result = {"same_visit_mid_gls": gls_comparison,
              "next_visit_deterioration": future_comparison,
              "n_gls_visits": len(gls[gls.model.eq("panecho_rwave")]),
              "n_transitions": len(wide),
              "caveat": "R-wave arm changes both time anchoring and number of windows."}
    (ROOT / "paired_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
