"""Build a non-Philips, current-video-only subset for the next-visit trial.

Source artifacts remain untouched. The target visit contributes only its label.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(r"D:\DS\ichilov3_temporal_trial_20260927\aligned")
MANIFEST = Path(r"D:\DS\ichilov3_stage2_padded_20260926\full_manifest.json")
OUTPUT = Path(r"D:\DS\ichilov3_nonphilips_future_20260930\aligned")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(SOURCE / "cohort.parquet")
    clips = pd.DataFrame(json.loads(MANIFEST.read_text(encoding="utf-8")))
    visit_vendor = clips.groupby(["patient", "visit_date"]).manufacturer.agg(
        lambda values: tuple(sorted(set(values)))
    ).reset_index()
    if not visit_vendor.manufacturer.map(len).eq(1).all():
        raise ValueError("A visit contains multiple manufacturers")
    visit_vendor["manufacturer"] = visit_vendor.manufacturer.str[0]
    annotated = cohort.merge(
        visit_vendor,
        left_on=["patient_id", "current_visit_date"],
        right_on=["patient", "visit_date"],
        validate="many_to_one",
    )
    if len(annotated) != len(cohort):
        raise ValueError("Missing current-visit manufacturer")
    keep = annotated.manufacturer.ne("Philips Medical Systems").to_numpy()
    retained = annotated.loc[keep, cohort.columns].reset_index(drop=True)
    retained.to_parquet(OUTPUT / "cohort.parquet", index=False)
    annotated.to_parquet(OUTPUT.parent / "source_cohort_vendor_audit.parquet", index=False)

    with np.load(SOURCE / "features.npz") as features:
        assert np.array_equal(features["transition_ids"], cohort.transition_id.to_numpy(str))
        filtered = {
            name: (features[name][keep] if features[name].shape[0] == len(cohort)
                   and name not in {"views", "strain_layers", "strain_channel_names"}
                   else features[name])
            for name in features.files
        }
    assert np.array_equal(filtered["transition_ids"], retained.transition_id.to_numpy(str))
    np.savez_compressed(OUTPUT / "features.npz", **filtered)
    np.savez_compressed(
        OUTPUT / "labels.npz",
        transition_ids=retained.transition_id.to_numpy(str),
        patient_ids=retained.patient_id.to_numpy(str),
        label=retained.label.to_numpy(np.int64),
        task=np.array("mid_first_rel15"),
    )

    folds = pd.read_parquet(SOURCE / "patient_folds.parquet")
    folds = folds[folds.patient_id.isin(retained.patient_id)].copy()
    folds.to_parquet(OUTPUT / "patient_folds.parquet", index=False)
    baseline = pd.read_parquet(SOURCE / "retained_baseline_oof.parquet")
    baseline = baseline[baseline.transition_id.isin(retained.transition_id)].copy()
    baseline.to_parquet(OUTPUT / "retained_baseline_oof.parquet", index=False)
    if len(baseline) != len(retained):
        raise ValueError("Incomplete retained baseline")

    fold_audit = []
    for (repeat, fold), assignment in folds.groupby(["repeat", "fold"]):
        test = retained[retained.patient_id.isin(assignment.patient_id)]
        train = retained[~retained.patient_id.isin(assignment.patient_id)]
        fold_audit.append(dict(repeat=int(repeat), fold=int(fold),
                               train_n=len(train), train_events=int(train.label.sum()),
                               test_n=len(test), test_events=int(test.label.sum())))
    pd.DataFrame(fold_audit).to_csv(OUTPUT.parent / "fold_audit.csv", index=False)
    if any(min(x["train_events"], x["test_events"]) < 1 for x in fold_audit):
        raise ValueError("A retained fold lacks positive events")

    summary = dict(
        source_transitions=len(cohort), excluded_philips=int((~keep).sum()),
        retained_transitions=len(retained), retained_patients=int(retained.patient_id.nunique()),
        retained_events=int(retained.label.sum()),
        vendor_counts=annotated.groupby("manufacturer").agg(
            transitions=("label", "size"), events=("label", "sum")
        ).to_dict("index"),
        note="Only current-visit Philips cines are excluded. No source DICOM or existing output is changed.",
    )
    (OUTPUT.parent / "subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
