"""Nested, patient-held-out linear calibration of GE Mid-GLS predictions.

The original EchoPrime embeddings, outer-fold Ridge predictions, and patient
folds are read-only. For each outer fold, a calibrator is fitted on inner-fold
out-of-fold predictions from outer-training patients only. Two prespecified
calibrators are evaluated: all-vendor and GE Healthcare-only.
"""

from pathlib import Path
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ichilov3_train_late_fusion import video_preprocess


BASE = Path(r"D:\DS\ichilov3_temporal_trial_20260927")
PHYS = BASE / "physiology"
FOLDS = Path(r"D:\DS\ichilov3_aligned_20260927\patient_folds.parquet")
VENDOR = BASE / "vendor_failure_analysis" / "prediction_points.csv"
OUT = Path(__file__).resolve().parent / "ge_gls_calibration_20261005"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "folds").mkdir(exist_ok=True)

cohort = pd.read_parquet(PHYS / "cohort.parquet")
groups = cohort.patient_id.astype(str).to_numpy()
y = cohort.gls_mid_magnitude.to_numpy(float)
assert len(cohort) == 398 and np.isfinite(y).all()

with np.load(BASE / "echoprime_visits.npz") as z:
    embedding = {(str(p), str(d)): v.copy() for p, d, v in
                 zip(z["patient_ids"], z["visit_dates"], z["embeddings"])}
x = np.stack([embedding[(str(row.patient_id), str(row.visit_date))]
              for row in cohort.itertuples()])
assert x.shape == (398, 3, 512)

vendor = pd.read_csv(VENDOR)
vendor = vendor.loc[(vendor.target == "mid_gls") & (vendor.model == "echoprime"),
                    ["visit_id", "vendor_short"]]
assert vendor.visit_id.is_unique and len(vendor) == 398
vendor_lookup = vendor.set_index("visit_id").vendor_short
ge = cohort.visit_id.map(vendor_lookup).eq("GE Healthcare").to_numpy()
assert ge.sum() == 230

folds = pd.read_parquet(FOLDS)
assert set(folds.columns) >= {"repeat", "fold", "patient_id"}

with threadpool_limits(limits=4):
    for (repeat, fold), ids in folds.groupby(["repeat", "fold"]):
        output = OUT / "folds" / f"r{repeat}_f{fold}.parquet"
        info_path = OUT / "folds" / f"r{repeat}_f{fold}.json"
        if output.exists() and info_path.exists():
            print(f"cached r{repeat} f{fold}", flush=True)
            continue
        heldout = np.isin(groups, ids.patient_id.to_numpy(str))
        train = np.flatnonzero(~heldout)
        test = np.flatnonzero(heldout)
        assert not set(groups[train]) & set(groups[test])
        seed = 20260929 + int(repeat) * 5 + int(fold)
        info = json.loads((PHYS / "models" /
                           f"echoprime_mid_gls_r{repeat}_f{fold}.json").read_text())
        best = info["best"]
        representation = best["representation"]
        alpha = float(best["alpha"])

        inner_oof = np.full(len(cohort), np.nan)
        splits = GroupKFold(n_splits=3, shuffle=True, random_state=seed).split(
            np.zeros(len(train)), y[train], groups[train])
        for ia, ib in splits:
            fit, validation = train[ia], train[ib]
            assert not set(groups[fit]) & set(groups[validation])
            xx, vx, _ = video_preprocess(x, fit, validation, representation, seed)
            probe = Ridge(alpha=alpha, solver="lsqr", tol=1e-5).fit(xx, y[fit])
            inner_oof[validation] = probe.predict(vx)
        assert np.isfinite(inner_oof[train]).all()

        original = pd.read_parquet(
            PHYS / "folds" / f"echoprime_mid_gls_r{repeat}_f{fold}.parquet")
        assert set(original.visit_id) == set(cohort.iloc[test].visit_id)
        result = original[["visit_id", "patient_id", "repeat", "fold", "value", "prediction"]].copy()
        result = result.rename(columns={"prediction": "raw_prediction"})
        calibration_info = {}
        for name, mask in [("all_vendor", np.ones(len(train), dtype=bool)),
                           ("ge_only", ge[train])]:
            fit_idx = train[mask]
            assert len(fit_idx) > 100 and np.isfinite(inner_oof[fit_idx]).all()
            calibration = LinearRegression().fit(
                inner_oof[fit_idx, None], y[fit_idx])
            slope = float(calibration.coef_[0])
            intercept = float(calibration.intercept_)
            assert np.isfinite([slope, intercept]).all() and slope > 0
            result[f"{name}_prediction"] = calibration.predict(
                result.raw_prediction.to_numpy(float)[:, None])
            calibration_info[name] = {
                "slope": slope, "intercept": intercept,
                "training_visits": len(fit_idx),
                "training_patients": int(np.unique(groups[fit_idx]).size),
            }
        result.to_parquet(output, index=False)
        info_path.write_text(json.dumps({"repeat": int(repeat), "fold": int(fold),
                                         "representation": representation,
                                         "ridge_alpha": alpha,
                                         "calibrators": calibration_info}, indent=2))
        print(f"finished r{repeat} f{fold}", flush=True)

raw = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "folds").glob("r*_f*.parquet"))],
                ignore_index=True)
assert raw.groupby("visit_id").size().eq(3).all() and raw.visit_id.nunique() == 398
oof = raw.groupby(["visit_id", "patient_id", "value"], as_index=False)[
    ["raw_prediction", "all_vendor_prediction", "ge_only_prediction"]].mean()
oof = oof.merge(vendor, on="visit_id", validate="one_to_one")

reference = pd.read_parquet(PHYS / "oof_predictions.parquet")
reference = reference.loc[(reference.model == "echoprime") &
                          (reference.target == "mid_gls"), ["visit_id", "prediction"]]
check = oof.merge(reference, on="visit_id", validate="one_to_one")
assert np.allclose(check.raw_prediction, check.prediction, atol=1e-5)
oof.to_parquet(OUT / "oof_predictions.parquet", index=False)


def metrics(data, column):
    true = data.value.to_numpy(float)
    pred = data[column].to_numpy(float)
    return {"visits": len(data), "patients": data.patient_id.nunique(),
            "mae": float(mean_absolute_error(true, pred)),
            "pearson_r": float(np.corrcoef(true, pred)[0, 1]),
            "mean_signed_error": float(np.mean(pred - true))}


results = {}
for name, subset in [("all_vendors", oof),
                     ("ge_healthcare", oof[oof.vendor_short == "GE Healthcare"]),
                     ("ge_healthcare_gt_8_24", oof[
                         oof.vendor_short.eq("GE Healthcare") & oof.value.between(8, 24)])]:
    results[name] = {k: metrics(subset, k) for k in
                     ["raw_prediction", "all_vendor_prediction", "ge_only_prediction"]}

calibrations = [json.loads(p.read_text())["calibrators"] for p in
                sorted((OUT / "folds").glob("r*_f*.json"))]
results["calibrator_ranges"] = {
    name: {field: [min(c[name][field] for c in calibrations),
                   max(c[name][field] for c in calibrations)]
           for field in ["slope", "intercept"]}
    for name in ["all_vendor", "ge_only"]
}
ge_oof = oof[oof.vendor_short.eq("GE Healthcare")]

# Paired patient bootstrap: resample patients, keeping all their visits together.
rng = np.random.default_rng(20261005)
by_patient = {p: g for p, g in ge_oof.groupby("patient_id")}
patients = np.array(list(by_patient))
for variant in ["all_vendor_prediction", "ge_only_prediction"]:
    differences = {"mae": [], "pearson_r": []}
    for _ in range(3000):
        sampled = pd.concat([by_patient[p] for p in rng.choice(patients, size=len(patients), replace=True)],
                            ignore_index=True)
        current = metrics(sampled, variant)
        baseline = metrics(sampled, "raw_prediction")
        differences["mae"].append(current["mae"] - baseline["mae"])
        differences["pearson_r"].append(current["pearson_r"] - baseline["pearson_r"])
    results["ge_healthcare"][variant]["paired_patient_bootstrap_delta_ci95"] = {
        metric: np.quantile(values, [.025, .975]).tolist()
        for metric, values in differences.items()
    }
(OUT / "results.json").write_text(json.dumps(results, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.7), sharex=True, sharey=True)
for ax, key, color, title in zip(
    axes,
    ["raw_prediction", "all_vendor_prediction"],
    ["#277f98", "#d68b33"],
    ["Original", "Train-only calibrated"],
):
    m = results["ge_healthcare"][key]
    ax.scatter(ge_oof.value, ge_oof[key], s=26, alpha=.58, c=color,
               edgecolors="white", linewidths=.25)
    ax.plot([4, 28], [4, 28], "k--", alpha=.58, lw=1.1)
    ax.set(xlim=(4, 28), ylim=(4, 28), aspect="equal",
           xlabel="Reported Mid-GLS (absolute %)",
           title=f"{title}\nMAE {m['mae']:.3f} · r {m['pearson_r']:.3f}")
    ax.grid(alpha=.15)
axes[0].set_ylabel("Predicted Mid-GLS (absolute %)")
fig.suptitle("GE Healthcare · held-out GLS predictions before/after calibration",
             fontsize=14, weight="bold")
fig.text(.5, .015,
         "230 visits / 79 patients; all-vendor calibrator fitted inside each outer training fold; dashed line = perfect agreement",
         ha="center", fontsize=9, color="#52606d")
fig.savefig(OUT / "calibration_before_after.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print(json.dumps(results, indent=2), flush=True)
