"""GE Healthcare-only EchoPrime Mid-GLS probe on existing patient-held-out folds.

The frozen video embeddings, patient folds and labels are read-only. Only GE
Healthcare visits train the supervised Ridge probe and its inner-fold choices.
The comparison uses the same held-out GE visits as the all-vendor-trained probe.
"""

from pathlib import Path
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ichilov3_train_late_fusion import video_preprocess


BASE = Path(r"D:\DS\ichilov3_temporal_trial_20260927")
PHYS = BASE / "physiology"
FOLD_SOURCE = Path(r"D:\DS\ichilov3_aligned_20260927\patient_folds.parquet")
VENDOR_SOURCE = BASE / "vendor_failure_analysis" / "prediction_points.csv"
OUT = Path(__file__).resolve().parent / "ge_only_gls_probe_20261005"
(OUT / "folds").mkdir(parents=True, exist_ok=True)

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
REPRESENTATIONS = ("full_embedding", "pca8_per_view")

cohort = pd.read_parquet(PHYS / "cohort.parquet")
folds = pd.read_parquet(FOLD_SOURCE)
vendor = pd.read_csv(VENDOR_SOURCE)
vendor = vendor.loc[vendor.target.eq("mid_gls") & vendor.model.eq("echoprime"),
                    ["visit_id", "vendor_short"]]
assert len(cohort) == 398 and vendor.visit_id.is_unique and len(vendor) == 398
cohort = cohort.merge(vendor, on="visit_id", validate="one_to_one")
ge = cohort.vendor_short.eq("GE Healthcare").to_numpy()
assert ge.sum() == 230 and cohort.loc[ge, "patient_id"].nunique() == 79

groups = cohort.patient_id.astype(str).to_numpy()
y = cohort.gls_mid_magnitude.to_numpy(float)
with np.load(BASE / "echoprime_visits.npz") as z:
    lookup = {(str(p), str(d)): v.copy() for p, d, v in
              zip(z["patient_ids"], z["visit_dates"], z["embeddings"])}
x = np.stack([lookup[(str(row.patient_id), str(row.visit_date))]
              for row in cohort.itertuples()])
assert x.shape == (398, 3, 512) and np.isfinite(x).all()

with threadpool_limits(limits=4):
    for (repeat, fold), fold_patients in folds.groupby(["repeat", "fold"]):
        out_file = OUT / "folds" / f"r{repeat}_f{fold}.parquet"
        meta_file = OUT / "folds" / f"r{repeat}_f{fold}.json"
        if out_file.exists() and meta_file.exists():
            print(f"cached r{repeat} f{fold}", flush=True)
            continue
        heldout = np.isin(groups, fold_patients.patient_id.to_numpy(str))
        train = np.flatnonzero(ge & ~heldout)
        test = np.flatnonzero(ge & heldout)
        assert len(train) > 150 and len(test) > 20
        assert not set(groups[train]) & set(groups[test])
        assert ge[train].all() and ge[test].all()
        seed = 20260929 + int(repeat) * 5 + int(fold)
        inner = list(GroupKFold(n_splits=3, shuffle=True, random_state=seed).split(
            np.zeros(len(train)), y[train], groups[train]))

        choices = []
        for representation in REPRESENTATIONS:
            errors = {alpha: [] for alpha in ALPHAS}
            for ia, ib in inner:
                fit, val = train[ia], train[ib]
                assert not set(groups[fit]) & set(groups[val])
                xx, vx, _ = video_preprocess(x, fit, val, representation, seed)
                for alpha in ALPHAS:
                    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-5).fit(xx, y[fit])
                    errors[alpha].append(float(mean_absolute_error(y[val], model.predict(vx))))
            choices.extend({"representation": representation, "alpha": alpha,
                            "inner_mae": float(np.mean(errors[alpha]))}
                           for alpha in ALPHAS)

        best = min(choices, key=lambda item: item["inner_mae"])
        xx, vx, _ = video_preprocess(x, train, test, best["representation"], seed)
        model = Ridge(alpha=best["alpha"], solver="lsqr", tol=1e-5).fit(xx, y[train])
        pred = model.predict(vx)
        assert np.isfinite(pred).all()
        result = cohort.iloc[test][["visit_id", "patient_id", "visit_date"]].copy()
        result = result.assign(repeat=int(repeat), fold=int(fold), value=y[test],
                               prediction=pred)
        result.to_parquet(out_file, index=False)
        meta_file.write_text(json.dumps({
            "repeat": int(repeat), "fold": int(fold),
            "train_visits": len(train), "train_patients": len(set(groups[train])),
            "test_visits": len(test), "test_patients": len(set(groups[test])),
            "best": best, "choices": choices,
        }, indent=2))
        print(f"finished r{repeat} f{fold}: train {len(train)}, test {len(test)}", flush=True)

raw = pd.concat([pd.read_parquet(f) for f in sorted((OUT / "folds").glob("r*_f*.parquet"))],
                ignore_index=True)
assert len(raw) == 690 and raw.groupby("visit_id").size().eq(3).all()
ge_only = raw.groupby(["visit_id", "patient_id", "visit_date", "value"], as_index=False).prediction.mean()
assert len(ge_only) == 230

all_vendor = pd.read_parquet(PHYS / "oof_predictions.parquet")
all_vendor = all_vendor.loc[all_vendor.target.eq("mid_gls") &
                            all_vendor.model.eq("echoprime"),
                            ["visit_id", "prediction"]].rename(columns={"prediction": "all_vendor_prediction"})
comparison = ge_only.merge(all_vendor, on="visit_id", validate="one_to_one")
assert len(comparison) == 230
comparison = comparison.rename(columns={"prediction": "ge_only_prediction"})
comparison.to_parquet(OUT / "oof_predictions.parquet", index=False)


def scores(frame, column):
    true = frame.value.to_numpy(float)
    pred = frame[column].to_numpy(float)
    return {"visits": len(frame), "patients": frame.patient_id.nunique(),
            "mae": float(mean_absolute_error(true, pred)),
            "pearson_r": float(np.corrcoef(true, pred)[0, 1]),
            "mean_signed_error": float(np.mean(pred - true))}


result = {
    "training_scope": "GE Healthcare only; frozen EchoPrime embeddings; patient-held-out Ridge probe",
    "all_ge": {key: scores(comparison, key) for key in
               ["all_vendor_prediction", "ge_only_prediction"]},
    "gt_8_24": {key: scores(comparison[comparison.value.between(8, 24)], key) for key in
                ["all_vendor_prediction", "ge_only_prediction"]},
}

# Paired patient bootstrap to compare training policies on identical visits.
rng = np.random.default_rng(20261005)
groups_by_patient = {p: g for p, g in comparison.groupby("patient_id")}
patients = np.array(list(groups_by_patient))
deltas = {"mae": [], "pearson_r": []}
for _ in range(3000):
    sample = pd.concat([groups_by_patient[p] for p in rng.choice(patients, len(patients), replace=True)],
                       ignore_index=True)
    a = scores(sample, "all_vendor_prediction")
    b = scores(sample, "ge_only_prediction")
    deltas["mae"].append(b["mae"] - a["mae"])
    deltas["pearson_r"].append(b["pearson_r"] - a["pearson_r"])
result["ge_only_minus_all_vendor_patient_bootstrap_ci95"] = {
    metric: np.quantile(values, [.025, .975]).tolist()
    for metric, values in deltas.items()
}
(OUT / "results.json").write_text(json.dumps(result, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.7), sharex=True, sharey=True)
for ax, key, color, title in zip(
    axes,
    ["all_vendor_prediction", "ge_only_prediction"],
    ["#277f98", "#d68b33"],
    ["Probe trained on all vendors", "Probe trained on GE Healthcare only"],
):
    metric = result["all_ge"][key]
    ax.scatter(comparison.value, comparison[key], s=26, alpha=.58,
               color=color, edgecolors="white", linewidths=.25)
    ax.plot([4, 28], [4, 28], "k--", alpha=.58, lw=1.1)
    ax.set(xlim=(4, 28), ylim=(4, 28), aspect="equal",
           xlabel="Reported Mid-GLS (absolute %)",
           title=f"{title}\nMAE {metric['mae']:.3f} · r {metric['pearson_r']:.3f}")
    ax.grid(alpha=.15)
axes[0].set_ylabel("Predicted Mid-GLS (absolute %)")
fig.suptitle("GE Healthcare · effect of training the GLS probe on GE only",
             fontsize=14, weight="bold")
fig.text(.5, .015, "Same 230 held-out GE visits / 79 patients; frozen EchoPrime; 3×5 patient folds",
         ha="center", fontsize=9, color="#52606d")
fig.savefig(OUT / "ge_only_vs_all_vendor_training.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print(json.dumps(result, indent=2), flush=True)
