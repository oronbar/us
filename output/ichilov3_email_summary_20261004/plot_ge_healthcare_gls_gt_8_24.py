"""Plot held-out GE Healthcare Mid-GLS predictions for reported GT 8-24%."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE = Path(r"D:\DS\ichilov3_temporal_trial_20260927\vendor_failure_analysis\prediction_points.csv")
OUTPUT = Path(__file__).resolve().parent / "ge_healthcare_mid_gls_pred_vs_gt_8_24.png"

all_ge = pd.read_csv(SOURCE)
all_ge = all_ge.loc[
    all_ge.target.eq("mid_gls")
    & all_ge.model.eq("echoprime")
    & all_ge.vendor_short.eq("GE Healthcare")
].copy()
assert len(all_ge) == 230 and all_ge.visit_id.is_unique

kept = all_ge.loc[all_ge.value.between(8, 24, inclusive="both")].copy()
assert len(kept) == 227 and kept.patient_id.nunique() == 79

gt = kept.value.to_numpy(float)
pred = kept.prediction.to_numpy(float)
r = float(np.corrcoef(gt, pred)[0, 1])
mae = float(np.mean(np.abs(pred - gt)))
assert np.isclose(r, 0.8047465778787506, atol=1e-8)
assert np.isclose(mae, 1.2516816762114535, atol=1e-8)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
fig, ax = plt.subplots(figsize=(7.4, 7.4))
ax.scatter(gt, pred, s=38, alpha=.65, color="#177a96",
           edgecolors="white", linewidths=.35)

lim = (8, 24)
ax.plot(lim, lim, linestyle="--", color="#34495e", linewidth=1.5,
        label="Perfect agreement")
fit = np.polyfit(gt, pred, 1)
ax.plot(lim, np.polyval(fit, lim), color="#d78735", linewidth=2,
        label="Linear trend")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect("equal", adjustable="box")
ax.set_xticks(np.arange(8, 25, 2))
ax.set_yticks(np.arange(8, 25, 2))
ax.set_xlabel("Reported Mid-GLS (absolute %)")
ax.set_ylabel("Predicted Mid-GLS (absolute %)")
ax.set_title("GE Healthcare · GT restricted to 8–24%", pad=18, weight="bold")
ax.grid(color="#e2e8f0", linewidth=.8)
ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=False)
ax.text(.03, .97,
        f"227 visits / 79 patients\nPearson r = {r:.3f}\nMAE = {mae:.2f} GLS points\n3 extreme-GT visits excluded",
        transform=ax.transAxes, va="top", ha="left", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=.55", facecolor="white",
                  edgecolor="#d8e2eb", alpha=.96))
fig.text(.5, .015,
         "Post-hoc GT filter, inclusive 8–24%; same held-out predictions, no retraining",
         ha="center", fontsize=9, color="#52606d")
fig.subplots_adjust(left=.13, right=.97, top=.90, bottom=.11)
fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"{OUTPUT}\nn={len(kept)}, patients={kept.patient_id.nunique()}, r={r:.6f}, MAE={mae:.6f}")
