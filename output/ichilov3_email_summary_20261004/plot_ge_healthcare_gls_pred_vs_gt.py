"""Plot the original held-out GE Healthcare Mid-GLS predictions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE = Path(r"D:\DS\ichilov3_temporal_trial_20260927\vendor_failure_analysis\prediction_points.csv")
OUTPUT = Path(__file__).resolve().parent / "ge_healthcare_mid_gls_pred_vs_gt.png"

data = pd.read_csv(SOURCE)
data = data.loc[
    data.target.eq("mid_gls")
    & data.model.eq("echoprime")
    & data.vendor_short.eq("GE Healthcare")
].copy()

assert len(data) == 230
assert data.visit_id.is_unique
assert data.patient_id.nunique() == 79

gt = data.value.to_numpy(dtype=float)
pred = data.prediction.to_numpy(dtype=float)
r = float(np.corrcoef(gt, pred)[0, 1])
mae = float(np.mean(np.abs(gt - pred)))
assert np.isclose(r, 0.8165638205451579, atol=1e-8)
assert np.isclose(mae, 1.2995917560743246, atol=1e-8)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
fig, ax = plt.subplots(figsize=(7.4, 7.4))
ax.scatter(gt, pred, s=38, alpha=.65, color="#177a96", edgecolors="white", linewidths=.35)

limits = (4, 28)
ax.plot(limits, limits, linestyle="--", color="#34495e", linewidth=1.5,
        label="Perfect agreement")
coef = np.polyfit(gt, pred, 1)
xx = np.array(limits)
ax.plot(xx, np.polyval(coef, xx), color="#d78735", linewidth=2,
        label="Linear trend")

ax.set_xlim(limits)
ax.set_ylim(limits)
ax.set_aspect("equal", adjustable="box")
ax.set_xticks(np.arange(4, 29, 4))
ax.set_yticks(np.arange(4, 29, 4))
ax.set_xlabel("Reported Mid-GLS (absolute %)")
ax.set_ylabel("Predicted Mid-GLS (absolute %)")
ax.set_title("GE Healthcare · EchoPrime video → same-visit Mid-GLS", pad=18, weight="bold")
ax.grid(color="#e2e8f0", linewidth=.8)
ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=False)
ax.text(.03, .97, f"230 visits / 79 patients\nPearson r = {r:.3f}\nMAE = {mae:.2f} GLS points",
        transform=ax.transAxes, va="top", ha="left", fontsize=11,
        bbox=dict(boxstyle="round,pad=.55", facecolor="white", edgecolor="#d8e2eb", alpha=.96))
fig.text(.5, .015, "Patient-held-out out-of-fold predictions · estimated-heartbeat sampling",
         ha="center", fontsize=9, color="#52606d")
fig.subplots_adjust(left=.13, right=.97, top=.90, bottom=.11)
fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"{OUTPUT}\nn={len(data)}, patients={data.patient_id.nunique()}, r={r:.6f}, MAE={mae:.6f}")
