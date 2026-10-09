from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
})

# All rows use the same 182 GE/non-Philips transitions, 86 patients and 38 events.
names = [
    "Strain + clinical baseline",
    "Frozen EchoPrime",
    "Frozen PanEcho",
    "Frozen EchoFM",
    "GE-adapted EchoPrime (view-specific, history)",
    "Frozen EchoJEPA",
]
auc = np.array([.726, .701, .722, .713, .727, .720])
ap = np.array([.379, .347, .378, .387, .355, .368])
colors = ["#143c72"] + ["#288f9e"] * 5
y = np.arange(len(names))[::-1]
fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9), sharey=True,
                         gridspec_kw={"wspace": .08})
for ax, vals, label, lim, ref in zip(
    axes, [auc, ap], ["AUROC", "Average precision (AP)"],
    [(0.65, .765), (.30, .41)], [.726, .379]
):
    ax.axvline(ref, color="#143c72", linestyle="--", lw=1.3, alpha=.65)
    for yi, v, c in zip(y, vals, colors):
        ax.plot([lim[0], v], [yi, yi], color=c, alpha=.25, lw=2)
        ax.scatter(v, yi, s=65, color=c, zorder=3)
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_xlim(*lim)
    ax.set_xlabel(label, weight="bold")
    ax.grid(axis="x", color="#e2e8f0", zorder=0)
    ax.set_axisbelow(True)
axes[0].set_yticks(y, names)
axes[0].tick_params(axis="y", length=0)
axes[1].tick_params(axis="y", left=False)
fig.suptitle("Prediction of deterioration at the next visit", fontsize=15, weight="bold")
fig.text(.5, .045,
         "Same 182 non-Philips transitions; 3 × 5 patient-held-out folds. Dashed line = strain/clinical baseline.\n"
         "Fixed 25% video-score fusion; adapted EchoPrime uses prior-visit video when available. Exploratory comparison.",
         ha="center", fontsize=9, color="#52606d")
fig.subplots_adjust(left=.40, right=.96, top=.86, bottom=.19)
fig.savefig(OUT / "next_visit_results_without_heads.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), gridspec_kw={"wspace": .34})
ax = axes[0]
x = np.arange(2)
w = .32
before = [1.607, 2.025]
after = [1.459, 1.603]
ax.bar(x-w/2, before, w, label="Earlier temporal sampling", color="#9ab4ca")
ax.bar(x+w/2, after, w, label="Estimated heartbeat sampling", color="#258f9d")
for i in range(2):
    ax.text(x[i]-w/2, before[i]+.04, f"{before[i]:.3f}", ha="center", fontsize=10)
    ax.text(x[i]+w/2, after[i]+.04, f"{after[i]:.3f}", ha="center", fontsize=10)
ax.set_xticks(x, ["EchoPrime", "PanEcho"])
ax.set_ylim(0, 2.45)
ax.set_ylabel("Mid-GLS MAE (percentage points)")
ax.set_title("Temporal sampling · 398 matched visits", weight="bold")
ax.legend(frameon=False, fontsize=9, loc="upper left")

ax = axes[1]
vendors = ["GE Healthcare\nn=230, MAE=1.30", "GE Vingmed\nn=68, MAE=1.65", "Philips\nn=100, MAE=1.70"]
r = [.817, .777, .590]
mae = [1.300, 1.649, 1.696]
count = [230, 68, 100]
yc = np.arange(3)[::-1]
ax.barh(yc, r, color=["#258f9d", "#7dbcc4", "#e6a34c"], height=.53)
for yi, rv in zip(yc, r):
    ax.text(rv-.015, yi, f"r={rv:.3f}", ha="right", va="center", fontsize=9,
            color="white", weight="bold")
ax.set_yticks(yc, vendors)
ax.set_xlim(0, 1.0)
ax.set_xlabel("Pearson correlation (r)")
ax.set_title("Same-visit Mid-GLS by vendor", weight="bold")
ax.grid(axis="x", color="#e2e8f0")
ax.set_axisbelow(True)
fig.suptitle("What the video encoder learned about current physiology", fontsize=15, weight="bold")
fig.text(.5, .028,
         "Patient-held-out predictions. Vendor differences are descriptive and may reflect patient/scanner mix.",
         ha="center", fontsize=9, color="#52606d")
fig.subplots_adjust(left=.09, right=.98, top=.80, bottom=.20)
fig.savefig(OUT / "gls_results.png", dpi=180)
plt.close(fig)
