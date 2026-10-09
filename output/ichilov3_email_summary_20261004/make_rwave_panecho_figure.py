"""Update the email comparison with the completed R-wave PanEcho experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
names = [
    "Strain + clinical baseline",
    "Frozen EchoPrime",
    "Frozen PanEcho · estimated beat",
    "Frozen EchoFM",
    "GE-adapted EchoPrime · view-specific + history",
    "Frozen EchoJEPA",
    "Frozen PanEcho · R-wave cycle",
    "Adapted PanEcho · R-wave cycle",
]
auc = np.array([.726, .701, .722, .713, .727, .720, .705, .706])
ap = np.array([.379, .347, .378, .387, .355, .368, .349, .356])
colors = ["#143c72"] + ["#288f9e"] * 5 + ["#e09c48", "#c55f44"]
y = np.arange(len(names))[::-1]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white"})
fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.1), sharey=True,
                         gridspec_kw={"wspace": .08})
for ax, values, xlabel, limits, reference in zip(
    axes, [auc, ap], ["AUROC", "Average precision (AP)"],
    [(0.65, .765), (.30, .415)], [.726, .379]
):
    ax.axvline(reference, color="#143c72", linestyle="--", lw=1.3, alpha=.65)
    for yi, value, color in zip(y, values, colors):
        ax.plot([limits[0], value], [yi, yi], color=color, alpha=.25, lw=2)
        ax.scatter(value, yi, s=65, color=color, zorder=3)
        ax.annotate(f"{value:.3f}", (value, yi), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_xlim(*limits)
    ax.set_xlabel(xlabel, weight="bold")
    ax.grid(axis="x", color="#e2e8f0", zorder=0)
    ax.set_axisbelow(True)
axes[0].set_yticks(y, names)
axes[0].tick_params(axis="y", length=0)
axes[1].tick_params(axis="y", left=False)
fig.suptitle("Prediction of deterioration at the next visit", fontsize=15, weight="bold")
fig.text(.5, .045,
         "Same 182 GE/non-Philips transitions; 3 × 5 patient-held-out folds. Dashed line = strain/clinical baseline.\n"
         "Fixed 25% video-score fusion. PanEcho R-wave rows use the first complete R–R cycle; others use earlier sampling.",
         ha="center", fontsize=9, color="#52606d")
fig.subplots_adjust(left=.42, right=.96, top=.88, bottom=.18)
fig.savefig(OUT / "next_visit_results_with_rwave_panecho.png", dpi=180)
plt.close(fig)
