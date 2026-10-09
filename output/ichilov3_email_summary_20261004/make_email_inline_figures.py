"""Reproducible figures for the October 2026 supervisor email.

Values are transcribed from email_draft.md and make_figures.py. These are
descriptive, patient-held-out comparisons; the post-hoc view selection is
explicitly marked as exploratory.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
NAVY = "#173c70"
TEAL = "#278d99"
AMBER = "#d99535"
GRAY = "#8da5b6"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "savefig.facecolor": "white",
})


def finish(fig, name, note):
    fig.text(.5, .025, note, ha="center", color="#52606d", fontsize=8.5)
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bars(ax, labels, values, colors, title, xlabel, lim=None, digits=0):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=.58)
    ax.set_yticks(y, labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xlim(*(lim or (0, max(values) * 1.18)))
    ax.grid(axis="x", color="#e7edf2")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for yy, value in zip(y, values):
        ax.text(value + (ax.get_xlim()[1] - ax.get_xlim()[0]) * .012, yy,
                f"{value:.{digits}f}", va="center", fontsize=9)


def metric_pair(axs, labels, auc, ap, colors, title, note, name,
                auc_range=(0, .8), ap_range=(0, .45)):
    for ax, values, metric, lim in zip(axs, [auc, ap], ["AUROC", "Average precision (AP)"],
                                        [auc_range, ap_range]):
        bars(ax, labels, values, colors, metric, metric, lim=lim, digits=3)
    axs[1].set_yticklabels([])
    axs[1].tick_params(axis="y", left=False)
    axs[0].figure.suptitle(title, weight="bold", fontsize=14)
    axs[0].figure.subplots_adjust(left=.30, right=.96, top=.81, bottom=.22, wspace=.19)
    finish(axs[0].figure, name, note)


# Inventory: counts refer to distinct visits, except the note on report files.
fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.15))
bars(axs[0], ["Strain-report visits", "DICOM visits", "Exact matches", "3-view sets"],
     [400, 413, 398, 411], [NAVY, GRAY, TEAL, TEAL],
     "Dataset alignment", "Visits", lim=(0, 470))
bars(axs[1], ["GE Healthcare", "GE Vingmed", "Philips"],
     [239, 70, 104], [TEAL, "#79b9c2", AMBER],
     "DICOM visits by manufacturer", "Visits", lim=(0, 280))
fig.suptitle("Connecting reports to the selected echo cines", weight="bold", fontsize=14)
fig.subplots_adjust(left=.19, right=.97, top=.80, bottom=.22, wspace=.52)
finish(fig, "email_01_inventory.png",
       "416 report files represent 400 visits; 3 clips per selected visit = 1,233 cines. Two report visits lack matching 3-view sets.")


# Full cohort: 238 transitions, 49 events.
fig, axs = plt.subplots(1, 2, figsize=(11.8, 3.85))
metric_pair(axs,
    ["Strain + clinical", "+ 25% EchoPrime"],
    [.698, .694], [.353, .330], [NAVY, TEAL],
    "Next-visit deterioration · full cohort",
    "238 transitions; 49 events. Same patient-held-out evaluation; fixed 25% late fusion.",
    "email_02_full_cohort.png", auc_range=(0, .8), ap_range=(0, .43))


# Time sampling: GLS same-visit and downstream deterioration are different outcomes.
fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.2))
labels = ["EchoPrime", "PanEcho"]
x = np.arange(2)
w = .33
old = [1.607, 2.025]
new = [1.459, 1.603]
axs[0].bar(x-w/2, old, w, color=GRAY, label="Original")
axs[0].bar(x+w/2, new, w, color=TEAL, label="Estimated heartbeat")
for xx, a, b in zip(x, old, new):
    axs[0].text(xx-w/2, a+.04, f"{a:.3f}", ha="center", fontsize=9)
    axs[0].text(xx+w/2, b+.04, f"{b:.3f}", ha="center", fontsize=9)
axs[0].set_xticks(x, labels)
axs[0].set_ylim(0, 2.4)
axs[0].set_title("Same-visit Mid-GLS")
axs[0].set_ylabel("MAE (GLS points; lower is better)")
axs[0].legend(frameon=False, fontsize=8)
bars(axs[1], ["Original", "Heartbeat"], [.694, .690], [GRAY, TEAL],
     "Next-visit AUROC", "AUROC", lim=(0, .8), digits=3)
bars(axs[2], ["Original", "Heartbeat"], [.330, .333], [GRAY, TEAL],
     "Next-visit AP", "AP", lim=(0, .4), digits=3)
fig.suptitle("Estimated heartbeat sampling: clearer current GLS, no future gain",
             weight="bold", fontsize=14)
fig.subplots_adjust(left=.075, right=.975, top=.81, bottom=.23, wspace=.34)
finish(fig, "email_03_sampling.png",
       "GLS: 398 matched visits. Deterioration: full cohort, EchoPrime 25% fusion. Outcomes and scales differ across panels.")


# Vendor contrast for same-visit GLS prediction.
fig, axs = plt.subplots(1, 2, figsize=(12, 3.95))
vendor = ["GE Healthcare (n=230)", "GE Vingmed (n=68)", "Philips (n=100)"]
vendor_colors = [TEAL, "#79b9c2", AMBER]
bars(axs[0], vendor, [.817, .777, .590], vendor_colors,
     "Pearson correlation", "r (higher is better)", lim=(0, 1), digits=3)
bars(axs[1], vendor, [1.300, 1.649, 1.696], vendor_colors,
     "Absolute error", "MAE (GLS points; lower is better)", lim=(0, 2.1), digits=2)
fig.suptitle("Same-visit Mid-GLS prediction by manufacturer", weight="bold", fontsize=14)
fig.subplots_adjust(left=.25, right=.97, top=.80, bottom=.22, wspace=.50)
finish(fig, "email_04_vendors.png",
       "398 matched visits. Descriptive subgroup results; vendor differences may reflect scanner and patient mix.")


# All following next-visit charts use the non-Philips subset: 182 transitions.
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.1))
metric_pair(axs,
    ["Strain + clinical", "Frozen EchoPrime", "Frozen PanEcho", "Frozen EchoFM", "Frozen EchoJEPA"],
    [.726, .701, .722, .713, .720], [.379, .347, .378, .387, .368],
    [NAVY, TEAL, TEAL, TEAL, TEAL],
    "Frozen video encoders · non-Philips visits",
    "182 transitions; 86 patients; 38 events. Fixed 25% video-score fusion; patient-held-out folds.",
    "email_05_frozen_encoders.png", auc_range=(0, .83), ap_range=(0, .46))


fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
metric_pair(axs,
    ["Strain + clinical", "Frozen EchoPrime", "Adapted: all views", "Adapted: view-specific"],
    [.726, .701, .710, .727], [.379, .347, .360, .355],
    [NAVY, GRAY, TEAL, TEAL],
    "GE-adapted EchoPrime · non-Philips visits",
    "182 transitions. Adapted models include video history when available; old GE data trained the encoder only.",
    "email_06_finetuning.png", auc_range=(0, .83), ap_range=(0, .46))


fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
metric_pair(axs,
    ["Strain + clinical", "A3C+A4C (post hoc)", "Views chosen in training"],
    [.726, .736, .707], [.379, .355, .345],
    [NAVY, AMBER, TEAL],
    "View selection · non-Philips visits",
    "182 transitions. Post-hoc A3C+A4C was selected after seeing held-out results; it is not a valid improvement estimate.",
    "email_07_view_selection.png", auc_range=(0, .83), ap_range=(0, .46))
