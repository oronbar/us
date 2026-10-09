"""Plot paired patient-held-out ROC curves for the view-fusion comparison."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


ROOT = Path(r"D:\DS\ichilov3_view_fusion_20261004")
PREDICTIONS = ROOT / "oof_predictions.parquet"
SERIES = (
    ("strain_clinical_reference", "Strain + clinical baseline", "#334155", "--"),
    ("adapted_specific_A2C_A3C_A4C_late25", "EchoPrime: A2C + A3C + A4C", "#2563eb", "-"),
    ("adapted_specific_A3C_A4C_late25", "EchoPrime: A3C + A4C", "#e7682a", "-"),
)


def main() -> None:
    data = pd.read_parquet(PREDICTIONS)
    keys = ["transition_id", "patient_id", "label"]
    indexed = {}
    for name, _, _, _ in SERIES:
        selected = data[data.model.eq(name)].set_index(keys).sort_index()
        if selected.index.has_duplicates:
            raise ValueError(f"Duplicate held-out prediction: {name}")
        indexed[name] = selected
    first = indexed[SERIES[0][0]]
    if any(not first.index.equals(frame.index) for frame in indexed.values()):
        raise ValueError("ROC arms do not cover identical transitions")
    if len(first) != 182 or first.index.get_level_values("patient_id").nunique() != 86:
        raise ValueError("Unexpected evaluation cohort")
    y = first.index.get_level_values("label").to_numpy(int)
    if int(y.sum()) != 38:
        raise ValueError("Unexpected event count")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.2, 7.0), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfe")
    ax.plot([0, 1], [0, 1], color="#b5becb", lw=1.2, ls=":", label="Chance (AUC 0.500)")
    for name, label, color, style in SERIES:
        score = indexed[name]["score"].to_numpy(float)
        if not np.isfinite(score).all():
            raise ValueError(f"Nonfinite score: {name}")
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        ax.plot(fpr, tpr, color=color, lw=2.6, ls=style,
                label=f"{label}  (AUC {auc:.3f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#e5eaf0", lw=.7)
    ax.set_xlabel("False positive rate  (1 − specificity)", fontsize=11)
    ax.set_ylabel("True positive rate  (sensitivity)", fontsize=11)
    fig.suptitle("Next-visit Mid-GLS deterioration: ROC", x=.11, y=.965,
                 ha="left", fontsize=15, fontweight="bold", color="#152033")
    fig.text(.11, .922, "Patient-held-out predictions · 182 transitions · 86 patients · 38 events",
             fontsize=10, color="#526276")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#dbe2ea",
              fontsize=9.5)
    fig.text(.11, .052,
             "A3C+A4C was identified in an exploratory view comparison. Its paired AUC gain over all three views\n"
             "was +0.009 (95% patient-bootstrap interval −0.014 to +0.032).",
             fontsize=9, color="#526276")
    fig.subplots_adjust(left=.115, right=.96, bottom=.16, top=.87)
    for suffix in ("png", "pdf"):
        path = ROOT / f"roc_adapted_a3c_a4c_vs_baseline.{suffix}"
        fig.savefig(path, dpi=230, facecolor="white", bbox_inches="tight")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
