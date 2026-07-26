#!/usr/bin/env python3
"""Plot patient-held-out ROC curves for the primary cardiotoxicity task."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve


ROOT = Path(__file__).resolve().parent
NEXT_VISIT_PREDICTIONS = (
    ROOT / "cardiotoxicity_next_visit_gpu_results" / "oof_predictions.parquet"
)
PLATEAU_PREDICTIONS = (
    ROOT / "cardiotoxicity_plateau_results" / "plateau_oof_predictions.parquet"
)
OUTPUT_DIR = ROOT / "cardiotoxicity_plateau_results" / "figures"
TASK = "mid_first_rel15"

MODEL_SPECS = [
    ("Clinical ridge", "next", "clinical_ridge", "#0072B2", "-"),
    (
        "Clinical + Endo–Mid",
        "next",
        "clinical_plus_transmural_sparse",
        "#D55E00",
        "--",
    ),
    (
        "Clinical + variability",
        "next",
        "clinical_plus_variability_sparse",
        "#009E73",
        ":",
    ),
    ("Combined Extra Trees", "next", "combined_extra_trees", "#CC79A7", "-."),
    ("Uniform CNN", "plateau", "uniform_binary", "#7B61FF", (0, (2, 2))),
    ("Attention CNN", "plateau", "attention_binary", "#2CB1A6", "-"),
]


def load_model_predictions(
    frame: pd.DataFrame, model: str
) -> pd.DataFrame:
    selected = frame.loc[
        frame["task"].eq(TASK) & frame["model"].eq(model),
        ["transition_id", "patient_id", "label", "score"],
    ].copy()
    selected = selected.sort_values("transition_id").reset_index(drop=True)
    if len(selected) != 238 or int(selected["label"].sum()) != 49:
        raise ValueError(
            f"Unexpected primary cohort for {model}: "
            f"n={len(selected)}, events={int(selected['label'].sum())}"
        )
    return selected


def main() -> None:
    next_visit = pd.read_parquet(NEXT_VISIT_PREDICTIONS)
    plateau = pd.read_parquet(PLATEAU_PREDICTIONS)
    sources = {"next": next_visit, "plateau": plateau}

    curves: list[tuple[str, float, list[float], list[float], str, object]] = []
    reference_ids: list[str] | None = None
    reference_labels: list[int] | None = None

    for display_name, source, model, color, linestyle in MODEL_SPECS:
        selected = load_model_predictions(sources[source], model)
        ids = selected["transition_id"].tolist()
        labels = selected["label"].astype(int).tolist()
        if reference_ids is None:
            reference_ids = ids
            reference_labels = labels
        elif ids != reference_ids or labels != reference_labels:
            raise ValueError(f"Prediction cohort mismatch for {model}")

        false_positive_rate, true_positive_rate, _ = roc_curve(
            selected["label"], selected["score"]
        )
        roc_auc = auc(false_positive_rate, true_positive_rate)
        curves.append(
            (
                display_name,
                roc_auc,
                false_positive_rate,
                true_positive_rate,
                color,
                linestyle,
            )
        )

    figure, axis = plt.subplots(figsize=(9.2, 7.0))
    axis.plot(
        [0, 1],
        [0, 1],
        color="#777777",
        linewidth=1.4,
        linestyle=(0, (5, 4)),
        label="Random guess — AUC 0.500",
        zorder=1,
    )

    for display_name, roc_auc, fpr, tpr, color, linestyle in curves:
        linewidth = 2.8 if display_name == "Attention CNN" else 2.0
        axis.plot(
            fpr,
            tpr,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=f"{display_name} — AUC {roc_auc:.3f}",
            zorder=3 if display_name == "Attention CNN" else 2,
        )

    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC curves: next-visit 15% Mid-GLS deterioration",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.02,
        0.98,
        "First-visit baseline · n=238 transitions · 49 events",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        color="#555555",
    )
    axis.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.95,
        fontsize=9,
    )
    figure.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_DIR / "roc_model_comparison.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
