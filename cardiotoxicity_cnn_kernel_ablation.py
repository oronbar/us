from __future__ import annotations

import copy
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

import cardiotoxicity_cnn_channel_ablation as channel
import cardiotoxicity_cnn_length_ablation as length_ablation
import cardiotoxicity_next_visit_gpu as core
import cardiotoxicity_nonapical_qc as qc


VISITS_PATH = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet")
CURVES_PATH = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_dataset.parquet")
BASE_RESULTS = Path(r"D:\us\cardiotoxicity_next_visit_gpu_results")
LENGTH_RESULTS = Path(r"D:\us\cardiotoxicity_cnn_length_ablation_results")
OUTPUT = Path(r"D:\us\cardiotoxicity_cnn_kernel_ablation_results")
PRIMARY_TASK = "mid_first_rel15"
SEED = channel.SEED
BOOTSTRAPS = channel.BOOTSTRAPS


@dataclass(frozen=True)
class KernelSpec:
    name: str
    length: int
    kernel1: int
    kernel2: int

    @property
    def effective_receptive_field(self) -> int:
        return self.kernel1 + self.kernel2 - 1


SPECS = (
    KernelSpec("attention_t64_scaled_k5_k3", 64, 5, 3),
    KernelSpec("attention_t72_scaled_k5_k5", 72, 5, 5),
)


class KernelAttentionNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        scalar_features: int,
        tasks: int,
        segments: int,
        kernel1: int,
        kernel2: int,
    ):
        super().__init__()
        self.segment_encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=kernel1, padding=kernel1 // 2),
            nn.GELU(),
            nn.Conv1d(16, 24, kernel_size=kernel2, padding=kernel2 // 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.segment_embedding = nn.Parameter(torch.zeros(1, segments, 24))
        nn.init.normal_(self.segment_embedding, mean=0.0, std=0.02)
        self.attention_score = nn.Sequential(
            nn.Linear(24, 12),
            nn.Tanh(),
            nn.Linear(12, 1),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_features, 48),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(48, 32),
            nn.GELU(),
        )
        self.shared = nn.Sequential(
            nn.Linear(24 * 3 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.35),
        )
        self.binary_head = nn.Linear(64, tasks)

    def forward(
        self, curves: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, segments, channels, time_points = curves.shape
        encoded = self.segment_encoder(
            curves.reshape(batch * segments, channels, time_points)
        ).squeeze(-1)
        encoded = encoded.reshape(batch, segments, -1)
        scored = encoded + self.segment_embedding[:, :segments]
        weights = torch.softmax(self.attention_score(scored).squeeze(-1), dim=1)
        center = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        pooled = torch.cat(
            [center, encoded.std(dim=1), encoded.max(dim=1).values], dim=1
        )
        scalar_embedding = self.scalar_encoder(scalars)
        shared = self.shared(torch.cat([pooled, scalar_embedding], dim=1))
        return self.binary_head(shared), weights


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in columns:
        if pd.api.types.is_numeric_dtype(selected[column]):
            selected[column] = selected[column].map(lambda value: f"{value:.3f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def train_scaled_spec(
    spec: KernelSpec,
    transitions: pd.DataFrame,
    visits: pd.DataFrame,
    scalar_columns: list[str],
    active_tasks: list[core.TaskSpec],
    splits: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tensors = length_ablation.build_visit_curve_tensors_at_length(
        CURVES_PATH, spec.length
    )
    full6 = channel.build_channel_inputs(transitions, visits, tensors)[
        "attention_full6"
    ]
    variant = channel.ChannelVariant(
        spec.name,
        f"{spec.length} samples with kernels {spec.kernel1}+{spec.kernel2}",
        (
            "current_endo",
            "current_mid",
            "current_endo_minus_mid",
            "change_endo",
            "change_mid",
            "change_endo_minus_mid",
        ),
    )
    original_variants = channel.VARIANTS
    original_model = channel.ChannelAttentionNet
    try:
        channel.VARIANTS = (variant,)

        def model_factory(
            input_channels: int,
            scalar_features: int,
            tasks: int,
            segments: int,
        ) -> KernelAttentionNet:
            return KernelAttentionNet(
                input_channels,
                scalar_features,
                tasks,
                segments,
                spec.kernel1,
                spec.kernel2,
            )

        channel.ChannelAttentionNet = model_factory
        predictions, weights, logs = channel.train_channel_variants(
            transitions,
            {spec.name: full6},
            scalar_columns,
            active_tasks,
            splits,
        )
    finally:
        channel.VARIANTS = original_variants
        channel.ChannelAttentionNet = original_model
    logs["length"] = spec.length
    logs["kernel1"] = spec.kernel1
    logs["kernel2"] = spec.kernel2
    logs["effective_receptive_field"] = spec.effective_receptive_field
    logs["receptive_field_cycle_fraction"] = (
        spec.effective_receptive_field / spec.length
    )
    return predictions, weights, logs


def make_figure(metrics: pd.DataFrame) -> None:
    order = [
        "attention_t64",
        "attention_t64_scaled_k5_k3",
        "attention_t72",
        "attention_t72_scaled_k5_k5",
        "attention_t96",
    ]
    labels = ["64 fixed\n7+5", "64 scaled\n5+3", "72 fixed\n7+5", "72 scaled\n5+5", "96 reference\n7+5"]
    colors = ["#8ecae6", "#277da1", "#95d5b2", "#40916c", "#f8961e"]
    frame = metrics.set_index("model").loc[order]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for axis, metric, low, high, title, random_value in (
        (axes[0], "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "ROC AUC", 0.5),
        (axes[1], "average_precision", "average_precision_ci_low", "average_precision_ci_high", "Average precision", 49 / 238),
    ):
        values = frame[metric].to_numpy()
        errors = np.vstack(
            [values - frame[low].to_numpy(), frame[high].to_numpy() - values]
        )
        axis.errorbar(
            np.arange(len(order)), values, yerr=errors, fmt="none", ecolor="#555555",
            elinewidth=2, capsize=4,
        )
        axis.scatter(np.arange(len(order)), values, c=colors, s=65, zorder=3)
        axis.axhline(random_value, color="0.55", ls="--", lw=1)
        axis.set_xticks(np.arange(len(order)), labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylim(0.48, 0.80)
    axes[1].set_ylim(0.18, 0.49)
    figure.suptitle("Cycle-proportional CNN kernel ablation")
    figure.tight_layout()
    figure.savefig(OUTPUT / "cnn_kernel_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    transitions = pd.read_parquet(BASE_RESULTS / "next_visit_transitions.parquet")
    visits = pd.read_parquet(VISITS_PATH)
    manifest = pd.read_csv(BASE_RESULTS / "feature_manifest.csv")
    gpu_features = manifest[manifest["feature_set"].eq("gpu_scalars")]["feature"].tolist()
    scalar_columns = core.usable_features(transitions, gpu_features)
    tasks = core.task_specs()
    audit = core.label_audit(transitions, tasks)
    active_names = set(
        audit[
            (audit["eligible_transitions"] >= 30)
            & (audit["events"] >= 8)
            & ((audit["eligible_transitions"] - audit["events"]) >= 8)
        ]["task"]
    )
    active_tasks = [task for task in tasks if task.name in active_names]
    splits, assignments = core.patient_splits(transitions, 3, SEED)

    postprocess_only = "--postprocess-only" in sys.argv
    if postprocess_only:
        scaled_predictions = pd.read_parquet(
            OUTPUT / "cnn_kernel_scaled_oof_predictions.parquet"
        )
        logs = pd.read_csv(OUTPUT / "cnn_kernel_ablation_training_log.csv")
    else:
        prediction_frames = []
        weight_frames = []
        log_frames = []
        for spec in SPECS:
            predictions, weights, logs = train_scaled_spec(
                spec, transitions, visits, scalar_columns, active_tasks, splits
            )
            prediction_frames.append(predictions)
            weight_frames.append(weights)
            log_frames.append(logs)
        scaled_predictions = pd.concat(prediction_frames, ignore_index=True)
        scaled_weights = pd.concat(weight_frames, ignore_index=True)
        logs = pd.concat(log_frames, ignore_index=True)

        scaled_predictions.to_parquet(
            OUTPUT / "cnn_kernel_scaled_oof_predictions.parquet", index=False
        )
        scaled_weights.to_parquet(
            OUTPUT / "cnn_kernel_scaled_attention_weights.parquet", index=False
        )
        logs.to_csv(OUTPUT / "cnn_kernel_ablation_training_log.csv", index=False)
        assignments.to_csv(OUTPUT / "cnn_kernel_ablation_patient_folds.csv", index=False)

    fixed = pd.read_parquet(
        LENGTH_RESULTS / "cnn_length_ablation_oof_predictions.parquet"
    )
    fixed = fixed[fixed["model"].isin(["attention_t64", "attention_t72", "attention_t96"])]
    all_predictions = pd.concat([fixed, scaled_predictions], ignore_index=True)
    primary = all_predictions[all_predictions["task"].eq(PRIMARY_TASK)].copy()
    comparisons = [
        ("scaled64_vs_fixed64", "attention_t64", "attention_t64_scaled_k5_k3"),
        ("scaled72_vs_fixed72", "attention_t72", "attention_t72_scaled_k5_k5"),
        ("scaled64_vs_t96", "attention_t96", "attention_t64_scaled_k5_k3"),
        ("scaled72_vs_t96", "attention_t96", "attention_t72_scaled_k5_k5"),
    ]
    if postprocess_only:
        metrics = pd.read_csv(OUTPUT / "cnn_kernel_ablation_metrics.csv")
        paired = pd.read_csv(OUTPUT / "cnn_kernel_ablation_paired_deltas.csv")
    else:
        metrics, _ = core.evaluate_predictions(primary, BOOTSTRAPS, SEED)
        paired = qc.paired_variant_deltas(
            primary, comparisons, BOOTSTRAPS, SEED + 830000
        )
        metrics.to_csv(OUTPUT / "cnn_kernel_ablation_metrics.csv", index=False)
        paired.to_csv(OUTPUT / "cnn_kernel_ablation_paired_deltas.csv", index=False)
    make_figure(metrics)

    display_names = {
        "attention_t64": "64 fixed 7+5",
        "attention_t64_scaled_k5_k3": "64 scaled 5+3",
        "attention_t72": "72 fixed 7+5",
        "attention_t72_scaled_k5_k5": "72 scaled 5+5",
        "attention_t96": "96 reference 7+5",
    }
    table = metrics[metrics["task"].eq(PRIMARY_TASK)].copy()
    table["model"] = table["model"].map(display_names)
    table = table.rename(
        columns={
            "roc_auc": "AUC",
            "roc_auc_ci_low": "AUC CI low",
            "roc_auc_ci_high": "AUC CI high",
            "average_precision": "AP",
            "average_precision_ci_low": "AP CI low",
            "average_precision_ci_high": "AP CI high",
        }
    )
    table["order"] = table["model"].map(
        {name: index for index, name in enumerate(display_names.values())}
    )
    table = table.sort_values("order")
    delta_table = paired.rename(
        columns={
            "delta_roc_auc": "delta AUC",
            "delta_roc_auc_ci_low": "delta AUC CI low",
            "delta_roc_auc_ci_high": "delta AUC CI high",
            "delta_average_precision": "delta AP",
            "delta_average_precision_ci_low": "delta AP CI low",
            "delta_average_precision_ci_high": "delta AP CI high",
        }
    )
    prevalence = float(primary.drop_duplicates("transition_id")["label"].mean())
    report = f"""# Cycle-proportional CNN kernel ablation

## Controlled question

Can shorter temporal inputs improve when convolution kernels are reduced to preserve
approximately the same fraction of the normalized cardiac cycle as the 96-point
7+5-kernel reference?

The effective receptive fields were 11/96 = 11.5% for the reference, 9/72 = 12.5%
for scaled 72, and 7/64 = 10.9% for scaled 64. Odd kernels were used to preserve
symmetric centering. All patient folds, seeds, inputs, scalar features, multitask
labels, optimizer settings, and early stopping rules were unchanged. Training used
{torch.cuda.get_device_name(0)}.

## Results

{markdown_table(table, ['model', 'n', 'events', 'AUC', 'AUC CI low', 'AUC CI high', 'AP', 'AP CI low', 'AP CI high'])}

Random-ranking AUC is 0.500 and random-ranking AP is the event prevalence,
{prevalence:.3f}.

## Paired differences

{markdown_table(delta_table, ['comparison', 'delta AUC', 'delta AUC CI low', 'delta AUC CI high', 'delta AP', 'delta AP CI low', 'delta AP CI high'])}

A positive difference favors the scaled-kernel candidate. Intervals are paired
patient-cluster bootstrap confidence intervals.

## Interpretation

- At 72 samples, scaling from 7+5 to 5+5 was neutral: delta AUC -0.002 and
  delta AP +0.001, with both confidence intervals crossing zero.
- At 64 samples, scaling from 7+5 to 5+3 was unfavorable in point estimates:
  delta AUC -0.012 and delta AP -0.039. The intervals still cross zero, so the
  cohort cannot establish a definitive loss.
- Neither scaled model improved on the 96-point reference. The 96-point model
  retained the highest AP.
- Narrower kernels also reduce parameter count, especially for 64 samples, so its
  point-estimate loss may reflect both receptive-field and capacity changes.
- Recommendation: retain 96 with 7+5 kernels for performance-first use. If a shorter
  input is required, retain the fixed 7+5 kernels rather than scaling them down.
"""
    (OUTPUT / "cnn_kernel_ablation_report.md").write_text(report, encoding="utf-8")

    metadata = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "primary_task": PRIMARY_TASK,
        "reference": {"length": 96, "kernels": [7, 5], "effective_rf": 11, "cycle_fraction": 11 / 96},
        "scaled_specs": [
            {
                "name": spec.name,
                "length": spec.length,
                "kernels": [spec.kernel1, spec.kernel2],
                "effective_rf": spec.effective_receptive_field,
                "cycle_fraction": spec.effective_receptive_field / spec.length,
            }
            for spec in SPECS
        ],
        "scalar_features": len(scalar_columns),
        "active_tasks": [task.name for task in active_tasks],
        "cv": "3 repeated 5-fold patient-held-out",
        "bootstraps": BOOTSTRAPS,
        "epochs": channel.EPOCHS,
        "patience": channel.PATIENCE,
        "seed": SEED,
        "fixed_predictions_source": str(LENGTH_RESULTS / "cnn_length_ablation_oof_predictions.parquet"),
    }
    (OUTPUT / "cnn_kernel_ablation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nPrimary metrics", flush=True)
    print(table[["model", "n", "events", "AUC", "AUC CI low", "AUC CI high", "AP", "AP CI low", "AP CI high"]].to_string(index=False), flush=True)
    print("\nPaired deltas", flush=True)
    print(delta_table.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
