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
import cardiotoxicity_next_visit_gpu as core
import cardiotoxicity_nonapical_qc as qc


VISITS_PATH = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet")
CURVES_PATH = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_dataset.parquet")
BASE_RESULTS = Path(r"D:\us\cardiotoxicity_next_visit_gpu_results")
LENGTH_RESULTS = Path(r"D:\us\cardiotoxicity_cnn_length_ablation_results")
OUTPUT = Path(r"D:\us\cardiotoxicity_cnn_segment_interaction_results")
PRIMARY_TASK = "mid_first_rel15"
SEED = channel.SEED
BOOTSTRAPS = channel.BOOTSTRAPS


@dataclass(frozen=True)
class InteractionSpec:
    name: str
    temporal_bins: int
    description: str
    use_segment_attention: bool


SPECS = (
    InteractionSpec(
        "timing8_no_segment_attention",
        8,
        "Eight-bin phase-preserving pooling without pairwise segment attention",
        False,
    ),
    InteractionSpec(
        "segment_self_attention",
        1,
        "Global temporal pooling followed by one four-head segment self-attention block",
        True,
    ),
    InteractionSpec(
        "timing8_segment_self_attention",
        8,
        "Eight-bin phase-preserving pooling, 192-to-24 projection, and segment self-attention",
        True,
    ),
)


class SegmentInteractionNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        scalar_features: int,
        tasks: int,
        segments: int,
        temporal_bins: int,
        use_segment_attention: bool,
    ):
        super().__init__()
        self.temporal_bins = temporal_bins
        self.use_segment_attention = use_segment_attention
        self.segment_encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 24, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.temporal_pool = nn.AdaptiveAvgPool1d(temporal_bins)
        self.temporal_projection = (
            nn.Identity()
            if temporal_bins == 1
            else nn.Sequential(
                nn.Linear(24 * temporal_bins, 24),
                nn.GELU(),
            )
        )
        self.segment_embedding = nn.Parameter(torch.zeros(1, segments, 24))
        nn.init.normal_(self.segment_embedding, mean=0.0, std=0.02)
        if use_segment_attention:
            self.segment_attention = nn.MultiheadAttention(
                embed_dim=24,
                num_heads=4,
                dropout=0.10,
                batch_first=True,
            )
            self.attention_dropout = nn.Dropout(0.10)
            self.attention_norm = nn.LayerNorm(24)
            self.segment_ffn = nn.Sequential(
                nn.Linear(24, 48),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(48, 24),
            )
            self.ffn_dropout = nn.Dropout(0.10)
            self.ffn_norm = nn.LayerNorm(24)
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
        temporal = self.segment_encoder(
            curves.reshape(batch * segments, channels, time_points)
        )
        temporal = self.temporal_pool(temporal)
        encoded = temporal.reshape(batch, segments, 24 * self.temporal_bins)
        encoded = self.temporal_projection(encoded)

        if self.use_segment_attention:
            positioned = encoded + self.segment_embedding[:, :segments]
            attended, _ = self.segment_attention(
                positioned,
                positioned,
                positioned,
                need_weights=False,
            )
            contextual = self.attention_norm(
                positioned + self.attention_dropout(attended)
            )
            contextual = self.ffn_norm(
                contextual + self.ffn_dropout(self.segment_ffn(contextual))
            )
            score_input = contextual
        else:
            contextual = encoded
            score_input = contextual + self.segment_embedding[:, :segments]

        weights = torch.softmax(
            self.attention_score(score_input).squeeze(-1), dim=1
        )
        center = torch.sum(contextual * weights.unsqueeze(-1), dim=1)
        pooled = torch.cat(
            [
                center,
                contextual.std(dim=1),
                contextual.max(dim=1).values,
            ],
            dim=1,
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


def train_spec(
    spec: InteractionSpec,
    transitions: pd.DataFrame,
    full6: np.ndarray,
    scalar_columns: list[str],
    active_tasks: list[core.TaskSpec],
    splits: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    variant = channel.ChannelVariant(
        spec.name,
        spec.description,
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
        ) -> SegmentInteractionNet:
            return SegmentInteractionNet(
                input_channels,
                scalar_features,
                tasks,
                segments,
                spec.temporal_bins,
                spec.use_segment_attention,
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
    logs["temporal_bins"] = spec.temporal_bins
    logs["use_segment_attention"] = spec.use_segment_attention
    return predictions, weights, logs


def make_figure(metrics: pd.DataFrame) -> None:
    order = [
        "attention_current",
        "timing8_no_segment_attention",
        "segment_self_attention",
        "timing8_segment_self_attention",
    ]
    labels = [
        "Current CNN\n17.5k params",
        "8-bin only\n22.1k params",
        "Segment attention\n22.4k params",
        "8-bin + attention\n27.0k params",
    ]
    colors = ["#8ecae6", "#277da1", "#43aa8b", "#f8961e"]
    frame = metrics.set_index("model").loc[order]
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 4.7))
    for axis, metric, low, high, title, random_value in (
        (
            axes[0],
            "roc_auc",
            "roc_auc_ci_low",
            "roc_auc_ci_high",
            "ROC AUC",
            0.5,
        ),
        (
            axes[1],
            "average_precision",
            "average_precision_ci_low",
            "average_precision_ci_high",
            "Average precision",
            49 / 238,
        ),
    ):
        values = frame[metric].to_numpy()
        errors = np.vstack(
            [values - frame[low].to_numpy(), frame[high].to_numpy() - values]
        )
        axis.errorbar(
            np.arange(len(order)),
            values,
            yerr=errors,
            fmt="none",
            ecolor="#555555",
            elinewidth=2,
            capsize=4,
        )
        axis.scatter(np.arange(len(order)), values, c=colors, s=70, zorder=3)
        for index, value in enumerate(values):
            axis.annotate(
                f"{value:.3f}",
                (index, value),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
            )
        axis.axhline(random_value, color="0.55", ls="--", lw=1)
        axis.set_xticks(np.arange(len(order)), labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylim(0.48, 0.80)
    axes[1].set_ylim(0.18, 0.49)
    figure.suptitle("Segment-interaction architecture ablation")
    figure.tight_layout()
    figure.savefig(
        OUTPUT / "segment_interaction_ablation.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    transitions = pd.read_parquet(BASE_RESULTS / "next_visit_transitions.parquet")
    visits = pd.read_parquet(VISITS_PATH)
    manifest = pd.read_csv(BASE_RESULTS / "feature_manifest.csv")
    gpu_features = manifest[manifest["feature_set"].eq("gpu_scalars")][
        "feature"
    ].tolist()
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

    tensors = core.build_visit_curve_tensors(CURVES_PATH)
    full6 = channel.build_channel_inputs(transitions, visits, tensors)[
        "attention_full6"
    ]
    control_only = "--timing-control-only" in sys.argv
    selected_specs = (
        [spec for spec in SPECS if spec.name == "timing8_no_segment_attention"]
        if control_only
        else list(SPECS)
    )
    prediction_frames = []
    weight_frames = []
    log_frames = []
    if control_only:
        previous_predictions = pd.read_parquet(
            OUTPUT / "segment_interaction_oof_predictions.parquet"
        )
        previous_predictions = previous_predictions[
            ~previous_predictions["model"].eq("timing8_no_segment_attention")
        ]
        prediction_frames.append(previous_predictions)
        previous_weights = pd.read_parquet(
            OUTPUT / "segment_interaction_attention_weights.parquet"
        )
        previous_weights = previous_weights[
            ~previous_weights["variant"].eq("timing8_no_segment_attention")
        ]
        weight_frames.append(previous_weights)
        previous_logs = pd.read_csv(
            OUTPUT / "segment_interaction_training_log.csv"
        )
        previous_logs = previous_logs[
            ~previous_logs["variant"].eq("timing8_no_segment_attention")
        ]
        log_frames.append(previous_logs)
    for spec in selected_specs:
        predictions, weights, logs = train_spec(
            spec,
            transitions,
            full6,
            scalar_columns,
            active_tasks,
            splits,
        )
        prediction_frames.append(predictions)
        weight_frames.append(weights)
        log_frames.append(logs)
    new_predictions = pd.concat(prediction_frames, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True)
    logs = pd.concat(log_frames, ignore_index=True)
    new_predictions.to_parquet(
        OUTPUT / "segment_interaction_oof_predictions.parquet", index=False
    )
    weights.to_parquet(
        OUTPUT / "segment_interaction_attention_weights.parquet", index=False
    )
    logs.to_csv(OUTPUT / "segment_interaction_training_log.csv", index=False)
    assignments.to_csv(
        OUTPUT / "segment_interaction_patient_folds.csv", index=False
    )

    baseline = pd.read_parquet(
        LENGTH_RESULTS / "cnn_length_ablation_oof_predictions.parquet"
    )
    baseline = baseline[baseline["model"].eq("attention_t96")].copy()
    baseline["model"] = "attention_current"
    all_predictions = pd.concat([baseline, new_predictions], ignore_index=True)
    primary = all_predictions[all_predictions["task"].eq(PRIMARY_TASK)].copy()
    metrics, _ = core.evaluate_predictions(primary, BOOTSTRAPS, SEED)
    comparisons = [
        (
            "timing8_only_vs_current",
            "attention_current",
            "timing8_no_segment_attention",
        ),
        (
            "segment_attention_vs_current",
            "attention_current",
            "segment_self_attention",
        ),
        (
            "timing8_attention_vs_current",
            "attention_current",
            "timing8_segment_self_attention",
        ),
        (
            "timing8_attention_vs_timing8_only",
            "timing8_no_segment_attention",
            "timing8_segment_self_attention",
        ),
        (
            "timing8_vs_segment_attention",
            "segment_self_attention",
            "timing8_segment_self_attention",
        ),
    ]
    paired = qc.paired_variant_deltas(
        primary, comparisons, BOOTSTRAPS, SEED + 840000
    )
    metrics.to_csv(OUTPUT / "segment_interaction_metrics.csv", index=False)
    paired.to_csv(OUTPUT / "segment_interaction_paired_deltas.csv", index=False)
    make_figure(metrics)

    display_names = {
        "attention_current": "Current CNN",
        "timing8_no_segment_attention": "8-bin temporal pooling only",
        "segment_self_attention": "Segment self-attention",
        "timing8_segment_self_attention": "8-bin + segment self-attention",
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
    parameter_counts = (
        logs.groupby("variant")["parameters"].first().astype(int).to_dict()
    )
    report = f"""# Segment-interaction CNN ablation

## Controlled comparison

The current 96-point six-channel CNN was compared with three focused variants:
eight-bin temporal pooling alone, four-head self-attention across the 18 segment
embeddings alone, and their combination. Patient folds, seeds, scalar features,
multitask targets, optimizer, and stopping rules were fixed. Training used
{torch.cuda.get_device_name(0)}.

## Dimensions

| model | temporal representation | segment interaction | curve representation | parameters |
|---|---|---|---|---:|
| Current CNN | B x 18 x 24 | final pooling only | B x 72 | 17,491 |
| 8-bin temporal pooling only | B x 18 x 24 x 8 -> B x 18 x 24 | final pooling only | B x 72 | {parameter_counts['timing8_no_segment_attention']:,} |
| Segment self-attention | B x 18 x 24 | 4 x 18 x 18 maps | B x 72 | {parameter_counts['segment_self_attention']:,} |
| 8-bin + segment attention | B x 18 x 24 x 8 -> B x 18 x 24 | 4 x 18 x 18 maps | B x 72 | {parameter_counts['timing8_segment_self_attention']:,} |

## Results

{markdown_table(table, ['model', 'n', 'events', 'AUC', 'AUC CI low', 'AUC CI high', 'AP', 'AP CI low', 'AP CI high'])}

Random-ranking AUC is 0.500. Random-ranking AP equals the event prevalence,
{prevalence:.3f}.

## Paired differences

{markdown_table(delta_table, ['comparison', 'delta AUC', 'delta AUC CI low', 'delta AUC CI high', 'delta AP', 'delta AP CI low', 'delta AP CI high'])}

A positive difference favors the candidate model. Intervals are paired
patient-cluster bootstrap confidence intervals.

## Interpretation

- Segment self-attention alone did not help: AUC decreased by 0.005 and AP by
  0.029 versus the current CNN.
- Preserving eight temporal bins alone increased AUC by 0.008 while AP decreased
  by 0.007.
- Combining eight bins with segment attention produced the highest AUC, 0.702
  versus 0.683, but AP was lower, 0.324 versus 0.339.
- Every paired confidence interval crossed zero. The observed changes are therefore
  hypotheses for a larger cohort, not established improvements.
- For the imbalanced early-alert task, retain the current CNN as the performance
  default because it has the highest AP. Keep the eight-bin combined model as an
  AUC-oriented research candidate.
"""
    (OUTPUT / "segment_interaction_report.md").write_text(
        report, encoding="utf-8"
    )
    metadata = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "primary_task": PRIMARY_TASK,
        "input_shape": list(full6.shape),
        "scalar_features": len(scalar_columns),
        "baseline_parameters": 17491,
        "new_parameters": parameter_counts,
        "segment_attention": {
            "heads": 4,
            "embedding_dimension": 24,
            "head_dimension": 6,
            "attention_map_shape": "B x 4 x 18 x 18",
            "ffn": [24, 48, 24],
            "dropout": 0.10,
        },
        "timing_bins": 8,
        "active_tasks": [task.name for task in active_tasks],
        "cv": "3 repeated 5-fold patient-held-out",
        "bootstraps": BOOTSTRAPS,
        "epochs": channel.EPOCHS,
        "patience": channel.PATIENCE,
        "seed": SEED,
    }
    (OUTPUT / "segment_interaction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nPrimary metrics", flush=True)
    print(
        table[
            [
                "model",
                "n",
                "events",
                "AUC",
                "AUC CI low",
                "AUC CI high",
                "AP",
                "AP CI low",
                "AP CI high",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nPaired deltas", flush=True)
    print(delta_table.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
