from __future__ import annotations

import argparse
import itertools
import json
import math
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from torch import nn

import cardiotoxicity_next_visit_gpu as core
import cardiotoxicity_nonapical_qc as qc
import cardiotoxicity_timeseries_round1 as round1
import cardiotoxicity_timeseries_round2 as round2


ROOT = Path(r"D:\us")
OUTPUT = ROOT / "cardiotoxicity_timeseries_round3_results"
SMOKE_OUTPUT = ROOT / "cardiotoxicity_timeseries_round3_smoke"
ROUND2_RESULTS = ROOT / "cardiotoxicity_timeseries_round2_results"
PRIMARY_TASK = "mid_first_rel15"
SEED = 20260722
BOOTSTRAPS = 2000
EPOCHS = 180
PATIENCE = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round 3 focused strain-curve experiments")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class PatchTokenEncoder(nn.Module):
    """The exact patch representation used by the Round 2 TimeMIL-lite model."""

    def __init__(self):
        super().__init__()
        self.patch_size = 8
        self.patches = 12
        self.patch_encoder = nn.Sequential(
            nn.LayerNorm(6 * self.patch_size),
            nn.Linear(6 * self.patch_size, 32),
            nn.GELU(),
        )
        self.segment_embedding = nn.Parameter(torch.zeros(18, 32))
        self.time_embedding = nn.Parameter(torch.zeros(self.patches, 32))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 32))
        nn.init.normal_(self.segment_embedding, std=0.02)
        positions = torch.linspace(0, 1, self.patches).unsqueeze(1)
        frequencies = torch.arange(1, 17, dtype=torch.float32).unsqueeze(0)
        wave = torch.cat(
            [
                torch.sin(2 * math.pi * positions * frequencies),
                torch.cos(2 * math.pi * positions * frequencies),
            ],
            dim=1,
        )[:, :32]
        with torch.no_grad():
            self.time_embedding.copy_(wave)

    def forward(self, curves: torch.Tensor) -> torch.Tensor:
        batch = curves.shape[0]
        patches = curves.unfold(-1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 1, 3, 2, 4).reshape(batch, 18, self.patches, -1)
        tokens = self.patch_encoder(patches)
        tokens = (
            tokens
            + self.segment_embedding[None, :, None, :]
            + self.time_embedding[None, None, :, :]
        )
        if self.training:
            drop = torch.rand(batch, 18, self.patches, 1, device=tokens.device) < 0.10
            tokens = torch.where(drop, self.mask_token.expand_as(tokens), tokens)
        return tokens


def transformer_layer() -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        dropout=0.10,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class CardioHead(nn.Module):
    def __init__(self, scalar_features: int, tasks: int, use_scalars: bool):
        super().__init__()
        self.use_scalars = use_scalars
        if use_scalars:
            self.scalar_branch = round2.ScalarBranch(scalar_features)
        dimensions = 32 + (32 if use_scalars else 0)
        self.shared = nn.Sequential(nn.Linear(dimensions, 64), nn.GELU(), nn.Dropout(0.35))
        self.head = nn.Linear(64, tasks)

    def forward(self, pooled: torch.Tensor, scalars: torch.Tensor | None) -> torch.Tensor:
        if self.use_scalars:
            pooled = torch.cat([pooled, self.scalar_branch(scalars)], dim=1)
        return self.head(self.shared(pooled))


class TimeMILUniformCardio(nn.Module):
    """Round 2 TimeMIL-lite with attention replaced by uniform mean pooling."""

    def __init__(self, scalar_features: int, tasks: int, use_scalars: bool):
        super().__init__()
        self.tokens = PatchTokenEncoder()
        self.context = nn.TransformerEncoder(transformer_layer(), num_layers=1)
        self.output = CardioHead(scalar_features, tasks, use_scalars)

    def forward(self, curves: torch.Tensor, scalars: torch.Tensor | None):
        batch = curves.shape[0]
        tokens = self.tokens(curves).reshape(batch, 18 * 12, 32)
        tokens = self.context(tokens)
        pooled = tokens.mean(dim=1)
        weights = torch.full(
            (batch, 18, 12), 1.0 / (18 * 12), device=curves.device, dtype=curves.dtype
        )
        return self.output(pooled, scalars), weights


class TimeMILHierarchicalCardio(nn.Module):
    """Temporal attention within segments, then contextual segment attention."""

    def __init__(self, scalar_features: int, tasks: int, use_scalars: bool):
        super().__init__()
        self.tokens = PatchTokenEncoder()
        self.temporal_context = nn.TransformerEncoder(transformer_layer(), num_layers=1)
        self.temporal_score = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
        self.segment_context = nn.TransformerEncoder(transformer_layer(), num_layers=1)
        self.segment_score = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
        self.output = CardioHead(scalar_features, tasks, use_scalars)

    def forward(self, curves: torch.Tensor, scalars: torch.Tensor | None):
        batch = curves.shape[0]
        tokens = self.tokens(curves)
        temporal = self.temporal_context(tokens.reshape(batch * 18, 12, 32))
        temporal_weight = torch.softmax(self.temporal_score(temporal).squeeze(-1), dim=1)
        segment_tokens = torch.sum(temporal * temporal_weight.unsqueeze(-1), dim=1)
        segment_tokens = self.segment_context(segment_tokens.reshape(batch, 18, 32))
        segment_weight = torch.softmax(self.segment_score(segment_tokens).squeeze(-1), dim=1)
        pooled = torch.sum(segment_tokens * segment_weight.unsqueeze(-1), dim=1)
        instance_weight = segment_weight.unsqueeze(-1) * temporal_weight.reshape(batch, 18, 12)
        return self.output(pooled, scalars), instance_weight


def model_specs() -> list[tuple[str, type[nn.Module], bool]]:
    return [
        ("timemil_uniform_curves", TimeMILUniformCardio, False),
        ("timemil_uniform_curves_scalars", TimeMILUniformCardio, True),
        ("timemil_hierarchical_curves", TimeMILHierarchicalCardio, False),
        ("timemil_hierarchical_curves_scalars", TimeMILHierarchicalCardio, True),
    ]


def simplex_weights(dimensions: int, denominator: int = 10) -> np.ndarray:
    rows = []
    for values in itertools.product(range(denominator + 1), repeat=dimensions):
        if sum(values) == denominator:
            rows.append(np.asarray(values, dtype=float) / denominator)
    return np.stack(rows)


def add_fixed_ensembles(base: pd.DataFrame, combinations: dict[str, list[str]]) -> pd.DataFrame:
    template = base[base["model"].eq("current_cnn")].copy()
    pivot = base.pivot(index="transition_id", columns="model", values="score")
    rows = []
    for name, models in combinations.items():
        scores = pivot[models].mean(axis=1)
        frame = template.drop(columns="score").merge(
            scores.rename("score"), left_on="transition_id", right_index=True, how="left"
        )
        frame["model"] = f"ensemble_equal_{name}"
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def add_crossfit_blends(
    base: pd.DataFrame,
    transitions: pd.DataFrame,
    splits: list[dict[str, object]],
    combinations: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    template = base[base["model"].eq("current_cnn")].copy()
    transition_order = template["transition_id"].tolist()
    pivot = base.pivot(index="transition_id", columns="model", values="score").loc[transition_order]
    patient = template.set_index("transition_id").loc[transition_order, "patient_id"].astype(str)
    labels = template.set_index("transition_id").loc[transition_order, "label"].to_numpy(int)
    rows = []
    weight_rows = []
    for name, models in combinations.items():
        candidates = simplex_weights(len(models))
        score_sum = np.zeros(len(template), dtype=float)
        score_count = np.zeros(len(template), dtype=int)
        x = pivot[models].to_numpy(float)
        for split in splits:
            train_mask = patient.isin(split["train_patients"]).to_numpy()
            test_mask = patient.isin(split["test_patients"]).to_numpy()
            best_ap = -np.inf
            best_weight = candidates[0]
            # Select convex weights with training-patient AP only. A tiny equal-weight
            # preference resolves ties and reduces arbitrary extreme solutions.
            equal = np.full(len(models), 1.0 / len(models))
            for weight in candidates:
                ap = average_precision_score(labels[train_mask], x[train_mask] @ weight)
                objective = ap - 1e-6 * float(np.square(weight - equal).sum())
                if objective > best_ap:
                    best_ap = objective
                    best_weight = weight
            score_sum[test_mask] += x[test_mask] @ best_weight
            score_count[test_mask] += 1
            for model, weight in zip(models, best_weight):
                weight_rows.append(
                    {
                        "ensemble": name,
                        "repeat": split["repeat"],
                        "fold": split["fold"],
                        "model": model,
                        "weight": float(weight),
                        "train_ap": float(average_precision_score(labels[train_mask], x[train_mask] @ best_weight)),
                    }
                )
        if not np.all(score_count == 3):
            raise RuntimeError(f"Incomplete cross-fit blend {name}: {np.unique(score_count, return_counts=True)}")
        frame = template.copy()
        frame["model"] = f"ensemble_crossfit_{name}"
        frame["score"] = score_sum / score_count
        frame["prediction_repeats"] = score_count
        rows.append(frame)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(weight_rows)


def make_figure(output: Path, predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    selected = [
        "current_cnn",
        "mantis_random_frozen_curves_scalars",
        "timemil_curves_scalars",
        "timemil_uniform_curves_scalars",
        "timemil_hierarchical_curves_scalars",
        "ensemble_equal_cnn_mantis_timemil",
        "ensemble_crossfit_cnn_mantis_timemil",
    ]
    labels = {
        "current_cnn": "Current CNN",
        "mantis_random_frozen_curves_scalars": "Random Mantis",
        "timemil_curves_scalars": "TimeMIL attention",
        "timemil_uniform_curves_scalars": "TimeMIL uniform",
        "timemil_hierarchical_curves_scalars": "TimeMIL hierarchical",
        "ensemble_equal_cnn_mantis_timemil": "Equal ensemble",
        "ensemble_crossfit_cnn_mantis_timemil": "Cross-fit ensemble",
    }
    metric_lookup = metrics.set_index("model")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for model in selected:
        group = predictions[predictions["model"].eq(model)]
        if group.empty:
            continue
        y = group["label"].to_numpy(int)
        score = group["score"].to_numpy(float)
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        axes[0].plot(fpr, tpr, linewidth=2, label=f"{labels[model]} ({metric_lookup.loc[model, 'roc_auc']:.3f})")
        axes[1].plot(recall, precision, linewidth=2, label=f"{labels[model]} ({metric_lookup.loc[model, 'average_precision']:.3f})")
    prevalence = predictions[predictions["model"].eq("current_cnn")]["label"].mean()
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].axhline(prevalence, color="k", linestyle="--", alpha=0.5)
    axes[0].set(title="ROC curves", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].set(title="Precision-recall curves", xlabel="Recall", ylabel="Precision")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="best")
    figure.suptitle("Round 3: pooling architectures and leakage-safe ensembles")
    figure.tight_layout()
    figure.savefig(output / "round3_roc_pr_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return round1.markdown_table(frame, columns)


def write_report(
    output: Path,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    ablations: pd.DataFrame,
    ensemble_ablations: pd.DataFrame,
    weights: pd.DataFrame,
    attention: pd.DataFrame,
) -> None:
    names = {
        "current_cnn": "Current CNN",
        "mantis_random_frozen_curves_scalars": "Random Mantis + scalars",
        "moment_small_frozen_curves_scalars": "MOMENT-small + scalars",
        "scalar_mlp": "Scalar-only MLP",
        "timemil_curves": "TimeMIL attention: curves",
        "timemil_curves_scalars": "TimeMIL attention: curves + scalars",
        "timemil_uniform_curves": "TimeMIL uniform: curves",
        "timemil_uniform_curves_scalars": "TimeMIL uniform: curves + scalars",
        "timemil_hierarchical_curves": "TimeMIL hierarchical: curves",
        "timemil_hierarchical_curves_scalars": "TimeMIL hierarchical: curves + scalars",
    }
    for model in metrics["model"]:
        if model.startswith("ensemble_equal_"):
            names[model] = "Equal ensemble: " + model.removeprefix("ensemble_equal_").replace("_", " + ")
        elif model.startswith("ensemble_crossfit_"):
            names[model] = "Cross-fit ensemble: " + model.removeprefix("ensemble_crossfit_").replace("_", " + ")
    table = metrics.copy()
    table["Model"] = table["model"].map(names).fillna(table["model"])
    for source, target in [
        ("roc_auc", "AUC"),
        ("roc_auc_ci_low", "AUC CI low"),
        ("roc_auc_ci_high", "AUC CI high"),
        ("average_precision", "AP"),
        ("average_precision_ci_low", "AP CI low"),
        ("average_precision_ci_high", "AP CI high"),
    ]:
        table[target] = table[source].map(lambda value: f"{value:.3f}")
    table = table.sort_values("average_precision", ascending=False)

    delta_table = deltas.copy()
    delta_table["Model"] = delta_table["candidate_model"].map(names).fillna(delta_table["candidate_model"])
    for source, target in [
        ("delta_roc_auc", "delta AUC"),
        ("delta_roc_auc_ci_low", "delta AUC CI low"),
        ("delta_roc_auc_ci_high", "delta AUC CI high"),
        ("delta_average_precision", "delta AP"),
        ("delta_average_precision_ci_low", "delta AP CI low"),
        ("delta_average_precision_ci_high", "delta AP CI high"),
    ]:
        delta_table[target] = delta_table[source].map(lambda value: f"{value:.3f}")

    ablation_names = {
        "attention_vs_uniform_curves": "Attention vs uniform, curves",
        "attention_vs_uniform_scalars": "Attention vs uniform, curves + scalars",
        "hierarchical_vs_attention_curves": "Hierarchical vs flat attention, curves",
        "hierarchical_vs_attention_scalars": "Hierarchical vs flat attention, curves + scalars",
        "hierarchical_vs_uniform_curves": "Hierarchical vs uniform, curves",
        "hierarchical_vs_uniform_scalars": "Hierarchical vs uniform, curves + scalars",
        "hierarchical_scalars": "Add scalars to hierarchical TimeMIL",
    }
    ablation_table = ablations.copy()
    ablation_table["Ablation"] = ablation_table["comparison"].map(ablation_names)
    for source, target in [
        ("delta_roc_auc", "delta AUC"),
        ("delta_roc_auc_ci_low", "delta AUC CI low"),
        ("delta_roc_auc_ci_high", "delta AUC CI high"),
        ("delta_average_precision", "delta AP"),
        ("delta_average_precision_ci_low", "delta AP CI low"),
        ("delta_average_precision_ci_high", "delta AP CI high"),
    ]:
        ablation_table[target] = ablation_table[source].map(lambda value: f"{value:.3f}")

    ensemble_names = {
        "best_equal_vs_mantis": "Best equal ensemble vs random Mantis",
        "best_equal_vs_timemil": "Best equal ensemble vs TimeMIL attention",
        "best_equal_vs_crossfit": "Equal vs cross-fit, CNN + Mantis + TimeMIL",
        "equal_cnn_mantis_vs_crossfit": "Equal vs cross-fit, CNN + Mantis",
    }
    ensemble_table = ensemble_ablations.copy()
    ensemble_table["Comparison"] = ensemble_table["comparison"].map(ensemble_names)
    for source, target in [
        ("delta_roc_auc", "delta AUC"),
        ("delta_roc_auc_ci_low", "delta AUC CI low"),
        ("delta_roc_auc_ci_high", "delta AUC CI high"),
        ("delta_average_precision", "delta AP"),
        ("delta_average_precision_ci_low", "delta AP CI low"),
        ("delta_average_precision_ci_high", "delta AP CI high"),
    ]:
        ensemble_table[target] = ensemble_table[source].map(lambda value: f"{value:.3f}")

    weight_summary = weights.groupby(["ensemble", "model"], as_index=False)["weight"].agg(["mean", "std"]).reset_index()
    weight_summary["Model"] = weight_summary["model"].map(names).fillna(weight_summary["model"])
    weight_summary["Mean weight"] = weight_summary["mean"].map(lambda value: f"{value:.3f}")
    weight_summary["SD"] = weight_summary["std"].map(lambda value: f"{value:.3f}")
    weight_summary["Ensemble"] = weight_summary["ensemble"].str.replace("_", " + ")

    attention_summary = attention.groupby(["model", "segment"], as_index=False)["weight"].sum().groupby(["model", "segment"], as_index=False)["weight"].mean()
    # Sum within each transition first so every transition contributes equally.
    attention_summary = (
        attention.groupby(["model", "transition_id", "segment"], as_index=False)["weight"].sum()
        .groupby(["model", "segment"], as_index=False)["weight"].mean()
    )
    top_attention = attention_summary.sort_values(["model", "weight"], ascending=[True, False]).groupby("model").head(5)
    top_attention["Model"] = top_attention["model"].map(names)
    top_attention["Segment"] = top_attention["segment"].astype(int)
    top_attention["Mean segment mass"] = top_attention["weight"].map(lambda value: f"{value:.4f}")

    report = f"""# Round 3: focused pooling and ensemble experiments

## Locked protocol

The target is whether the immediately following visit is the first visit with at
least 15% relative Mid-GLS deterioration from first-visit baseline. Evaluation
uses 238 transitions, 49 events, 103 patients, the same three repeated five-fold
patient-held-out splits, and 2,000 patient-cluster bootstrap samples.

## Results

{markdown_table(table, ['Model', 'AUC', 'AUC CI low', 'AUC CI high', 'AP', 'AP CI low', 'AP CI high'])}

Random guessing has AUC 0.500 and expected AP 0.206.

## Paired changes from current CNN

{markdown_table(delta_table, ['Model', 'delta AUC', 'delta AUC CI low', 'delta AUC CI high', 'delta AP', 'delta AP CI low', 'delta AP CI high'])}

## Pooling ablations

Positive values favor the first named method.

{markdown_table(ablation_table, ['Ablation', 'delta AUC', 'delta AUC CI low', 'delta AUC CI high', 'delta AP', 'delta AP CI low', 'delta AP CI high'])}

## Architecture definitions

- Uniform TimeMIL retains the same 216 patch tokens and global transformer as
  Round 2, but replaces learned MIL attention with an unweighted mean.
- Hierarchical TimeMIL first applies temporal self-attention to the 12 patches
  within each segment, pools each segment, then applies a second self-attention
  layer and learned pooling across the 18 segments.
- Every neural variant uses the same scalar branch, multitask loss, class
  weighting, inner patient validation, and early stopping as the retained CNN.

## Cross-fitted ensemble weights

The convex blend weights were selected separately inside every outer split using
only training-patient labels and already patient-out-of-fold base predictions.
They were then applied to the held-out patients. Equal-weight ensembles are fully
prespecified controls.

{markdown_table(weight_summary, ['Ensemble', 'Model', 'Mean weight', 'SD'])}

## Ensemble ablations

Positive values favor the equal ensemble in each comparison.

{markdown_table(ensemble_table, ['Comparison', 'delta AUC', 'delta AUC CI low', 'delta AUC CI high', 'delta AP', 'delta AP CI low', 'delta AP CI high'])}

## Hierarchical attention

Uniform segment mass is 1/18 = 0.0556. Attention weights are descriptive rather
than causal feature importance.

{markdown_table(top_attention, ['Model', 'Segment', 'Mean segment mass'])}
"""
    (output / "round3_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Round 3")
    output = SMOKE_OUTPUT if args.smoke else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    transitions, curves, scalar_columns, active_tasks, all_splits = round1.load_inputs()
    splits = all_splits[:1] if args.smoke else all_splits
    epochs = 2 if args.smoke else EPOCHS
    patience = 1 if args.smoke else PATIENCE

    started = time.time()
    new_predictions, training_log, attention = round2.supervised_oof(
        transitions,
        curves,
        scalar_columns,
        active_tasks,
        splits,
        epochs,
        patience,
        specs_override=model_specs(),
    )
    new_predictions.to_parquet(output / "round3_new_model_oof_predictions.parquet", index=False)
    training_log.to_csv(output / "round3_training_log.csv", index=False)
    attention.to_parquet(output / "round3_attention.parquet", index=False)
    if args.smoke:
        new_predictions.to_parquet(output / "round3_smoke_predictions.parquet", index=False)
        training_log.to_csv(output / "round3_smoke_training_log.csv", index=False)
        attention.to_parquet(output / "round3_smoke_attention.parquet", index=False)
        print(new_predictions.groupby("model").size())
        print(training_log.to_string(index=False))
        return 0

    prior = pd.read_parquet(ROUND2_RESULTS / "round2_oof_predictions.parquet")
    keep_prior = [
        "clinical_ridge",
        "current_cnn",
        "mantis_random_frozen_curves_scalars",
        "moment_small_frozen_curves_scalars",
        "scalar_mlp",
        "timemil_curves",
        "timemil_curves_scalars",
    ]
    base = pd.concat([prior[prior["model"].isin(keep_prior)], new_predictions], ignore_index=True)
    combinations = {
        "cnn_mantis": ["current_cnn", "mantis_random_frozen_curves_scalars"],
        "cnn_moment": ["current_cnn", "moment_small_frozen_curves_scalars"],
        "cnn_timemil": ["current_cnn", "timemil_curves_scalars"],
        "cnn_hierarchical": ["current_cnn", "timemil_hierarchical_curves_scalars"],
        "cnn_mantis_moment": ["current_cnn", "mantis_random_frozen_curves_scalars", "moment_small_frozen_curves_scalars"],
        "cnn_mantis_timemil": ["current_cnn", "mantis_random_frozen_curves_scalars", "timemil_curves_scalars"],
        "cnn_mantis_moment_timemil": ["current_cnn", "mantis_random_frozen_curves_scalars", "moment_small_frozen_curves_scalars", "timemil_curves_scalars"],
    }
    fixed = add_fixed_ensembles(base, combinations)
    crossfit, ensemble_weights = add_crossfit_blends(base, transitions, all_splits, combinations)
    predictions = pd.concat([base, fixed, crossfit], ignore_index=True)
    completeness = predictions.groupby("model")["transition_id"].nunique()
    if not (completeness == 238).all():
        raise RuntimeError(f"Incomplete OOF predictions: {completeness.to_dict()}")

    metrics, _ = core.evaluate_predictions(predictions, BOOTSTRAPS, SEED + 2010000)
    candidate_models = [
        name for name in predictions["model"].unique()
        if name not in {"clinical_ridge", "current_cnn", "mantis_random_frozen_curves_scalars", "moment_small_frozen_curves_scalars", "scalar_mlp", "timemil_curves", "timemil_curves_scalars"}
    ]
    comparisons = [(f"{name}_vs_cnn", "current_cnn", name) for name in candidate_models]
    deltas = qc.paired_variant_deltas(predictions, comparisons, BOOTSTRAPS, SEED + 2020000)
    ablation_specs = [
        ("attention_vs_uniform_curves", "timemil_uniform_curves", "timemil_curves"),
        ("attention_vs_uniform_scalars", "timemil_uniform_curves_scalars", "timemil_curves_scalars"),
        ("hierarchical_vs_attention_curves", "timemil_curves", "timemil_hierarchical_curves"),
        ("hierarchical_vs_attention_scalars", "timemil_curves_scalars", "timemil_hierarchical_curves_scalars"),
        ("hierarchical_vs_uniform_curves", "timemil_uniform_curves", "timemil_hierarchical_curves"),
        ("hierarchical_vs_uniform_scalars", "timemil_uniform_curves_scalars", "timemil_hierarchical_curves_scalars"),
        ("hierarchical_scalars", "timemil_hierarchical_curves", "timemil_hierarchical_curves_scalars"),
    ]
    ablations = qc.paired_variant_deltas(predictions, ablation_specs, BOOTSTRAPS, SEED + 2030000)
    ensemble_ablation_specs = [
        ("best_equal_vs_mantis", "mantis_random_frozen_curves_scalars", "ensemble_equal_cnn_mantis_timemil"),
        ("best_equal_vs_timemil", "timemil_curves_scalars", "ensemble_equal_cnn_mantis_timemil"),
        ("best_equal_vs_crossfit", "ensemble_crossfit_cnn_mantis_timemil", "ensemble_equal_cnn_mantis_timemil"),
        ("equal_cnn_mantis_vs_crossfit", "ensemble_crossfit_cnn_mantis", "ensemble_equal_cnn_mantis"),
    ]
    ensemble_ablations = qc.paired_variant_deltas(
        predictions, ensemble_ablation_specs, BOOTSTRAPS, SEED + 2040000
    )

    predictions.to_parquet(output / "round3_oof_predictions.parquet", index=False)
    metrics.to_csv(output / "round3_metrics.csv", index=False)
    deltas.to_csv(output / "round3_paired_deltas_vs_cnn.csv", index=False)
    ablations.to_csv(output / "round3_pooling_ablation_deltas.csv", index=False)
    ensemble_ablations.to_csv(output / "round3_ensemble_ablation_deltas.csv", index=False)
    ensemble_weights.to_csv(output / "round3_ensemble_weights.csv", index=False)
    training_log.to_csv(output / "round3_training_log.csv", index=False)
    attention.to_parquet(output / "round3_attention.parquet", index=False)
    make_figure(output, predictions, metrics)
    write_report(output, metrics, deltas, ablations, ensemble_ablations, ensemble_weights, attention)
    metadata = {
        "task": PRIMARY_TASK,
        "patients": int(transitions["patient_id"].nunique()),
        "eligible_transitions": 238,
        "events": 49,
        "event_rate": 49 / 238,
        "curve_shape": list(curves.shape),
        "cv": "3 repeated 5-fold patient-held-out",
        "bootstraps": BOOTSTRAPS,
        "device": torch.cuda.get_device_name(0),
        "seconds": time.time() - started,
    }
    (output / "round3_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics.sort_values("average_precision", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
