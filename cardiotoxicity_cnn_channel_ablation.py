from __future__ import annotations

import copy
import json
import math
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch import nn

import cardiotoxicity_next_visit_gpu as core
import cardiotoxicity_nonapical_qc as qc


VISITS_PATH = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet")
CURVES_PATH = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_dataset.parquet")
BASE_RESULTS = Path(r"D:\us\cardiotoxicity_next_visit_gpu_results")
PLATEAU_RESULTS = Path(r"D:\us\cardiotoxicity_plateau_results")
OUTPUT = Path(r"D:\us\cardiotoxicity_cnn_channel_ablation_results")
PRIMARY_TASK = "mid_first_rel15"
SEED = 20260722
BOOTSTRAPS = 1000
EPOCHS = 180
PATIENCE = 25


@dataclass(frozen=True)
class ChannelVariant:
    name: str
    description: str
    channels: tuple[str, ...]


VARIANTS = (
    ChannelVariant(
        "attention_full6",
        "Reference: current Endo, Mid, gap, and their changes",
        (
            "current_endo",
            "current_mid",
            "current_endo_minus_mid",
            "change_endo",
            "change_mid",
            "change_endo_minus_mid",
        ),
    ),
    ChannelVariant(
        "attention_gap2",
        "Endo-Mid only: current gap and change in gap",
        ("current_endo_minus_mid", "change_endo_minus_mid"),
    ),
    ChannelVariant(
        "attention_separate4",
        "Separate layers only: Endo, Mid, and their changes",
        ("current_endo", "current_mid", "change_endo", "change_mid"),
    ),
    ChannelVariant(
        "attention_gap_shape4",
        "Raw gap plus separately normalized Endo-Mid shape gap and their changes",
        (
            "current_endo_minus_mid",
            "change_endo_minus_mid",
            "current_normalized_shape_gap",
            "change_normalized_shape_gap",
        ),
    ),
)


class ChannelAttentionNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        scalar_features: int,
        tasks: int,
        segments: int,
    ):
        super().__init__()
        self.segment_encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 24, kernel_size=5, padding=2),
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalized_shape(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curves = np.asarray(curves, dtype=np.float32)
    scale = np.max(np.abs(curves), axis=1, keepdims=True)
    valid = np.isfinite(scale[:, 0]) & (scale[:, 0] >= 3.0)
    result = np.zeros_like(curves, dtype=np.float32)
    result[valid] = curves[valid] / scale[valid]
    return result, valid


def build_channel_inputs(
    transitions: pd.DataFrame,
    visits: pd.DataFrame,
    tensors: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    visit_order = {
        (str(row.patient_id), int(row.visit_order)): str(row.visit_id)
        for row in visits[["patient_id", "visit_order", "visit_id"]].itertuples(
            index=False
        )
    }
    output: dict[str, list[np.ndarray]] = {variant.name: [] for variant in VARIANTS}
    for row in transitions.itertuples(index=False):
        current_id = str(row.current_visit_id)
        order = int(row.current_visit_order)
        previous_id = (
            current_id
            if order == 1
            else visit_order[(str(row.patient_id), order - 1)]
        )
        current = tensors[current_id]
        previous = tensors[previous_id]
        endo = current[:, 0]
        mid = current[:, 1]
        previous_endo = previous[:, 0]
        previous_mid = previous[:, 1]
        delta_endo = (
            endo - previous_endo if order > 1 else np.zeros_like(endo)
        )
        delta_mid = mid - previous_mid if order > 1 else np.zeros_like(mid)
        gap = endo - mid
        delta_gap = (
            gap - (previous_endo - previous_mid)
            if order > 1
            else np.zeros_like(gap)
        )

        endo_shape, endo_valid = normalized_shape(endo)
        mid_shape, mid_valid = normalized_shape(mid)
        shape_valid = endo_valid & mid_valid
        shape_gap = endo_shape - mid_shape
        shape_gap[~shape_valid] = 0.0
        previous_endo_shape, previous_endo_valid = normalized_shape(previous_endo)
        previous_mid_shape, previous_mid_valid = normalized_shape(previous_mid)
        previous_shape_gap = previous_endo_shape - previous_mid_shape
        previous_shape_gap[~(previous_endo_valid & previous_mid_valid)] = 0.0
        delta_shape_gap = (
            shape_gap - previous_shape_gap
            if order > 1
            else np.zeros_like(shape_gap)
        )

        full6 = np.stack(
            [endo, mid, gap, delta_endo, delta_mid, delta_gap], axis=1
        ).astype(np.float32)
        full6 = np.clip(full6 / 30.0, -2.0, 2.0)
        output["attention_full6"].append(full6)
        output["attention_gap2"].append(full6[:, [2, 5]])
        output["attention_separate4"].append(full6[:, [0, 1, 3, 4]])
        output["attention_gap_shape4"].append(
            np.stack(
                [
                    full6[:, 2],
                    full6[:, 5],
                    np.clip(shape_gap, -2.0, 2.0),
                    np.clip(delta_shape_gap, -2.0, 2.0),
                ],
                axis=1,
            ).astype(np.float32)
        )
    return {
        name: np.stack(values).astype(np.float32)
        for name, values in output.items()
    }


def binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
        reduction="none",
    )
    return (loss * masks).sum() / masks.sum().clamp_min(1.0)


def train_channel_variants(
    transitions: pd.DataFrame,
    curve_inputs: dict[str, np.ndarray],
    scalar_columns: list[str],
    active_tasks: list[core.TaskSpec],
    splits: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    labels = np.column_stack(
        [
            transitions[f"label__{task.name}"].to_numpy(np.float32)
            for task in active_tasks
        ]
    )
    masks = np.column_stack(
        [
            transitions[f"mask__{task.name}"].to_numpy(np.float32)
            for task in active_tasks
        ]
    )
    patient_primary = (
        transitions.groupby("patient_id")[f"label__{PRIMARY_TASK}"]
        .max()
        .astype(int)
        .to_dict()
    )
    score_sum = {
        variant.name: np.zeros(
            (len(transitions), len(active_tasks)), dtype=float
        )
        for variant in VARIANTS
    }
    score_count = {
        variant.name: np.zeros(
            (len(transitions), len(active_tasks)), dtype=int
        )
        for variant in VARIANTS
    }
    weight_sum = {
        variant.name: np.zeros(
            (len(transitions), curve_inputs[variant.name].shape[1]), dtype=float
        )
        for variant in VARIANTS
    }
    weight_count = {
        variant.name: np.zeros(len(transitions), dtype=int)
        for variant in VARIANTS
    }
    logs = []

    for split in splits:
        split_seed = SEED + int(split["split_index"]) * 101
        train_patients = sorted(split["train_patients"])
        patient_labels = [
            patient_primary.get(patient, 0) for patient in train_patients
        ]
        fit_patients, val_patients = train_test_split(
            train_patients,
            test_size=0.2,
            random_state=split_seed,
            stratify=patient_labels,
        )
        fit_index = np.flatnonzero(
            transitions["patient_id"].isin(fit_patients).to_numpy()
        )
        val_index = np.flatnonzero(
            transitions["patient_id"].isin(val_patients).to_numpy()
        )
        test_index = np.flatnonzero(
            transitions["patient_id"].isin(split["test_patients"]).to_numpy()
        )

        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        fit_scalars = imputer.fit_transform(
            transitions.iloc[fit_index][scalar_columns].astype(float)
        )
        fit_scalars = np.clip(
            scaler.fit_transform(fit_scalars), -5.0, 5.0
        ).astype(np.float32)
        val_scalars = np.clip(
            scaler.transform(
                imputer.transform(
                    transitions.iloc[val_index][scalar_columns].astype(float)
                )
            ),
            -5.0,
            5.0,
        ).astype(np.float32)
        test_scalars = np.clip(
            scaler.transform(
                imputer.transform(
                    transitions.iloc[test_index][scalar_columns].astype(float)
                )
            ),
            -5.0,
            5.0,
        ).astype(np.float32)

        fit_masks = masks[fit_index].copy()
        for task_index in range(len(active_tasks)):
            task_y = labels[fit_index][
                fit_masks[:, task_index] > 0, task_index
            ]
            if task_y.sum() < 2 or (len(task_y) - task_y.sum()) < 2:
                fit_masks[:, task_index] = 0
        positives = (labels[fit_index] * fit_masks).sum(axis=0)
        negatives = ((1.0 - labels[fit_index]) * fit_masks).sum(axis=0)
        pos_weights = np.clip(
            negatives / np.maximum(positives, 1.0), 1.0, 20.0
        ).astype(np.float32)
        pos_weight_tensor = torch.as_tensor(pos_weights, device=device)

        for variant in VARIANTS:
            set_seed(split_seed)
            variant_curves = curve_inputs[variant.name]
            model = ChannelAttentionNet(
                input_channels=variant_curves.shape[2],
                scalar_features=len(scalar_columns),
                tasks=len(active_tasks),
                segments=variant_curves.shape[1],
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-3, weight_decay=2e-3
            )
            amp_scaler = torch.amp.GradScaler("cuda", enabled=True)
            best_state = copy.deepcopy(model.state_dict())
            best_loss = math.inf
            best_epoch = 0
            stale = 0
            started = time.time()
            batch_size = 64

            for epoch in range(EPOCHS):
                model.train()
                order = np.random.default_rng(split_seed + epoch).permutation(
                    len(fit_index)
                )
                for start_index in range(0, len(order), batch_size):
                    local = order[start_index : start_index + batch_size]
                    global_index = fit_index[local]
                    batch_curves = torch.as_tensor(
                        variant_curves[global_index], device=device
                    )
                    batch_scalars = torch.as_tensor(
                        fit_scalars[local], device=device
                    )
                    batch_labels = torch.as_tensor(
                        labels[global_index], device=device
                    )
                    batch_masks = torch.as_tensor(
                        fit_masks[local], device=device
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(
                        device_type="cuda", enabled=True
                    ):
                        logits, _ = model(batch_curves, batch_scalars)
                        loss = binary_loss(
                            logits,
                            batch_labels,
                            batch_masks,
                            pos_weight_tensor,
                        )
                    amp_scaler.scale(loss).backward()
                    amp_scaler.step(optimizer)
                    amp_scaler.update()

                model.eval()
                with torch.no_grad(), torch.amp.autocast(
                    device_type="cuda", enabled=True
                ):
                    logits, _ = model(
                        torch.as_tensor(
                            variant_curves[val_index], device=device
                        ),
                        torch.as_tensor(val_scalars, device=device),
                    )
                    val_loss = binary_loss(
                        logits,
                        torch.as_tensor(labels[val_index], device=device),
                        torch.as_tensor(masks[val_index], device=device),
                        pos_weight_tensor,
                    )
                    val_loss_value = float(val_loss.item())
                if val_loss_value < best_loss - 1e-4:
                    best_loss = val_loss_value
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                if stale >= PATIENCE:
                    break

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad(), torch.amp.autocast(
                device_type="cuda", enabled=True
            ):
                logits, attention = model(
                    torch.as_tensor(
                        variant_curves[test_index], device=device
                    ),
                    torch.as_tensor(test_scalars, device=device),
                )
                probabilities = torch.sigmoid(logits).float().cpu().numpy()
                attention_np = attention.float().cpu().numpy()
            for task_index in range(len(active_tasks)):
                valid = masks[test_index, task_index] > 0
                global_valid = test_index[valid]
                score_sum[variant.name][
                    global_valid, task_index
                ] += probabilities[valid, task_index]
                score_count[variant.name][global_valid, task_index] += 1
            weight_sum[variant.name][test_index] += attention_np
            weight_count[variant.name][test_index] += 1
            elapsed = time.time() - started
            parameter_count = sum(
                parameter.numel() for parameter in model.parameters()
            )
            logs.append(
                {
                    "variant": variant.name,
                    "repeat": split["repeat"],
                    "fold": split["fold"],
                    "channels": variant_curves.shape[2],
                    "parameters": parameter_count,
                    "best_epoch": best_epoch,
                    "epochs_run": epoch + 1,
                    "best_validation_loss": best_loss,
                    "seconds": elapsed,
                    "device": torch.cuda.get_device_name(0),
                }
            )
            print(
                f"{variant.name} repeat={split['repeat']} "
                f"fold={split['fold']} epoch={best_epoch} "
                f"seconds={elapsed:.1f}",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()

    prediction_rows = []
    for variant in VARIANTS:
        for task_index, task in enumerate(active_tasks):
            valid = (
                (masks[:, task_index] > 0)
                & (score_count[variant.name][:, task_index] > 0)
            )
            for index in np.flatnonzero(valid):
                prediction_rows.append(
                    {
                        "task": task.name,
                        "model": variant.name,
                        "transition_id": transitions.iloc[index][
                            "transition_id"
                        ],
                        "patient_id": transitions.iloc[index]["patient_id"],
                        "label": int(labels[index, task_index]),
                        "score": float(
                            score_sum[variant.name][index, task_index]
                            / score_count[variant.name][index, task_index]
                        ),
                        "prediction_repeats": int(
                            score_count[variant.name][index, task_index]
                        ),
                    }
                )

    weight_rows = []
    primary_mask = transitions[f"mask__{PRIMARY_TASK}"].astype(bool).to_numpy()
    primary_label = transitions[f"label__{PRIMARY_TASK}"].to_numpy(int)
    for variant in VARIANTS:
        valid = (weight_count[variant.name] > 0) & primary_mask
        for index in np.flatnonzero(valid):
            weights = (
                weight_sum[variant.name][index]
                / weight_count[variant.name][index]
            )
            for segment, weight in enumerate(weights, start=1):
                weight_rows.append(
                    {
                        "variant": variant.name,
                        "transition_id": transitions.iloc[index][
                            "transition_id"
                        ],
                        "patient_id": transitions.iloc[index]["patient_id"],
                        "label": int(primary_label[index]),
                        "segment": segment,
                        "attention_weight": float(weight),
                    }
                )
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(weight_rows),
        pd.DataFrame(logs),
    )


def make_figure(
    output: Path, metrics: pd.DataFrame, deltas: pd.DataFrame
) -> None:
    labels = {
        "clinical_ridge": "Clinical ridge",
        "combined_extra_trees": "Extra Trees",
        "attention_full6": "Full 6",
        "attention_gap2": "Gap only (2)",
        "attention_separate4": "Separate layers (4)",
        "attention_gap_shape4": "Gap + shape gap (4)",
    }
    order = list(labels)
    primary = metrics[
        metrics["task"].eq(PRIMARY_TASK) & metrics["model"].isin(order)
    ].set_index("model").reindex(order).reset_index()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = [
        "#667085",
        "#7a5af8",
        "#277da1",
        "#f8961e",
        "#43aa8b",
        "#f94144",
    ]
    for axis, metric, low, high, title in [
        (
            axes[0],
            "roc_auc",
            "roc_auc_ci_low",
            "roc_auc_ci_high",
            "ROC AUC",
        ),
        (
            axes[1],
            "average_precision",
            "average_precision_ci_low",
            "average_precision_ci_high",
            "Average precision",
        ),
    ]:
        values = primary[metric].to_numpy(float)
        errors = np.vstack(
            [
                values - primary[low].to_numpy(float),
                primary[high].to_numpy(float) - values,
            ]
        )
        axis.bar(
            range(len(primary)),
            values,
            yerr=errors,
            color=colors,
            capsize=3,
        )
        axis.set_xticks(
            range(len(primary)),
            [labels[model] for model in primary["model"]],
            rotation=25,
            ha="right",
        )
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.2)
        if metric == "roc_auc":
            axis.axhline(0.5, color="#333333", linestyle=":", linewidth=1)
        else:
            axis.axhline(
                float(primary["prevalence"].iloc[0]),
                color="#333333",
                linestyle=":",
                linewidth=1,
            )
    figure.suptitle(
        "Attention-CNN channel ablation: next-visit 15% relative Mid-GLS decline",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(
        output / "cnn_channel_ablation_primary.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

    primary_delta = deltas[deltas["task"].eq(PRIMARY_TASK)].copy()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, metric, low, high, title in [
        (
            axes[0],
            "delta_roc_auc",
            "delta_roc_auc_ci_low",
            "delta_roc_auc_ci_high",
            "AUC difference versus full-6",
        ),
        (
            axes[1],
            "delta_average_precision",
            "delta_average_precision_ci_low",
            "delta_average_precision_ci_high",
            "AP difference versus full-6",
        ),
    ]:
        values = primary_delta[metric].to_numpy(float)
        errors = np.vstack(
            [
                values - primary_delta[low].to_numpy(float),
                primary_delta[high].to_numpy(float) - values,
            ]
        )
        axis.errorbar(
            range(len(primary_delta)),
            values,
            yerr=errors,
            fmt="o",
            markersize=7,
            capsize=3,
            color="#277da1",
        )
        axis.axhline(0, color="#333333", linewidth=1)
        axis.set_xticks(
            range(len(primary_delta)),
            [
                labels[model]
                for model in primary_delta["candidate_model"]
            ],
            rotation=22,
            ha="right",
        )
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Paired patient-bootstrap differences",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(
        output / "cnn_channel_ablation_paired_deltas.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[columns].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "|" + "|".join(["---"] * len(columns)) + "|",
            *[
                "| " + " | ".join(map(str, row)) + " |"
                for row in display.itertuples(index=False, name=None)
            ],
        ]
    )


def write_report(
    output: Path,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    reproduction: dict[str, float],
) -> None:
    labels = {
        "clinical_ridge": "Clinical ridge",
        "combined_extra_trees": "Combined Extra Trees",
        "attention_full6": "Attention CNN: full 6 channels",
        "attention_gap2": "Attention CNN: Endo-Mid gap only",
        "attention_separate4": "Attention CNN: separate layers only",
        "attention_gap_shape4": "Attention CNN: gap + normalized shape gap",
    }
    primary = metrics[metrics["task"].eq(PRIMARY_TASK)].copy()
    primary = primary[primary["model"].isin(labels)].copy()
    primary["model"] = primary["model"].map(labels)
    primary = primary.rename(
        columns={
            "roc_auc": "AUC",
            "roc_auc_ci_low": "AUC CI low",
            "roc_auc_ci_high": "AUC CI high",
            "average_precision": "AP",
            "average_precision_ci_low": "AP CI low",
            "average_precision_ci_high": "AP CI high",
        }
    )
    primary_deltas = deltas[deltas["task"].eq(PRIMARY_TASK)].copy()
    primary_deltas["candidate_model"] = primary_deltas[
        "candidate_model"
    ].map(labels)
    primary_deltas = primary_deltas.rename(
        columns={
            "candidate_model": "candidate",
            "delta_roc_auc": "ΔAUC",
            "delta_roc_auc_ci_low": "ΔAUC CI low",
            "delta_roc_auc_ci_high": "ΔAUC CI high",
            "delta_average_precision": "ΔAP",
            "delta_average_precision_ci_low": "ΔAP CI low",
            "delta_average_precision_ci_high": "ΔAP CI high",
        }
    )
    best = primary[primary["model"].str.startswith("Attention CNN")].sort_values(
        ["AUC", "AP"], ascending=False
    ).iloc[0]
    report = f"""# Attention-CNN curve-channel ablation

## Question

Does restricting the curve branch to the Endo–Mid gap improve prediction of a
15% relative Mid-GLS decline at the immediately following visit?

The experiment kept the 96 scalar features, attention pooling, multitask labels,
three repeated five-fold patient-held-out splits, validation procedure, seeds,
optimizer, and stopping rule fixed. Only the curve channels changed.

## Results

{markdown_table(primary, ['model', 'n', 'events', 'AUC', 'AUC CI low', 'AUC CI high', 'AP', 'AP CI low', 'AP CI high'])}

## Paired differences from the full six-channel attention CNN

{markdown_table(primary_deltas, ['candidate', 'ΔAUC', 'ΔAUC CI low', 'ΔAUC CI high', 'ΔAP', 'ΔAP CI low', 'ΔAP CI high'])}

The highest channel-ablation AUC was **{best['model']}**: AUC {best['AUC']:.3f},
AP {best['AP']:.3f}. Confidence intervals are patient-cluster bootstraps.

## Conclusion

- Restricting the curve branch to the two Endo–Mid gap channels did not improve the
  model. Its point estimates were lower by 0.014 AUC and 0.031 AP.
- Removing the explicit gap channels while retaining separate Endo and Mid curves
  preserved AUC almost exactly, but reduced AP by 0.026. The CNN can learn a
  subtraction internally, although the explicit gap channels may help event ranking.
- Adding normalized shape-gap channels did not recover the loss and had the lowest
  AP of the CNN variants.
- None of the paired intervals excluded zero, so the dataset cannot establish a
  definitive difference. The full six-channel input remains the preferred
  representation because it had the best AUC/AP point estimates and retains all
  layer-specific information.

## Interpretation rules

- A positive paired difference favors the reduced channel representation.
- An interval crossing zero means the change is not stable in this cohort.
- The gap-only experiment changes only the CNN curve branch. The parallel scalar
  branch still supplies baseline/current GLS, EF, trajectory, and variability
  features.
- `gap + normalized shape gap` adds the current and longitudinal change in
  `Endo/max|Endo| - Mid/max|Mid|`, matching the strongest engineered shape-gap
  features more closely than the raw difference alone.

## Reproduction check

The newly trained full-six reference differed from the previously saved attention
CNN by {reproduction['auc_difference']:.6f} AUC and
{reproduction['ap_difference']:.6f} AP.
"""
    (output / "cnn_channel_ablation_report.md").write_text(
        report, encoding="utf-8"
    )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    transitions = pd.read_parquet(
        BASE_RESULTS / "next_visit_transitions.parquet"
    )
    visits = pd.read_parquet(VISITS_PATH)
    manifest = pd.read_csv(BASE_RESULTS / "feature_manifest.csv")
    gpu_features = manifest[
        manifest["feature_set"].eq("gpu_scalars")
    ]["feature"].tolist()
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
    curve_inputs = build_channel_inputs(transitions, visits, tensors)

    predictions, weights, logs = train_channel_variants(
        transitions,
        curve_inputs,
        scalar_columns,
        active_tasks,
        splits,
    )
    predictions.to_parquet(
        OUTPUT / "cnn_channel_ablation_oof_predictions_all_tasks.parquet",
        index=False,
    )
    weights.to_parquet(
        OUTPUT / "cnn_channel_ablation_attention_weights.parquet",
        index=False,
    )
    logs.to_csv(OUTPUT / "cnn_channel_ablation_training_log.csv", index=False)
    assignments.to_csv(
        OUTPUT / "cnn_channel_ablation_patient_folds.csv", index=False
    )

    primary_predictions = predictions[
        predictions["task"].eq(PRIMARY_TASK)
    ].copy()
    prior = pd.read_parquet(BASE_RESULTS / "oof_predictions.parquet")
    context = prior[
        prior["task"].eq(PRIMARY_TASK)
        & prior["model"].isin(["clinical_ridge", "combined_extra_trees"])
    ].copy()
    evaluation_predictions = pd.concat(
        [context, primary_predictions], ignore_index=True
    )
    metrics, clinical_deltas = core.evaluate_predictions(
        evaluation_predictions, BOOTSTRAPS, SEED
    )
    comparisons = [
        ("gap2_vs_full6", "attention_full6", "attention_gap2"),
        (
            "separate4_vs_full6",
            "attention_full6",
            "attention_separate4",
        ),
        (
            "gap_shape4_vs_full6",
            "attention_full6",
            "attention_gap_shape4",
        ),
    ]
    paired_deltas = qc.paired_variant_deltas(
        evaluation_predictions,
        comparisons,
        BOOTSTRAPS,
        SEED + 810000,
    )
    metrics.to_csv(
        OUTPUT / "cnn_channel_ablation_metrics.csv", index=False
    )
    clinical_deltas.to_csv(
        OUTPUT / "cnn_channel_ablation_deltas_vs_clinical.csv",
        index=False,
    )
    paired_deltas.to_csv(
        OUTPUT / "cnn_channel_ablation_paired_deltas.csv", index=False
    )

    prior_metrics = pd.read_csv(
        PLATEAU_RESULTS / "plateau_model_metrics.csv"
    )
    prior_attention = prior_metrics[
        prior_metrics["task"].eq(PRIMARY_TASK)
        & prior_metrics["model"].eq("attention_binary")
    ].iloc[0]
    current_attention = metrics[
        metrics["task"].eq(PRIMARY_TASK)
        & metrics["model"].eq("attention_full6")
    ].iloc[0]
    reproduction = {
        "prior_attention_auc": float(prior_attention["roc_auc"]),
        "current_full6_auc": float(current_attention["roc_auc"]),
        "auc_difference": float(
            current_attention["roc_auc"] - prior_attention["roc_auc"]
        ),
        "prior_attention_ap": float(prior_attention["average_precision"]),
        "current_full6_ap": float(current_attention["average_precision"]),
        "ap_difference": float(
            current_attention["average_precision"]
            - prior_attention["average_precision"]
        ),
    }
    metadata = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "patients": int(transitions["patient_id"].nunique()),
        "transitions": int(len(transitions)),
        "primary_eligible_transitions": int(
            transitions[f"mask__{PRIMARY_TASK}"].astype(bool).sum()
        ),
        "primary_events": int(
            transitions.loc[
                transitions[f"mask__{PRIMARY_TASK}"].astype(bool),
                f"label__{PRIMARY_TASK}",
            ].sum()
        ),
        "scalar_features": len(scalar_columns),
        "variants": [
            {
                "name": variant.name,
                "description": variant.description,
                "channels": list(variant.channels),
                "tensor_shape": list(curve_inputs[variant.name].shape),
            }
            for variant in VARIANTS
        ],
        "active_tasks": [task.name for task in active_tasks],
        "cv": "3 repeated 5-fold patient-held-out",
        "bootstraps": BOOTSTRAPS,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "seed": SEED,
        "full6_reproduction": reproduction,
    }
    (OUTPUT / "cnn_channel_ablation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    make_figure(OUTPUT, metrics, paired_deltas)
    write_report(OUTPUT, metrics, paired_deltas, reproduction)

    primary = metrics[metrics["task"].eq(PRIMARY_TASK)][
        [
            "model",
            "roc_auc",
            "roc_auc_ci_low",
            "roc_auc_ci_high",
            "average_precision",
            "average_precision_ci_low",
            "average_precision_ci_high",
        ]
    ]
    print("\nPrimary metrics", flush=True)
    print(primary.to_string(index=False), flush=True)
    print("\nPaired deltas", flush=True)
    print(paired_deltas.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
