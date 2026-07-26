from __future__ import annotations

import copy
import json
import math
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch import nn

import cardiotoxicity_next_visit_gpu as core
import cardiotoxicity_nonapical_qc as qc


VISITS_PATH = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet")
CURVES_PATH = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_dataset.parquet")
BASE_RESULTS = Path(r"D:\us\cardiotoxicity_next_visit_gpu_results")
OUTPUT = Path(r"D:\us\cardiotoxicity_plateau_results")
SEED = 20260722
AUXILIARY_WEIGHT = 0.25
BOOTSTRAPS = 500


@dataclass(frozen=True)
class Variant:
    name: str
    attention: bool
    auxiliary: bool


VARIANTS = (
    Variant("uniform_binary", False, False),
    Variant("attention_binary", True, False),
    Variant("uniform_gls_aux", False, True),
    Variant("attention_gls_aux", True, True),
)


SEGMENT_NAMES = {
    1: "basal anterior",
    2: "basal anteroseptal",
    3: "basal inferoseptal",
    4: "basal inferior",
    5: "basal inferolateral",
    6: "basal anterolateral",
    7: "mid anterior",
    8: "mid anteroseptal",
    9: "mid inferoseptal",
    10: "mid inferior",
    11: "mid inferolateral",
    12: "mid anterolateral",
    13: "apical anterior",
    14: "apical anteroseptal",
    15: "apical inferoseptal",
    16: "apical inferior",
    17: "apical inferolateral",
    18: "apical anterolateral",
}


class PlateauNet(nn.Module):
    def __init__(
        self,
        scalar_features: int,
        tasks: int,
        segments: int,
        attention: bool,
        auxiliary: bool,
    ):
        super().__init__()
        self.attention_enabled = attention
        self.auxiliary_enabled = auxiliary
        self.segment_encoder = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 24, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        if attention:
            self.segment_embedding = nn.Parameter(torch.zeros(1, segments, 24))
            nn.init.normal_(self.segment_embedding, mean=0.0, std=0.02)
            self.attention_score = nn.Sequential(
                nn.Linear(24, 12),
                nn.Tanh(),
                nn.Linear(12, 1),
            )
        else:
            self.register_parameter("segment_embedding", None)
            self.attention_score = None
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
        self.regression_head = nn.Linear(64, 1) if auxiliary else None

    def forward(
        self, curves: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        batch, segments, channels, time_points = curves.shape
        encoded = self.segment_encoder(
            curves.reshape(batch * segments, channels, time_points)
        ).squeeze(-1)
        encoded = encoded.reshape(batch, segments, -1)
        if self.attention_enabled:
            scored = encoded + self.segment_embedding[:, :segments]
            weights = torch.softmax(self.attention_score(scored).squeeze(-1), dim=1)
            center = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        else:
            weights = torch.full(
                (batch, segments),
                1.0 / segments,
                dtype=encoded.dtype,
                device=encoded.device,
            )
            center = encoded.mean(dim=1)
        pooled = torch.cat(
            [center, encoded.std(dim=1), encoded.max(dim=1).values], dim=1
        )
        scalar_embedding = self.scalar_encoder(scalars)
        shared = self.shared(torch.cat([pooled, scalar_embedding], dim=1))
        binary = self.binary_head(shared)
        regression = self.regression_head(shared).squeeze(-1) if self.regression_head else None
        return binary, regression, weights


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def transition_curve_inputs(
    transitions: pd.DataFrame,
    visits: pd.DataFrame,
    tensors: dict[str, np.ndarray],
) -> np.ndarray:
    visit_order = {
        (str(row.patient_id), int(row.visit_order)): str(row.visit_id)
        for row in visits[["patient_id", "visit_order", "visit_id"]].itertuples(index=False)
    }
    result = []
    for row in transitions.itertuples(index=False):
        current_id = str(row.current_visit_id)
        order = int(row.current_visit_order)
        previous_id = current_id if order == 1 else visit_order[(str(row.patient_id), order - 1)]
        current = tensors[current_id]
        previous = tensors[previous_id]
        endo = current[:, 0]
        mid = current[:, 1]
        delta_endo = endo - previous[:, 0] if order > 1 else np.zeros_like(endo)
        delta_mid = mid - previous[:, 1] if order > 1 else np.zeros_like(mid)
        value = np.stack(
            [endo, mid, endo - mid, delta_endo, delta_mid, delta_endo - delta_mid], axis=1
        ).astype(np.float32)
        result.append(np.clip(value / 30.0, -2.0, 2.0))
    return np.stack(result)


def immediate_gls_change(transitions: pd.DataFrame, visits: pd.DataFrame) -> np.ndarray:
    mid = visits.set_index("visit_id")["gls_mid_peak_avg"].abs().astype(float)
    target = transitions["target_visit_id"].map(mid).to_numpy(float)
    current = transitions["current_visit_id"].map(mid).to_numpy(float)
    return np.clip(1.0 - target / current, -0.60, 0.60).astype(np.float32)


def binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight, reduction="none"
    )
    return (loss * masks).sum() / masks.sum().clamp_min(1.0)


def train_variants(
    transitions: pd.DataFrame,
    curve_inputs: np.ndarray,
    scalar_columns: list[str],
    active_tasks: list[core.TaskSpec],
    splits: list[dict[str, object]],
    aux_target: np.ndarray,
    epochs: int = 180,
    patience: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    labels = np.column_stack(
        [transitions[f"label__{task.name}"].to_numpy(np.float32) for task in active_tasks]
    )
    masks = np.column_stack(
        [transitions[f"mask__{task.name}"].to_numpy(np.float32) for task in active_tasks]
    )
    patient_primary = (
        transitions.groupby("patient_id")["label__mid_first_rel15"].max().astype(int).to_dict()
    )
    score_sum = {
        variant.name: np.zeros((len(transitions), len(active_tasks)), dtype=float)
        for variant in VARIANTS
    }
    score_count = {
        variant.name: np.zeros((len(transitions), len(active_tasks)), dtype=int)
        for variant in VARIANTS
    }
    regression_sum = {
        variant.name: np.zeros(len(transitions), dtype=float)
        for variant in VARIANTS
        if variant.auxiliary
    }
    regression_count = {
        variant.name: np.zeros(len(transitions), dtype=int)
        for variant in VARIANTS
        if variant.auxiliary
    }
    weight_sum = {
        variant.name: np.zeros((len(transitions), curve_inputs.shape[1]), dtype=float)
        for variant in VARIANTS
        if variant.attention
    }
    weight_count = {
        variant.name: np.zeros(len(transitions), dtype=int)
        for variant in VARIANTS
        if variant.attention
    }
    logs = []

    for split in splits:
        split_seed = SEED + int(split["split_index"]) * 101
        train_patients = sorted(split["train_patients"])
        patient_labels = [patient_primary.get(patient, 0) for patient in train_patients]
        fit_patients, val_patients = train_test_split(
            train_patients,
            test_size=0.2,
            random_state=split_seed,
            stratify=patient_labels,
        )
        fit_index = np.flatnonzero(transitions["patient_id"].isin(fit_patients).to_numpy())
        val_index = np.flatnonzero(transitions["patient_id"].isin(val_patients).to_numpy())
        test_index = np.flatnonzero(
            transitions["patient_id"].isin(split["test_patients"]).to_numpy()
        )
        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        fit_scalars = imputer.fit_transform(
            transitions.iloc[fit_index][scalar_columns].astype(float)
        )
        fit_scalars = np.clip(scaler.fit_transform(fit_scalars), -5.0, 5.0).astype(np.float32)
        val_scalars = np.clip(
            scaler.transform(
                imputer.transform(transitions.iloc[val_index][scalar_columns].astype(float))
            ),
            -5.0,
            5.0,
        ).astype(np.float32)
        test_scalars = np.clip(
            scaler.transform(
                imputer.transform(transitions.iloc[test_index][scalar_columns].astype(float))
            ),
            -5.0,
            5.0,
        ).astype(np.float32)
        aux_mean = float(np.mean(aux_target[fit_index]))
        aux_sd = float(np.std(aux_target[fit_index], ddof=1))
        aux_sd = max(aux_sd, 1e-4)
        aux_standard = (aux_target - aux_mean) / aux_sd

        fit_masks = masks[fit_index].copy()
        for task_index in range(len(active_tasks)):
            task_y = labels[fit_index][fit_masks[:, task_index] > 0, task_index]
            if task_y.sum() < 2 or (len(task_y) - task_y.sum()) < 2:
                fit_masks[:, task_index] = 0
        positives = (labels[fit_index] * fit_masks).sum(axis=0)
        negatives = ((1.0 - labels[fit_index]) * fit_masks).sum(axis=0)
        pos_weights = np.clip(
            negatives / np.maximum(positives, 1.0), 1.0, 20.0
        ).astype(np.float32)

        for variant in VARIANTS:
            set_seed(split_seed)
            model = PlateauNet(
                len(scalar_columns),
                len(active_tasks),
                curve_inputs.shape[1],
                variant.attention,
                variant.auxiliary,
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-3)
            amp_scaler = torch.amp.GradScaler("cuda", enabled=True)
            best_state = copy.deepcopy(model.state_dict())
            best_loss = math.inf
            best_epoch = 0
            stale = 0
            start = time.time()
            batch_size = 64
            for epoch in range(epochs):
                model.train()
                order = np.random.default_rng(split_seed + epoch).permutation(len(fit_index))
                for start_index in range(0, len(order), batch_size):
                    local = order[start_index : start_index + batch_size]
                    global_index = fit_index[local]
                    batch_curves = torch.as_tensor(curve_inputs[global_index], device=device)
                    batch_scalars = torch.as_tensor(fit_scalars[local], device=device)
                    batch_labels = torch.as_tensor(labels[global_index], device=device)
                    batch_masks = torch.as_tensor(fit_masks[local], device=device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type="cuda", enabled=True):
                        logits, regression, _ = model(batch_curves, batch_scalars)
                        loss = binary_loss(
                            logits,
                            batch_labels,
                            batch_masks,
                            torch.as_tensor(pos_weights, device=device),
                        )
                        if variant.auxiliary:
                            aux_batch = torch.as_tensor(aux_standard[global_index], device=device)
                            loss = loss + AUXILIARY_WEIGHT * torch.nn.functional.smooth_l1_loss(
                                regression, aux_batch
                            )
                    amp_scaler.scale(loss).backward()
                    amp_scaler.step(optimizer)
                    amp_scaler.update()

                model.eval()
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True):
                    logits, regression, _ = model(
                        torch.as_tensor(curve_inputs[val_index], device=device),
                        torch.as_tensor(val_scalars, device=device),
                    )
                    val_loss = binary_loss(
                        logits,
                        torch.as_tensor(labels[val_index], device=device),
                        torch.as_tensor(masks[val_index], device=device),
                        torch.as_tensor(pos_weights, device=device),
                    )
                    if variant.auxiliary:
                        aux_val = torch.as_tensor(aux_standard[val_index], device=device)
                        val_loss = val_loss + AUXILIARY_WEIGHT * torch.nn.functional.smooth_l1_loss(
                            regression, aux_val
                        )
                    val_loss_value = float(val_loss.item())
                if val_loss_value < best_loss - 1e-4:
                    best_loss = val_loss_value
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                if stale >= patience:
                    break

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True):
                logits, regression, attention = model(
                    torch.as_tensor(curve_inputs[test_index], device=device),
                    torch.as_tensor(test_scalars, device=device),
                )
                probabilities = torch.sigmoid(logits).float().cpu().numpy()
                attention_np = attention.float().cpu().numpy()
                regression_np = (
                    regression.float().cpu().numpy() * aux_sd + aux_mean
                    if variant.auxiliary
                    else None
                )
            for task_index in range(len(active_tasks)):
                valid = masks[test_index, task_index] > 0
                global_valid = test_index[valid]
                score_sum[variant.name][global_valid, task_index] += probabilities[valid, task_index]
                score_count[variant.name][global_valid, task_index] += 1
            if variant.auxiliary:
                regression_sum[variant.name][test_index] += regression_np
                regression_count[variant.name][test_index] += 1
            if variant.attention:
                weight_sum[variant.name][test_index] += attention_np
                weight_count[variant.name][test_index] += 1
            logs.append(
                {
                    "variant": variant.name,
                    "repeat": split["repeat"],
                    "fold": split["fold"],
                    "device": torch.cuda.get_device_name(0),
                    "best_epoch": best_epoch,
                    "epochs_run": epoch + 1,
                    "best_validation_loss": best_loss,
                    "seconds": time.time() - start,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                }
            )
            del model
            torch.cuda.empty_cache()

    prediction_rows = []
    for variant in VARIANTS:
        for task_index, task in enumerate(active_tasks):
            valid = (masks[:, task_index] > 0) & (score_count[variant.name][:, task_index] > 0)
            for index in np.flatnonzero(valid):
                prediction_rows.append(
                    {
                        "task": task.name,
                        "model": variant.name,
                        "transition_id": transitions.iloc[index]["transition_id"],
                        "patient_id": transitions.iloc[index]["patient_id"],
                        "label": int(labels[index, task_index]),
                        "score": float(
                            score_sum[variant.name][index, task_index]
                            / score_count[variant.name][index, task_index]
                        ),
                        "prediction_repeats": int(score_count[variant.name][index, task_index]),
                    }
                )
    regression_rows = []
    for variant in VARIANTS:
        if not variant.auxiliary:
            continue
        valid = regression_count[variant.name] > 0
        for index in np.flatnonzero(valid):
            regression_rows.append(
                {
                    "variant": variant.name,
                    "transition_id": transitions.iloc[index]["transition_id"],
                    "patient_id": transitions.iloc[index]["patient_id"],
                    "observed_next_mid_relative_change": float(aux_target[index]),
                    "predicted_next_mid_relative_change": float(
                        regression_sum[variant.name][index] / regression_count[variant.name][index]
                    ),
                    "prediction_repeats": int(regression_count[variant.name][index]),
                }
            )
    weight_rows = []
    primary_mask = transitions["mask__mid_first_rel15"].astype(bool).to_numpy()
    primary_label = transitions["label__mid_first_rel15"].to_numpy(int)
    for variant in VARIANTS:
        if not variant.attention:
            continue
        valid = weight_count[variant.name] > 0
        for index in np.flatnonzero(valid):
            values = weight_sum[variant.name][index] / weight_count[variant.name][index]
            for segment_index, value in enumerate(values, start=1):
                weight_rows.append(
                    {
                        "variant": variant.name,
                        "transition_id": transitions.iloc[index]["transition_id"],
                        "patient_id": transitions.iloc[index]["patient_id"],
                        "primary_eligible": bool(primary_mask[index]),
                        "primary_label": int(primary_label[index]),
                        "segment": segment_index,
                        "segment_name": SEGMENT_NAMES[segment_index],
                        "attention_weight": float(value),
                        "weight_repeats": int(weight_count[variant.name][index]),
                    }
                )
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(regression_rows),
        pd.DataFrame(weight_rows),
        pd.DataFrame(logs),
    )


def regression_metrics(predictions: pd.DataFrame, bootstraps: int = BOOTSTRAPS) -> pd.DataFrame:
    rows = []
    for variant_index, (variant, group) in enumerate(predictions.groupby("variant")):
        observed = group["observed_next_mid_relative_change"].to_numpy(float)
        predicted = group["predicted_next_mid_relative_change"].to_numpy(float)
        patient = group["patient_id"].to_numpy(str)
        result = {
            "variant": variant,
            "n": len(group),
            "mae": mean_absolute_error(observed, predicted),
            "rmse": math.sqrt(mean_squared_error(observed, predicted)),
            "r2": r2_score(observed, predicted),
            "pearson_r": pearsonr(observed, predicted).statistic,
            "spearman_r": spearmanr(observed, predicted).statistic,
        }
        rng = np.random.default_rng(SEED + 400000 + variant_index)
        samples = {key: [] for key in ("mae", "pearson_r", "spearman_r")}
        for _ in range(bootstraps):
            index = core.cluster_sample_indices(patient, rng)
            samples["mae"].append(mean_absolute_error(observed[index], predicted[index]))
            if np.std(observed[index]) > 1e-9 and np.std(predicted[index]) > 1e-9:
                samples["pearson_r"].append(pearsonr(observed[index], predicted[index]).statistic)
                samples["spearman_r"].append(spearmanr(observed[index], predicted[index]).statistic)
        for metric, values in samples.items():
            result[f"{metric}_ci_low"] = float(np.quantile(values, 0.025))
            result[f"{metric}_ci_high"] = float(np.quantile(values, 0.975))
        rows.append(result)
    return pd.DataFrame(rows)


def attention_summary(weights: pd.DataFrame, bootstraps: int = BOOTSTRAPS) -> pd.DataFrame:
    eligible = weights[weights["primary_eligible"]].copy()
    rows = []
    for variant_index, (variant, variant_frame) in enumerate(eligible.groupby("variant")):
        wide = variant_frame.pivot_table(
            index=["transition_id", "patient_id", "primary_label"],
            columns="segment",
            values="attention_weight",
        ).reset_index()
        patient = wide["patient_id"].to_numpy(str)
        label = wide["primary_label"].to_numpy(int)
        rng = np.random.default_rng(SEED + 500000 + variant_index)
        for segment in range(1, 19):
            values = wide[segment].to_numpy(float)
            observed_all = float(np.mean(values))
            observed_event = float(np.mean(values[label == 1]))
            observed_nonevent = float(np.mean(values[label == 0]))
            bootstrap_all = []
            bootstrap_difference = []
            for _ in range(bootstraps):
                index = core.cluster_sample_indices(patient, rng)
                bootstrap_all.append(float(np.mean(values[index])))
                y = label[index]
                if y.sum() and (y == 0).sum():
                    bootstrap_difference.append(
                        float(np.mean(values[index][y == 1]) - np.mean(values[index][y == 0]))
                    )
            rows.append(
                {
                    "variant": variant,
                    "segment": segment,
                    "segment_name": SEGMENT_NAMES[segment],
                    "mean_weight": observed_all,
                    "mean_weight_ci_low": float(np.quantile(bootstrap_all, 0.025)),
                    "mean_weight_ci_high": float(np.quantile(bootstrap_all, 0.975)),
                    "event_mean_weight": observed_event,
                    "nonevent_mean_weight": observed_nonevent,
                    "event_minus_nonevent": observed_event - observed_nonevent,
                    "difference_ci_low": float(np.quantile(bootstrap_difference, 0.025)),
                    "difference_ci_high": float(np.quantile(bootstrap_difference, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    transitions = pd.read_parquet(BASE_RESULTS / "next_visit_transitions.parquet")
    visits = pd.read_parquet(VISITS_PATH)
    manifest = pd.read_csv(BASE_RESULTS / "feature_manifest.csv")
    gpu_features = manifest[manifest["feature_set"] == "gpu_scalars"]["feature"].tolist()
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
    curves = transition_curve_inputs(transitions, visits, tensors)
    auxiliary = immediate_gls_change(transitions, visits)
    predictions, regression, weights, logs = train_variants(
        transitions,
        curves,
        scalar_columns,
        active_tasks,
        splits,
        auxiliary,
    )

    predictions.to_parquet(OUTPUT / "ablation_gpu_predictions_raw.parquet", index=False)
    regression.to_parquet(OUTPUT / "auxiliary_gls_oof_predictions.parquet", index=False)
    weights.to_parquet(OUTPUT / "oof_attention_weights.parquet", index=False)
    logs.to_csv(OUTPUT / "plateau_gpu_training_log.csv", index=False)
    print("GPU predictions serialized", flush=True)

    prior = pd.read_parquet(BASE_RESULTS / "oof_predictions.parquet")
    mid_names = {task.name for task in active_tasks if task.name.startswith("mid_")}
    context = prior[
        prior["model"].isin(["clinical_ridge"])
        & prior["task"].isin(mid_names)
    ].copy()
    context["model"] = context["model"].replace(
        {
            "combined_extra_trees": "existing_combined_trees",
            "gpu_segment_curve_net": "existing_gpu",
        }
    )
    selected_predictions = predictions[predictions["task"].isin(mid_names)].copy()
    all_predictions = pd.concat([context, selected_predictions], ignore_index=True)
    metrics, clinical_deltas = core.evaluate_predictions(all_predictions, BOOTSTRAPS, SEED)
    comparisons = [
        ("attention_vs_uniform", "uniform_binary", "attention_binary"),
        ("uniform_aux_vs_binary", "uniform_binary", "uniform_gls_aux"),
        ("attention_aux_vs_binary", "attention_binary", "attention_gls_aux"),
        ("attention_aux_vs_uniform", "uniform_binary", "attention_gls_aux"),
    ]
    ablation_deltas = qc.paired_variant_deltas(
        all_predictions, comparisons, BOOTSTRAPS, SEED + 600000
    )
    regression_summary = regression_metrics(regression)
    weight_summary = attention_summary(weights)

    all_predictions.to_parquet(OUTPUT / "plateau_oof_predictions.parquet", index=False)
    metrics.to_csv(OUTPUT / "plateau_model_metrics.csv", index=False)
    clinical_deltas.to_csv(OUTPUT / "plateau_deltas_vs_clinical.csv", index=False)
    ablation_deltas.to_csv(OUTPUT / "plateau_ablation_deltas.csv", index=False)
    regression_summary.to_csv(OUTPUT / "auxiliary_gls_metrics.csv", index=False)
    weight_summary.to_csv(OUTPUT / "attention_weight_summary.csv", index=False)
    assignments.to_csv(OUTPUT / "patient_fold_assignments.csv", index=False)
    metadata = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "patients": int(transitions["patient_id"].nunique()),
        "transitions": int(len(transitions)),
        "curve_input": list(curves.shape),
        "scalar_features": len(scalar_columns),
        "active_tasks": [task.name for task in active_tasks],
        "variants": [variant.__dict__ for variant in VARIANTS],
        "auxiliary_target": "relative Mid-GLS change from current to immediately next visit",
        "auxiliary_loss": "0.25 * standardized SmoothL1 added to masked binary BCE",
        "cv": "3 repeated 5-fold patient-held-out",
        "bootstraps": BOOTSTRAPS,
        "seed": SEED,
    }
    (OUTPUT / "plateau_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(metrics[metrics["task"] == "mid_first_rel15"].sort_values("average_precision").to_string(index=False))
    print(ablation_deltas[ablation_deltas["task"] == "mid_first_rel15"].to_string(index=False))
    print(regression_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
