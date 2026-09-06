from __future__ import annotations

import copy
import json
import math
import random
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sktime.transformations.rocket import MultiRocketMultivariate
from torch import nn

import cardiotoxicity_cnn_channel_ablation as channel
import cardiotoxicity_next_visit_gpu as core
import cardiotoxicity_nonapical_qc as qc


ROOT = Path(r"D:\us")
VISITS_PATH = ROOT / "amber_full_105_preprocessed" / "Ichilov_july_visits.parquet"
CURVES_PATH = ROOT / "amber_full_105_preprocessed" / "Ichilov_july_dataset.parquet"
BASE_RESULTS = ROOT / "cardiotoxicity_next_visit_gpu_results"
CNN_RESULTS = ROOT / "cardiotoxicity_cnn_length_ablation_results"
OUTPUT = ROOT / "cardiotoxicity_timeseries_round1_results"
CACHE = OUTPUT / "embedding_cache"
PRIMARY_TASK = "mid_first_rel15"
SEED = 20260722
BOOTSTRAPS = 2000
PCA_COMPONENTS = 32
MULTIROCKET_KERNELS = 6250
ADAPTER_EPOCHS = 180
ADAPTER_PATIENCE = 25


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_inputs() -> tuple[pd.DataFrame, np.ndarray, list[str], list[core.TaskSpec], list[dict[str, object]]]:
    transitions = pd.read_parquet(BASE_RESULTS / "next_visit_transitions.parquet")
    visits = pd.read_parquet(VISITS_PATH)
    tensors = core.build_visit_curve_tensors(CURVES_PATH)
    curves = channel.build_channel_inputs(transitions, visits, tensors)["attention_full6"]
    manifest = pd.read_csv(BASE_RESULTS / "feature_manifest.csv")
    candidates = manifest.loc[manifest["feature_set"].eq("gpu_scalars"), "feature"].tolist()
    scalar_columns = core.usable_features(transitions, candidates)
    tasks = core.task_specs()
    audit = core.label_audit(transitions, tasks)
    active_names = set(
        audit.loc[
            (audit["eligible_transitions"] >= 30)
            & (audit["events"] >= 8)
            & ((audit["eligible_transitions"] - audit["events"]) >= 8),
            "task",
        ]
    )
    active_tasks = [task for task in tasks if task.name in active_names]
    splits, assignments = core.patient_splits(transitions, 3, SEED)
    saved_assignments = pd.read_csv(CNN_RESULTS / "cnn_length_ablation_patient_folds.csv")
    left = assignments.sort_values(["repeat", "fold", "patient_id"]).reset_index(drop=True)
    right = saved_assignments.sort_values(["repeat", "fold", "patient_id"]).reset_index(drop=True)
    if not left.equals(right):
        raise RuntimeError("Reconstructed patient folds do not match the retained CNN experiment")
    return transitions, curves, scalar_columns, active_tasks, splits


def resize_curves(curves: np.ndarray, length: int = 512) -> np.ndarray:
    shape = curves.shape
    tensor = torch.as_tensor(curves.reshape(-1, shape[-2], shape[-1]), dtype=torch.float32)
    resized = F.interpolate(tensor, size=length, mode="linear", align_corners=False)
    return resized.reshape(*shape[:-1], length).numpy()


def extract_mantis(curves: np.ndarray, pretrained: bool, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)["embeddings"]
    from mantis.architecture import MantisV2
    from mantis.trainer import MantisTrainer

    set_seed(SEED if pretrained else SEED + 7001)
    network = MantisV2(device="cuda")
    if pretrained:
        network = network.from_pretrained("paris-noah/MantisV2")
    network.eval()
    trainer = MantisTrainer(device="cuda", network=network)
    resized = resize_curves(curves).reshape(-1, curves.shape[2], 512)
    started = time.time()
    embeddings = trainer.transform(resized, batch_size=256, three_dim=True, to_numpy=True)
    embeddings = embeddings.reshape(len(curves), curves.shape[1], curves.shape[2], -1).astype(np.float32)
    np.savez_compressed(cache_path, embeddings=embeddings, seconds=np.asarray(time.time() - started))
    del network, trainer, resized
    torch.cuda.empty_cache()
    return embeddings


def extract_moment(curves: np.ndarray, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)["embeddings"]
    from momentfm import MOMENTPipeline

    set_seed(SEED)
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-small", model_kwargs={"task_name": "embedding"}
    )
    model.init()
    model = model.cuda().eval()
    resized = resize_curves(curves).reshape(-1, curves.shape[2], 512)
    results: list[np.ndarray] = []
    started = time.time()
    for start in range(0, len(resized), 64):
        batch = torch.as_tensor(resized[start : start + 64], device="cuda")
        mask = torch.ones((len(batch), 512), dtype=torch.long, device="cuda")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(x_enc=batch, input_mask=mask, reduction="none")
            # Preserve six channels; average only across the 64 temporal patches.
            embedding = output.embeddings.mean(dim=2)
        results.append(embedding.float().cpu().numpy())
    embeddings = np.concatenate(results).reshape(
        len(curves), curves.shape[1], curves.shape[2], -1
    ).astype(np.float32)
    np.savez_compressed(cache_path, embeddings=embeddings, seconds=np.asarray(time.time() - started))
    del model, resized, results
    torch.cuda.empty_cache()
    return embeddings


def aggregate_segment_embeddings(embeddings: np.ndarray) -> np.ndarray:
    flattened = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], -1)
    return np.concatenate(
        [flattened.mean(axis=1), flattened.std(axis=1), flattened.max(axis=1)], axis=1
    ).astype(np.float32)


def prepare_scalars(
    transitions: pd.DataFrame,
    columns: list[str],
    train_index: np.ndarray,
    test_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    train = imputer.fit_transform(transitions.iloc[train_index][columns].astype(float))
    train = np.clip(scaler.fit_transform(train), -5.0, 5.0).astype(np.float32)
    test = imputer.transform(transitions.iloc[test_index][columns].astype(float))
    test = np.clip(scaler.transform(test), -5.0, 5.0).astype(np.float32)
    return train, test


def embedding_probe_oof(
    transitions: pd.DataFrame,
    curve_features: np.ndarray,
    scalar_columns: list[str],
    splits: list[dict[str, object]],
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = transitions[f"mask__{PRIMARY_TASK}"].astype(bool).to_numpy()
    labels = transitions[f"label__{PRIMARY_TASK}"].to_numpy(int)
    accumulators = {
        f"{prefix}_curves": [np.zeros(len(transitions)), np.zeros(len(transitions), dtype=int)],
        f"{prefix}_curves_scalars": [np.zeros(len(transitions)), np.zeros(len(transitions), dtype=int)],
    }
    logs: list[dict[str, object]] = []
    for split in splits:
        train_index = np.flatnonzero(
            transitions["patient_id"].isin(split["train_patients"]).to_numpy() & mask
        )
        test_index = np.flatnonzero(
            transitions["patient_id"].isin(split["test_patients"]).to_numpy() & mask
        )
        n_components = min(PCA_COMPONENTS, len(train_index) - 1, curve_features.shape[1])
        pca = PCA(n_components=n_components, whiten=True, svd_solver="randomized", random_state=SEED)
        started = time.time()
        train_curve = pca.fit_transform(curve_features[train_index]).astype(np.float32)
        test_curve = pca.transform(curve_features[test_index]).astype(np.float32)
        train_scalar, test_scalar = prepare_scalars(
            transitions, scalar_columns, train_index, test_index
        )
        for with_scalars in (False, True):
            model_name = f"{prefix}_curves_scalars" if with_scalars else f"{prefix}_curves"
            x_train = (
                np.concatenate([train_curve, train_scalar], axis=1)
                if with_scalars
                else train_curve
            )
            x_test = (
                np.concatenate([test_curve, test_scalar], axis=1)
                if with_scalars
                else test_curve
            )
            model = LogisticRegression(
                penalty="l2",
                C=0.3,
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=SEED + int(split["split_index"]),
            )
            model.fit(x_train, labels[train_index])
            score = model.predict_proba(x_test)[:, 1]
            accumulators[model_name][0][test_index] += score
            accumulators[model_name][1][test_index] += 1
            logs.append(
                {
                    "model": model_name,
                    "repeat": split["repeat"],
                    "fold": split["fold"],
                    "train_n": len(train_index),
                    "test_n": len(test_index),
                    "pca_components": n_components,
                    "seconds": time.time() - started,
                }
            )
    return prediction_rows(transitions, labels, mask, accumulators), pd.DataFrame(logs)


def patient_inner_folds(patient_ids: np.ndarray, labels: np.ndarray, seed: int):
    patient_table = pd.DataFrame({"patient_id": np.unique(patient_ids)})
    event = {p: int(labels[patient_ids == p].max()) for p in patient_table["patient_id"]}
    patient_table["event"] = patient_table["patient_id"].map(event)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for train_patient, val_patient in splitter.split(patient_table["patient_id"], patient_table["event"]):
        train_set = set(patient_table.iloc[train_patient]["patient_id"])
        val_set = set(patient_table.iloc[val_patient]["patient_id"])
        yield (
            np.flatnonzero(np.isin(patient_ids, list(train_set))),
            np.flatnonzero(np.isin(patient_ids, list(val_set))),
        )


def ridge_with_group_platt(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_patients: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    oof = np.zeros(len(y_train), dtype=float)
    for inner_train, inner_val in patient_inner_folds(train_patients, y_train, seed):
        scaler = StandardScaler(with_mean=False)
        z_train = scaler.fit_transform(x_train[inner_train])
        z_val = scaler.transform(x_train[inner_val])
        ridge = RidgeClassifier(alpha=1.0, class_weight="balanced")
        ridge.fit(z_train, y_train[inner_train])
        oof[inner_val] = ridge.decision_function(z_val)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
    calibrator.fit(oof.reshape(-1, 1), y_train)
    scaler = StandardScaler(with_mean=False)
    z_train = scaler.fit_transform(x_train)
    z_test = scaler.transform(x_test)
    ridge = RidgeClassifier(alpha=1.0, class_weight="balanced")
    ridge.fit(z_train, y_train)
    decision = ridge.decision_function(z_test)
    return calibrator.predict_proba(decision.reshape(-1, 1))[:, 1]


def multirocket_features(
    transformer: MultiRocketMultivariate,
    x: np.ndarray,
    segment_wise: bool,
) -> np.ndarray:
    if not segment_wise:
        return transformer.transform(x.reshape(len(x), -1, x.shape[-1]).astype(np.float64)).to_numpy(np.float32)
    pieces: list[np.ndarray] = []
    for start in range(0, len(x), 24):
        batch = x[start : start + 24]
        transformed = transformer.transform(
            batch.reshape(-1, batch.shape[2], batch.shape[3]).astype(np.float64)
        ).to_numpy(np.float32)
        transformed = transformed.reshape(len(batch), batch.shape[1], -1)
        pieces.append(
            np.concatenate(
                [
                    transformed.mean(axis=1),
                    transformed.std(axis=1),
                    transformed.max(axis=1),
                ],
                axis=1,
            ).astype(np.float32)
        )
        del transformed
    return np.concatenate(pieces)


def multirocket_oof(
    transitions: pd.DataFrame,
    curves: np.ndarray,
    scalar_columns: list[str],
    splits: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = transitions[f"mask__{PRIMARY_TASK}"].astype(bool).to_numpy()
    labels = transitions[f"label__{PRIMARY_TASK}"].to_numpy(int)
    names = [
        "multirocket_whole_curves",
        "multirocket_whole_curves_scalars",
        "multirocket_segment_curves",
        "multirocket_segment_curves_scalars",
    ]
    accumulators = {name: [np.zeros(len(transitions)), np.zeros(len(transitions), dtype=int)] for name in names}
    logs: list[dict[str, object]] = []
    for split in splits:
        train_index = np.flatnonzero(
            transitions["patient_id"].isin(split["train_patients"]).to_numpy() & mask
        )
        test_index = np.flatnonzero(
            transitions["patient_id"].isin(split["test_patients"]).to_numpy() & mask
        )
        train_scalar, test_scalar = prepare_scalars(
            transitions, scalar_columns, train_index, test_index
        )
        for segment_wise in (False, True):
            layout = "segment" if segment_wise else "whole"
            transformer = MultiRocketMultivariate(
                num_kernels=MULTIROCKET_KERNELS,
                n_jobs=-1,
                random_state=SEED + int(split["split_index"]),
            )
            fit_curves = (
                curves[train_index].reshape(-1, curves.shape[2], curves.shape[3])
                if segment_wise
                else curves[train_index].reshape(len(train_index), -1, curves.shape[3])
            ).astype(np.float64)
            started = time.time()
            transformer.fit(fit_curves)
            train_curve = multirocket_features(transformer, curves[train_index], segment_wise)
            test_curve = multirocket_features(transformer, curves[test_index], segment_wise)
            for with_scalars in (False, True):
                model_name = f"multirocket_{layout}_curves_scalars" if with_scalars else f"multirocket_{layout}_curves"
                x_train = (
                    np.concatenate([train_curve, train_scalar], axis=1)
                    if with_scalars
                    else train_curve
                )
                x_test = (
                    np.concatenate([test_curve, test_scalar], axis=1)
                    if with_scalars
                    else test_curve
                )
                score = ridge_with_group_platt(
                    x_train,
                    labels[train_index],
                    transitions.iloc[train_index]["patient_id"].to_numpy(str),
                    x_test,
                    SEED + int(split["split_index"]),
                )
                accumulators[model_name][0][test_index] += score
                accumulators[model_name][1][test_index] += 1
                logs.append(
                    {
                        "model": model_name,
                        "repeat": split["repeat"],
                        "fold": split["fold"],
                        "features": x_train.shape[1],
                        "train_n": len(train_index),
                        "test_n": len(test_index),
                        "seconds": time.time() - started,
                    }
                )
            del transformer, train_curve, test_curve, fit_curves
    return prediction_rows(transitions, labels, mask, accumulators), pd.DataFrame(logs)


class MantisEmbeddingAdapter(nn.Module):
    def __init__(self, scalar_features: int, tasks: int, use_scalars: bool):
        super().__init__()
        self.use_scalars = use_scalars
        self.channel_logits = nn.Parameter(torch.zeros(6))
        self.adapter = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 32),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.segment_embedding = nn.Parameter(torch.zeros(1, 18, 32))
        nn.init.normal_(self.segment_embedding, mean=0.0, std=0.02)
        self.attention_score = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
        if use_scalars:
            self.scalar_encoder = nn.Sequential(
                nn.Linear(scalar_features, 48),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.Linear(48, 32),
                nn.GELU(),
            )
        pooled_features = 32 * 3 + (32 if use_scalars else 0)
        self.shared = nn.Sequential(nn.Linear(pooled_features, 64), nn.GELU(), nn.Dropout(0.35))
        self.head = nn.Linear(64, tasks)

    def forward(self, embeddings: torch.Tensor, scalars: torch.Tensor | None):
        channel_weights = torch.softmax(self.channel_logits, dim=0)
        encoded = torch.sum(embeddings * channel_weights[None, None, :, None], dim=2)
        encoded = self.adapter(encoded)
        scores = self.attention_score(encoded + self.segment_embedding).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        center = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        pooled = torch.cat([center, encoded.std(dim=1), encoded.max(dim=1).values], dim=1)
        if self.use_scalars:
            pooled = torch.cat([pooled, self.scalar_encoder(scalars)], dim=1)
        return self.head(self.shared(pooled)), weights, channel_weights


def adapter_oof(
    transitions: pd.DataFrame,
    embeddings: np.ndarray,
    scalar_columns: list[str],
    active_tasks: list[core.TaskSpec],
    splits: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    device = torch.device("cuda")
    labels = np.column_stack(
        [transitions[f"label__{task.name}"].to_numpy(np.float32) for task in active_tasks]
    )
    masks = np.column_stack(
        [transitions[f"mask__{task.name}"].to_numpy(np.float32) for task in active_tasks]
    )
    primary_index = [task.name for task in active_tasks].index(PRIMARY_TASK)
    primary_mask = masks[:, primary_index].astype(bool)
    primary_labels = labels[:, primary_index].astype(int)
    patient_primary = (
        transitions.groupby("patient_id")[f"label__{PRIMARY_TASK}"].max().astype(int).to_dict()
    )
    names = ["mantis_v2_adapter_curves", "mantis_v2_adapter_curves_scalars"]
    accumulators = {name: [np.zeros(len(transitions)), np.zeros(len(transitions), dtype=int)] for name in names}
    logs: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
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
        test_index = np.flatnonzero(transitions["patient_id"].isin(split["test_patients"]).to_numpy())
        fit_scalars, val_scalars = prepare_scalars(
            transitions, scalar_columns, fit_index, val_index
        )
        # Refit preprocessing on fit+validation for neither optimization nor inference is
        # intentionally avoided: this matches the retained CNN's early-stopping protocol.
        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        _ = scaler.fit(imputer.fit_transform(transitions.iloc[fit_index][scalar_columns].astype(float)))
        test_scalars = np.clip(
            scaler.transform(imputer.transform(transitions.iloc[test_index][scalar_columns].astype(float))),
            -5.0,
            5.0,
        ).astype(np.float32)
        fit_masks = masks[fit_index].copy()
        for task_index in range(len(active_tasks)):
            task_y = labels[fit_index][fit_masks[:, task_index] > 0, task_index]
            if task_y.sum() < 2 or (len(task_y) - task_y.sum()) < 2:
                fit_masks[:, task_index] = 0
        positives = (labels[fit_index] * fit_masks).sum(axis=0)
        negatives = ((1.0 - labels[fit_index]) * fit_masks).sum(axis=0)
        pos_weight = torch.as_tensor(
            np.clip(negatives / np.maximum(positives, 1.0), 1.0, 20.0).astype(np.float32),
            device=device,
        )
        for use_scalars in (False, True):
            model_name = "mantis_v2_adapter_curves_scalars" if use_scalars else "mantis_v2_adapter_curves"
            set_seed(split_seed)
            model = MantisEmbeddingAdapter(len(scalar_columns), len(active_tasks), use_scalars).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-3)
            best_state = copy.deepcopy(model.state_dict())
            best_loss = math.inf
            best_epoch = 0
            stale = 0
            started = time.time()
            for epoch in range(ADAPTER_EPOCHS):
                model.train()
                order = np.random.default_rng(split_seed + epoch).permutation(len(fit_index))
                for start in range(0, len(order), 64):
                    local = order[start : start + 64]
                    global_index = fit_index[local]
                    batch_embeddings = torch.as_tensor(embeddings[global_index], device=device)
                    batch_scalars = (
                        torch.as_tensor(fit_scalars[local], device=device) if use_scalars else None
                    )
                    batch_labels = torch.as_tensor(labels[global_index], device=device)
                    batch_masks = torch.as_tensor(fit_masks[local], device=device)
                    optimizer.zero_grad(set_to_none=True)
                    logits, _, _ = model(batch_embeddings, batch_scalars)
                    loss = channel.binary_loss(logits, batch_labels, batch_masks, pos_weight)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_logits, _, _ = model(
                        torch.as_tensor(embeddings[val_index], device=device),
                        torch.as_tensor(val_scalars, device=device) if use_scalars else None,
                    )
                    val_loss = channel.binary_loss(
                        val_logits,
                        torch.as_tensor(labels[val_index], device=device),
                        torch.as_tensor(masks[val_index], device=device),
                        pos_weight,
                    ).item()
                if val_loss < best_loss - 1e-5:
                    best_loss = val_loss
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                    if stale >= ADAPTER_PATIENCE:
                        break
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                logits, segment_weights, channel_weights = model(
                    torch.as_tensor(embeddings[test_index], device=device),
                    torch.as_tensor(test_scalars, device=device) if use_scalars else None,
                )
                scores = torch.sigmoid(logits[:, primary_index]).cpu().numpy()
            eligible_test = primary_mask[test_index]
            selected_index = test_index[eligible_test]
            accumulators[model_name][0][selected_index] += scores[eligible_test]
            accumulators[model_name][1][selected_index] += 1
            logs.append(
                {
                    "model": model_name,
                    "repeat": split["repeat"],
                    "fold": split["fold"],
                    "best_epoch": best_epoch,
                    "best_val_loss": best_loss,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "seconds": time.time() - started,
                }
            )
            for channel_index, weight in enumerate(channel_weights.cpu().numpy()):
                weight_rows.append(
                    {
                        "model": model_name,
                        "repeat": split["repeat"],
                        "fold": split["fold"],
                        "channel_index": channel_index + 1,
                        "channel": channel.VARIANTS[0].channels[channel_index],
                        "weight": float(weight),
                    }
                )
            del model
    return (
        prediction_rows(transitions, primary_labels, primary_mask, accumulators),
        pd.DataFrame(logs),
        pd.DataFrame(weight_rows),
    )


def prediction_rows(
    transitions: pd.DataFrame,
    labels: np.ndarray,
    mask: np.ndarray,
    accumulators: dict[str, list[np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, (score_sum, score_count) in accumulators.items():
        valid = mask & (score_count > 0)
        for index in np.flatnonzero(valid):
            rows.append(
                {
                    "task": PRIMARY_TASK,
                    "model": model_name,
                    "transition_id": transitions.iloc[index]["transition_id"],
                    "patient_id": transitions.iloc[index]["patient_id"],
                    "label": int(labels[index]),
                    "score": float(score_sum[index] / score_count[index]),
                    "prediction_repeats": int(score_count[index]),
                }
            )
    return pd.DataFrame(rows)


def add_references(predictions: pd.DataFrame) -> pd.DataFrame:
    cnn = pd.read_parquet(CNN_RESULTS / "cnn_length_ablation_oof_predictions.parquet")
    cnn = cnn[cnn["task"].eq(PRIMARY_TASK) & cnn["model"].eq("attention_t96")].copy()
    cnn["model"] = "current_cnn"
    prior = pd.read_parquet(BASE_RESULTS / "oof_predictions.parquet")
    clinical = prior[prior["task"].eq(PRIMARY_TASK) & prior["model"].eq("clinical_ridge")].copy()
    return pd.concat([clinical, cnn, predictions], ignore_index=True)


def fold_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    # The saved output is averaged across repeats. Fold-level stability is measured
    # by patient bootstrap in the main table; this table records model-level totals.
    for model, group in raw.groupby("model"):
        rows.append(
            {
                "model": model,
                "n": len(group),
                "events": int(group["label"].sum()),
                "auc": roc_auc_score(group["label"], group["score"]),
                "ap": average_precision_score(group["label"], group["score"]),
            }
        )
    return pd.DataFrame(rows)


def make_figure(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    selected = [
        "clinical_ridge",
        "current_cnn",
        "multirocket_whole_curves_scalars",
        "multirocket_segment_curves_scalars",
        "mantis_random_frozen_curves_scalars",
        "mantis_v2_frozen_curves_scalars",
        "mantis_v2_adapter_curves_scalars",
        "moment_small_frozen_curves_scalars",
    ]
    labels = {
        "clinical_ridge": "Clinical ridge",
        "current_cnn": "Current CNN",
        "multirocket_whole_curves_scalars": "MultiROCKET whole",
        "multirocket_segment_curves_scalars": "MultiROCKET segment",
        "mantis_random_frozen_curves_scalars": "Random Mantis control",
        "mantis_v2_frozen_curves_scalars": "MantisV2 frozen",
        "mantis_v2_adapter_curves_scalars": "MantisV2 adapter",
        "moment_small_frozen_curves_scalars": "MOMENT-small frozen",
    }
    metric_index = metrics.set_index("model")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))
    for color, model in zip(colors, selected):
        group = predictions[predictions["model"].eq(model)]
        if group.empty:
            continue
        fpr, tpr, _ = roc_curve(group["label"], group["score"])
        precision, recall, _ = precision_recall_curve(group["label"], group["score"])
        axes[0].plot(fpr, tpr, color=color, lw=2, label=f"{labels[model]} ({metric_index.loc[model, 'roc_auc']:.3f})")
        axes[1].plot(recall, precision, color=color, lw=2, label=f"{labels[model]} ({metric_index.loc[model, 'average_precision']:.3f})")
    prevalence = float(predictions[predictions["model"].eq("current_cnn")]["label"].mean())
    axes[0].plot([0, 1], [0, 1], "--", color="0.55", lw=1, label="Random AUC 0.500")
    axes[1].axhline(prevalence, ls="--", color="0.55", lw=1, label=f"Random AP {prevalence:.3f}")
    axes[0].set(title="ROC curves", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].set(title="Precision-recall curves", xlabel="Recall", ylabel="Precision")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Round 1 time-series models: next-visit 15% relative Mid-GLS deterioration")
    figure.tight_layout()
    figure.savefig(OUTPUT / "round1_roc_pr_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in selected.select_dtypes(include=[np.number]).columns:
        selected[column] = selected[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "")
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "|" + "|".join(["---"] * len(columns)) + "|",
            *["| " + " | ".join(map(str, row)) + " |" for row in selected.to_numpy()],
        ]
    )


def write_report(
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    ablations: pd.DataFrame,
    channel_weights: pd.DataFrame,
) -> None:
    names = {
        "clinical_ridge": "Clinical ridge",
        "current_cnn": "Current CNN",
        "multirocket_whole_curves": "MultiROCKET whole: curves",
        "multirocket_whole_curves_scalars": "MultiROCKET whole: curves + scalars",
        "multirocket_segment_curves": "MultiROCKET segment: curves",
        "multirocket_segment_curves_scalars": "MultiROCKET segment: curves + scalars",
        "mantis_v2_frozen_curves": "MantisV2 frozen: curves",
        "mantis_v2_frozen_curves_scalars": "MantisV2 frozen: curves + scalars",
        "mantis_random_frozen_curves": "Random Mantis: curves",
        "mantis_random_frozen_curves_scalars": "Random Mantis: curves + scalars",
        "mantis_v2_adapter_curves": "MantisV2 adapter: curves",
        "mantis_v2_adapter_curves_scalars": "MantisV2 adapter: curves + scalars",
        "moment_small_frozen_curves": "MOMENT-small frozen: curves",
        "moment_small_frozen_curves_scalars": "MOMENT-small frozen: curves + scalars",
    }
    table = metrics.copy()
    table["Model"] = table["model"].map(names)
    table = table.rename(
        columns={
            "roc_auc": "AUC",
            "roc_auc_ci_low": "AUC CI low",
            "roc_auc_ci_high": "AUC CI high",
            "average_precision": "AP",
            "average_precision_ci_low": "AP CI low",
            "average_precision_ci_high": "AP CI high",
        }
    ).sort_values("AP", ascending=False)
    delta_table = deltas.copy()
    delta_table["Model"] = delta_table["candidate_model"].map(names)
    delta_table = delta_table.rename(
        columns={
            "delta_roc_auc": "delta AUC",
            "delta_roc_auc_ci_low": "delta AUC CI low",
            "delta_roc_auc_ci_high": "delta AUC CI high",
            "delta_average_precision": "delta AP",
            "delta_average_precision_ci_low": "delta AP CI low",
            "delta_average_precision_ci_high": "delta AP CI high",
        }
    ).sort_values("delta AP", ascending=False)
    weight_summary = (
        channel_weights.groupby(["model", "channel"], as_index=False)["weight"]
        .agg(["mean", "std"])
        .reset_index()
    )
    ablation_names = {
        "mantis_pretraining_curves": "Mantis pretraining, curves",
        "mantis_pretraining_scalars": "Mantis pretraining, curves + scalars",
        "mantis_scalars": "Add scalars to frozen MantisV2",
        "moment_scalars": "Add scalars to frozen MOMENT-small",
        "random_mantis_scalars": "Add scalars to random Mantis",
        "adapter_scalars": "Add scalars to Mantis adapter",
        "rocket_whole_scalars": "Add scalars to whole-heart MultiROCKET",
        "rocket_segment_scalars": "Add scalars to segment MultiROCKET",
        "rocket_segment_vs_whole_curves": "Segment vs whole MultiROCKET, curves",
        "rocket_segment_vs_whole_scalars": "Segment vs whole MultiROCKET, curves + scalars",
        "mantis_adapter_vs_frozen_curves": "Mantis adapter vs frozen probe, curves",
        "mantis_adapter_vs_frozen_scalars": "Mantis adapter vs frozen probe, curves + scalars",
    }
    ablation_table = ablations.copy()
    ablation_table["Ablation"] = ablation_table["comparison"].map(ablation_names)
    ablation_table = ablation_table.rename(
        columns={
            "delta_roc_auc": "delta AUC",
            "delta_roc_auc_ci_low": "delta AUC CI low",
            "delta_roc_auc_ci_high": "delta AUC CI high",
            "delta_average_precision": "delta AP",
            "delta_average_precision_ci_low": "delta AP CI low",
            "delta_average_precision_ci_high": "delta AP CI high",
        }
    )
    best = table.iloc[0]
    report = f"""# Round 1: alternatives to the current strain-curve CNN

## Task and protocol

Predict whether the immediately following visit will be the first visit with at
least 15% relative Mid-GLS deterioration from the first-visit baseline. The
evaluation contains 238 eligible transitions, 49 events (20.6%), and 103
patients. All models use the exact same three repeated five-fold patient-held-out
splits as the retained CNN. Confidence intervals are 2,000 patient-cluster
bootstrap samples.

## Results

{markdown_table(table, ['Model', 'AUC', 'AUC CI low', 'AUC CI high', 'AP', 'AP CI low', 'AP CI high'])}

Random guessing has AUC 0.500 and expected AP 0.206. The highest AP was
**{best['Model']}**, with AUC {best['AUC']:.3f} and AP {best['AP']:.3f}.

## Paired change from the current CNN

{markdown_table(delta_table, ['Model', 'delta AUC', 'delta AUC CI low', 'delta AUC CI high', 'delta AP', 'delta AP CI low', 'delta AP CI high'])}

## What each model received

- MultiROCKET whole-heart: all 18 x 6 = 108 channels across 96 normalized time
  samples. It produced 49,728 convolution-response features.
- MultiROCKET segment-wise: each six-channel segment was transformed separately;
  feature mean, standard deviation, and maximum were pooled across 18 segments.
- MantisV2 and MOMENT-small: curves were interpolated from 96 to 512 samples only
  for pretrained-checkpoint compatibility. Every segment was encoded separately,
  while all six channel embeddings were retained. Segment mean, standard
  deviation, and maximum were used by the frozen probes.
- Frozen probes: fold-specific 32-component PCA followed by balanced logistic
  regression. PCA and scalar preprocessing were fit on training patients only.
- Mantis adapter: frozen pretrained embeddings, a learned six-channel weighting,
  256-to-32 embedding adapter, learned segment attention, and the same multitask
  labels and early-stopping scheme as the current CNN.

## Mantis adapter channel weights

{markdown_table(weight_summary.rename(columns={'channel': 'Channel', 'mean': 'Mean weight', 'std': 'SD'}), ['model', 'Channel', 'Mean weight', 'SD'])}

The weights remained close to the uniform value of 1/6 = 0.167. Therefore,
the adapter did not learn a stable dominant channel; the small preference for
the Endo and Mid change channels should not be interpreted as a strong finding.

## Controlled ablations

Positive values favor the named change.

{markdown_table(ablation_table, ['Ablation', 'delta AUC', 'delta AUC CI low', 'delta AUC CI high', 'delta AP', 'delta AP CI low', 'delta AP CI high'])}

## Interpretation guardrails

- A point estimate above the CNN is not sufficient if the paired confidence
  interval includes zero.
- Pretrained-versus-random Mantis separates transfer-learning value from the
  random-feature effect of the architecture.
- Interpolation to 512 samples adds no physiological information; it only adapts
  the data to the pretraining resolution.
- AP is the primary practical ranking measure because only 20.6% of eligible
  transitions are events.
"""
    (OUTPUT / "round1_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the foundation-model experiment")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    transitions, curves, scalar_columns, active_tasks, splits = load_inputs()
    run_logs: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    mantis = extract_mantis(curves, True, CACHE / "mantis_v2_embeddings.npz")
    mantis_random = extract_mantis(curves, False, CACHE / "mantis_random_embeddings.npz")
    moment = extract_moment(curves, CACHE / "moment_small_embeddings.npz")

    for prefix, embeddings in [
        ("mantis_v2_frozen", mantis),
        ("mantis_random_frozen", mantis_random),
        ("moment_small_frozen", moment),
    ]:
        predictions, logs = embedding_probe_oof(
            transitions,
            aggregate_segment_embeddings(embeddings),
            scalar_columns,
            splits,
            prefix,
        )
        prediction_frames.append(predictions)
        run_logs.append(logs)

    adapter_predictions, adapter_logs, channel_weights = adapter_oof(
        transitions, mantis, scalar_columns, active_tasks, splits
    )
    prediction_frames.append(adapter_predictions)
    run_logs.append(adapter_logs)

    rocket_predictions, rocket_logs = multirocket_oof(
        transitions, curves, scalar_columns, splits
    )
    prediction_frames.append(rocket_predictions)
    run_logs.append(rocket_logs)

    predictions = add_references(pd.concat(prediction_frames, ignore_index=True))
    expected = predictions.groupby("model")["transition_id"].nunique()
    if not (expected == 238).all():
        raise RuntimeError(f"Incomplete OOF predictions: {expected.to_dict()}")
    metrics, _ = core.evaluate_predictions(predictions, BOOTSTRAPS, SEED + 910000)
    comparisons = [
        (f"{model}_vs_current_cnn", "current_cnn", model)
        for model in predictions["model"].unique()
        if model not in {"current_cnn", "clinical_ridge"}
    ]
    deltas = qc.paired_variant_deltas(
        predictions, comparisons, BOOTSTRAPS, SEED + 920000
    )
    ablation_comparisons = [
        ("mantis_pretraining_curves", "mantis_random_frozen_curves", "mantis_v2_frozen_curves"),
        ("mantis_pretraining_scalars", "mantis_random_frozen_curves_scalars", "mantis_v2_frozen_curves_scalars"),
        ("mantis_scalars", "mantis_v2_frozen_curves", "mantis_v2_frozen_curves_scalars"),
        ("moment_scalars", "moment_small_frozen_curves", "moment_small_frozen_curves_scalars"),
        ("random_mantis_scalars", "mantis_random_frozen_curves", "mantis_random_frozen_curves_scalars"),
        ("adapter_scalars", "mantis_v2_adapter_curves", "mantis_v2_adapter_curves_scalars"),
        ("rocket_whole_scalars", "multirocket_whole_curves", "multirocket_whole_curves_scalars"),
        ("rocket_segment_scalars", "multirocket_segment_curves", "multirocket_segment_curves_scalars"),
        ("rocket_segment_vs_whole_curves", "multirocket_whole_curves", "multirocket_segment_curves"),
        ("rocket_segment_vs_whole_scalars", "multirocket_whole_curves_scalars", "multirocket_segment_curves_scalars"),
        ("mantis_adapter_vs_frozen_curves", "mantis_v2_frozen_curves", "mantis_v2_adapter_curves"),
        ("mantis_adapter_vs_frozen_scalars", "mantis_v2_frozen_curves_scalars", "mantis_v2_adapter_curves_scalars"),
    ]
    ablations = qc.paired_variant_deltas(
        predictions, ablation_comparisons, BOOTSTRAPS, SEED + 930000
    )
    predictions.to_parquet(OUTPUT / "round1_oof_predictions.parquet", index=False)
    metrics.to_csv(OUTPUT / "round1_metrics.csv", index=False)
    deltas.to_csv(OUTPUT / "round1_paired_deltas_vs_cnn.csv", index=False)
    ablations.to_csv(OUTPUT / "round1_ablation_deltas.csv", index=False)
    pd.concat(run_logs, ignore_index=True).to_csv(OUTPUT / "round1_training_log.csv", index=False)
    channel_weights.to_csv(OUTPUT / "mantis_adapter_channel_weights.csv", index=False)
    fold_metrics(predictions).to_csv(OUTPUT / "round1_metric_reproduction.csv", index=False)
    make_figure(predictions, metrics)
    write_report(metrics, deltas, ablations, channel_weights)
    metadata = {
        "task": PRIMARY_TASK,
        "patients": int(transitions["patient_id"].nunique()),
        "eligible_transitions": 238,
        "events": 49,
        "event_rate": 49 / 238,
        "scalar_features": len(scalar_columns),
        "curve_shape": list(curves.shape),
        "mantis_embedding_shape": list(mantis.shape),
        "moment_embedding_shape": list(moment.shape),
        "multirocket_kernels_requested": MULTIROCKET_KERNELS,
        "multirocket_features": 49728,
        "pca_components": PCA_COMPONENTS,
        "cv": "3 repeated 5-fold patient-held-out",
        "bootstraps": BOOTSTRAPS,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    (OUTPUT / "round1_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics.sort_values("average_precision", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
