from __future__ import annotations

import argparse
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
from aeon.classification.interval_based import DrCIFClassifier
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from xgboost import XGBClassifier

import cardiotoxicity_cnn_channel_ablation as channel
import cardiotoxicity_next_visit_gpu as core
import cardiotoxicity_nonapical_qc as qc
import cardiotoxicity_timeseries_round1 as round1
import cardiotoxicity_timeseries_round2 as round2


ROOT = Path(r"D:\us")
OUTPUT = ROOT / "cardiotoxicity_timeseries_round4_results"
SMOKE_OUTPUT = ROOT / "cardiotoxicity_timeseries_round4_smoke"
ROUND3_RESULTS = ROOT / "cardiotoxicity_timeseries_round3_results"
CACHE = OUTPUT / "cache"
PRIMARY_TASK = "mid_first_rel15"
SEED = 20260722
MODEL_SEEDS = 3
BOOTSTRAPS = 2000
EPOCHS = 120
PATIENCE = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round 4 dedicated time-series classifiers")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--neural-only", action="store_true")
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class InceptionModule(nn.Module):
    def __init__(self, input_channels: int, filters: int = 8, bottleneck: int = 16):
        super().__init__()
        self.bottleneck = (
            nn.Conv1d(input_channels, bottleneck, kernel_size=1, bias=False)
            if input_channels > 1
            else nn.Identity()
        )
        inner = bottleneck if input_channels > 1 else input_channels
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(inner, filters, kernel_size=size, padding=size // 2, bias=False)
                for size in (9, 19, 39)
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(input_channels, filters, kernel_size=1, bias=False),
        )
        self.norm = nn.BatchNorm1d(filters * 4)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(x)
        values = [branch(bottleneck) for branch in self.branches]
        values.append(self.pool_branch(x))
        return self.activation(self.norm(torch.cat(values, dim=1)))


class InceptionEncoder(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        channels = input_channels
        modules = []
        residuals = []
        for block in range(6):
            modules.append(InceptionModule(channels))
            channels = 32
            if block in (2, 5):
                residual_input = input_channels if block == 2 else 32
                residuals.append(
                    nn.Sequential(
                        nn.Conv1d(residual_input, 32, kernel_size=1, bias=False),
                        nn.BatchNorm1d(32),
                    )
                )
        self.modules_list = nn.ModuleList(modules)
        self.residuals = nn.ModuleList(residuals)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        residual_index = 0
        for index, module in enumerate(self.modules_list):
            x = module(x)
            if index in (2, 5):
                x = self.activation(x + self.residuals[residual_index](residual))
                residual = x
                residual_index += 1
        return x.mean(dim=-1)


class PredictionHead(nn.Module):
    def __init__(self, curve_dimensions: int, scalar_features: int, tasks: int, use_scalars: bool):
        super().__init__()
        self.use_scalars = use_scalars
        if use_scalars:
            self.scalar_branch = round2.ScalarBranch(scalar_features)
        dimensions = curve_dimensions + (32 if use_scalars else 0)
        self.shared = nn.Sequential(nn.Linear(dimensions, 64), nn.GELU(), nn.Dropout(0.35))
        self.head = nn.Linear(64, tasks)

    def forward(self, curves: torch.Tensor, scalars: torch.Tensor | None) -> torch.Tensor:
        if self.use_scalars:
            curves = torch.cat([curves, self.scalar_branch(scalars)], dim=1)
        return self.head(self.shared(curves))


class InceptionWholeCardio(nn.Module):
    def __init__(self, scalar_features: int, tasks: int, use_scalars: bool):
        super().__init__()
        self.encoder = InceptionEncoder(18 * 6)
        self.output = PredictionHead(32, scalar_features, tasks, use_scalars)

    def forward(self, curves: torch.Tensor, scalars: torch.Tensor | None):
        sequence = curves.reshape(curves.shape[0], 18 * 6, 96)
        return self.output(self.encoder(sequence), scalars), torch.empty(0, device=curves.device)


class InceptionSegmentCardio(nn.Module):
    def __init__(self, scalar_features: int, tasks: int, use_scalars: bool):
        super().__init__()
        self.encoder = InceptionEncoder(6)
        self.output = PredictionHead(32 * 3, scalar_features, tasks, use_scalars)

    def forward(self, curves: torch.Tensor, scalars: torch.Tensor | None):
        batch = curves.shape[0]
        encoded = self.encoder(curves.reshape(batch * 18, 6, 96)).reshape(batch, 18, 32)
        pooled = torch.cat(
            [encoded.mean(dim=1), encoded.std(dim=1, unbiased=False), encoded.amax(dim=1)], dim=1
        )
        return self.output(pooled, scalars), torch.empty(0, device=curves.device)


class SmallConvTranCardio(nn.Module):
    def __init__(self, scalar_features: int, tasks: int, use_scalars: bool):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv1d(18 * 6, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, 64))
        nn.init.normal_(self.cls, std=0.02)
        self.position = round2.SinusoidalPosition(97, 64)
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.20,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(64)
        self.output = PredictionHead(64, scalar_features, tasks, use_scalars)

    def forward(self, curves: torch.Tensor, scalars: torch.Tensor | None):
        batch = curves.shape[0]
        sequence = self.input_projection(curves.reshape(batch, 18 * 6, 96)).transpose(1, 2)
        tokens = torch.cat([self.cls.expand(batch, -1, -1), sequence], dim=1)
        tokens = tokens + self.position(tokens)
        representation = self.norm(self.transformer(tokens)[:, 0])
        return self.output(representation, scalars), torch.empty(0, device=curves.device)


def neural_specs() -> list[tuple[str, type[nn.Module], bool]]:
    return [
        ("inception_whole_curves", InceptionWholeCardio, False),
        ("inception_whole_curves_scalars", InceptionWholeCardio, True),
        ("inception_segment_curves", InceptionSegmentCardio, False),
        ("inception_segment_curves_scalars", InceptionSegmentCardio, True),
        ("convtran_small_curves", SmallConvTranCardio, False),
        ("convtran_small_curves_scalars", SmallConvTranCardio, True),
    ]


def empty_accumulators(models: list[str], seeds: int, rows: int):
    return {
        model: {
            "sum": np.zeros((seeds, rows), dtype=np.float64),
            "count": np.zeros((seeds, rows), dtype=np.int16),
        }
        for model in models
    }


def accumulators_to_predictions(
    transitions: pd.DataFrame,
    labels: np.ndarray,
    mask: np.ndarray,
    accumulators: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    average_rows = []
    seed_rows = []
    for model, values in accumulators.items():
        seed_scores = np.divide(
            values["sum"], values["count"], out=np.zeros_like(values["sum"]), where=values["count"] > 0
        )
        valid = mask & np.all(values["count"] > 0, axis=0)
        for index in np.flatnonzero(valid):
            average_rows.append(
                {
                    "task": PRIMARY_TASK,
                    "model": model,
                    "transition_id": transitions.iloc[index]["transition_id"],
                    "patient_id": transitions.iloc[index]["patient_id"],
                    "label": int(labels[index]),
                    "score": float(seed_scores[:, index].mean()),
                    "prediction_repeats": int(values["count"][:, index].sum()),
                }
            )
            for seed_index in range(seed_scores.shape[0]):
                seed_rows.append(
                    {
                        "task": PRIMARY_TASK,
                        "model": model,
                        "seed_index": seed_index,
                        "transition_id": transitions.iloc[index]["transition_id"],
                        "patient_id": transitions.iloc[index]["patient_id"],
                        "label": int(labels[index]),
                        "score": float(seed_scores[seed_index, index]),
                        "prediction_repeats": int(values["count"][seed_index, index]),
                    }
                )
    return pd.DataFrame(average_rows), pd.DataFrame(seed_rows)


def neural_oof(
    transitions: pd.DataFrame,
    curves: np.ndarray,
    scalar_columns: list[str],
    active_tasks: list[core.TaskSpec],
    splits: list[dict[str, object]],
    epochs: int,
    patience: int,
    seeds: int,
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
    patient_primary = transitions.groupby("patient_id")[f"label__{PRIMARY_TASK}"].max().astype(int).to_dict()
    specs = neural_specs()
    accumulators = empty_accumulators([x[0] for x in specs], seeds, len(transitions))
    logs = []

    from sklearn.model_selection import train_test_split

    for split in splits:
        split_base_seed = SEED + int(split["split_index"]) * 101
        train_patients = sorted(split["train_patients"])
        patient_labels = [patient_primary.get(patient, 0) for patient in train_patients]
        fit_patients, val_patients = train_test_split(
            train_patients, test_size=0.2, random_state=split_base_seed, stratify=patient_labels
        )
        fit_index = np.flatnonzero(transitions["patient_id"].isin(fit_patients).to_numpy())
        val_index = np.flatnonzero(transitions["patient_id"].isin(val_patients).to_numpy())
        test_index = np.flatnonzero(transitions["patient_id"].isin(split["test_patients"]).to_numpy())
        fit_scalars, val_scalars = round1.prepare_scalars(transitions, scalar_columns, fit_index, val_index)
        _, test_scalars = round1.prepare_scalars(transitions, scalar_columns, fit_index, test_index)
        fit_masks = masks[fit_index].copy()
        for task_index in range(len(active_tasks)):
            y = labels[fit_index][fit_masks[:, task_index] > 0, task_index]
            if y.sum() < 2 or len(y) - y.sum() < 2:
                fit_masks[:, task_index] = 0
        positives = (labels[fit_index] * fit_masks).sum(axis=0)
        negatives = ((1.0 - labels[fit_index]) * fit_masks).sum(axis=0)
        pos_weight = torch.as_tensor(
            np.clip(negatives / np.maximum(positives, 1.0), 1.0, 20.0).astype(np.float32), device=device
        )
        for model_name, model_class, use_scalars in specs:
            for seed_index in range(seeds):
                run_seed = split_base_seed + seed_index * 10007
                set_seed(run_seed)
                model = model_class(len(scalar_columns), len(active_tasks), use_scalars).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=3e-3)
                best_state = copy.deepcopy(model.state_dict())
                best_loss = math.inf
                best_epoch = 0
                stale = 0
                started = time.time()
                for epoch in range(epochs):
                    model.train()
                    order = np.random.default_rng(run_seed + epoch).permutation(len(fit_index))
                    for start in range(0, len(order), 32):
                        local = order[start : start + 32]
                        global_index = fit_index[local]
                        batch_curves = torch.as_tensor(curves[global_index], device=device)
                        batch_scalars = torch.as_tensor(fit_scalars[local], device=device) if use_scalars else None
                        batch_labels = torch.as_tensor(labels[global_index], device=device)
                        batch_masks = torch.as_tensor(fit_masks[local], device=device)
                        optimizer.zero_grad(set_to_none=True)
                        logits, _ = model(batch_curves, batch_scalars)
                        loss = channel.binary_loss(logits, batch_labels, batch_masks, pos_weight)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        logits, _ = model(
                            torch.as_tensor(curves[val_index], device=device),
                            torch.as_tensor(val_scalars, device=device) if use_scalars else None,
                        )
                        val_loss = channel.binary_loss(
                            logits,
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
                        if stale >= patience:
                            break
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    logits, _ = model(
                        torch.as_tensor(curves[test_index], device=device),
                        torch.as_tensor(test_scalars, device=device) if use_scalars else None,
                    )
                    score = torch.sigmoid(logits[:, primary_index]).cpu().numpy()
                eligible = primary_mask[test_index]
                selected = test_index[eligible]
                accumulators[model_name]["sum"][seed_index, selected] += score[eligible]
                accumulators[model_name]["count"][seed_index, selected] += 1
                logs.append(
                    {
                        "family": "neural",
                        "model": model_name,
                        "seed_index": seed_index,
                        "repeat": split["repeat"],
                        "fold": split["fold"],
                        "best_epoch": best_epoch,
                        "best_val_loss": best_loss,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "seconds": time.time() - started,
                    }
                )
                del model
                torch.cuda.empty_cache()
    predictions, seed_predictions = accumulators_to_predictions(
        transitions, primary_labels, primary_mask, accumulators
    )
    return predictions, seed_predictions, pd.DataFrame(logs)


def catch22_features(curves: np.ndarray, cache_path: Path) -> tuple[np.ndarray, list[str]]:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return cached["features"], cached["names"].tolist()
    import pycatch22

    example = pycatch22.catch22_all(curves[0, 0, 0].astype(float).tolist())
    base_names = example["names"]
    raw = np.zeros((len(curves), 18, 6, 22), dtype=np.float32)
    for row in range(len(curves)):
        for segment in range(18):
            for channel_index in range(6):
                values = pycatch22.catch22_all(
                    curves[row, segment, channel_index].astype(float).tolist()
                )["values"]
                raw[row, segment, channel_index] = np.asarray(values, dtype=np.float32)
    raw[~np.isfinite(raw)] = np.nan
    statistics = {
        "mean": np.nanmean(raw, axis=1),
        "std": np.nanstd(raw, axis=1),
        "min": np.nanmin(raw, axis=1),
        "max": np.nanmax(raw, axis=1),
        "median": np.nanmedian(raw, axis=1),
    }
    features = np.concatenate(list(statistics.values()), axis=2).reshape(len(curves), -1)
    names = []
    for channel_index in range(6):
        for statistic in statistics:
            for feature_name in base_names:
                names.append(f"c{channel_index + 1}_{statistic}_{feature_name}")
    # Concatenation is statistic-major within each channel; reorder to match the names.
    features = np.stack(
        [statistics[stat][:, channel_index, feature_index] for channel_index in range(6) for stat in statistics for feature_index in range(22)],
        axis=1,
    ).astype(np.float32)
    np.savez_compressed(cache_path, features=features, names=np.asarray(names, dtype=object))
    return features, names


def classical_oof(
    transitions: pd.DataFrame,
    curves: np.ndarray,
    scalar_columns: list[str],
    splits: list[dict[str, object]],
    seeds: int,
    smoke: bool,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    primary_mask = transitions[f"mask__{PRIMARY_TASK}"].astype(bool).to_numpy()
    labels = transitions[f"label__{PRIMARY_TASK}"].fillna(0).to_numpy(int)
    models = [
        "drcif_curves",
        "rdst_shapelet_curves",
        "rdst_shapelet_curves_scalars",
        "catch22_xgb_curves",
        "catch22_xgb_curves_scalars",
    ]
    accumulators = empty_accumulators(models, seeds, len(transitions))
    logs = []
    cache_path = output / "catch22_structured_features.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    catch_features, catch_names = catch22_features(curves, cache_path)
    flat_curves = curves.reshape(len(curves), 18 * 6, 96).copy()
    # Aeon rejects selected intervals that are locally flat, including after its
    # first-difference and periodogram representations. Add the same small,
    # broadband, label-independent perturbation to every classical curve. Its SD
    # is 0.001 versus median absolute curve magnitude 0.054.
    neutral_pattern = np.random.default_rng(0).standard_normal(96).astype(np.float32)
    neutral_pattern = neutral_pattern / neutral_pattern.std() * 1e-3
    flat_curves += neutral_pattern[None, None, :]
    # DrCIF also validates low-energy periodogram intervals against an absolute
    # 1e-7 threshold. Scaling every classical curve equally by 100 preserves all
    # relative morphology while keeping those numerical representations valid.
    flat_curves *= 100.0
    # RDST's numba kernels require a single floating dtype through normalized and
    # unnormalized paths; float64 avoids a float32/float64 unification failure.
    flat_curves = flat_curves.astype(np.float64, copy=False)
    fold_cache = output / "classical_fold_cache"
    fold_cache.mkdir(parents=True, exist_ok=True)

    for split in splits:
        train_mask = transitions["patient_id"].isin(split["train_patients"]).to_numpy() & primary_mask
        test_mask = transitions["patient_id"].isin(split["test_patients"]).to_numpy() & primary_mask
        train_index = np.flatnonzero(train_mask)
        test_index = np.flatnonzero(test_mask)
        y_train = labels[train_index]
        train_scalars, test_scalars = round1.prepare_scalars(
            transitions, scalar_columns, train_index, test_index
        )
        positive_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
        for seed_index in range(seeds):
            run_seed = SEED + int(split["split_index"]) * 101 + seed_index * 10007
            cache_file = fold_cache / f"split_{int(split['split_index']):02d}_seed_{seed_index}.npz"
            if cache_file.exists():
                cached = np.load(cache_file, allow_pickle=True)
                cached_index = cached["test_index"].astype(int)
                for name in models:
                    accumulators[name]["sum"][seed_index, cached_index] += cached[name]
                    accumulators[name]["count"][seed_index, cached_index] += 1
                logs.extend(json.loads(str(cached["logs"].item())))
                continue
            local_scores: dict[str, np.ndarray] = {}
            log_start = len(logs)

            started = time.time()
            drcif = DrCIFClassifier(
                n_estimators=20 if smoke else 60,
                att_subsample_size=10,
                use_pycatch22=True,
                random_state=run_seed,
                n_jobs=-1,
            )
            drcif.fit(flat_curves[train_index], y_train)
            score = drcif.predict_proba(flat_curves[test_index])[:, list(drcif.classes_).index(1)]
            local_scores["drcif_curves"] = score
            accumulators["drcif_curves"]["sum"][seed_index, test_index] += score
            accumulators["drcif_curves"]["count"][seed_index, test_index] += 1
            logs.append(
                {
                    "family": "classical",
                    "model": "drcif_curves",
                    "seed_index": seed_index,
                    "repeat": split["repeat"],
                    "fold": split["fold"],
                    "best_epoch": np.nan,
                    "best_val_loss": np.nan,
                    "parameters": 20 if smoke else 60,
                    "seconds": time.time() - started,
                }
            )

            started = time.time()
            shapelets = RandomDilatedShapeletTransform(
                max_shapelets=50 if smoke else 1200,
                random_state=run_seed,
                n_jobs=-1,
            )
            train_shapelet = shapelets.fit_transform(flat_curves[train_index], y_train)
            test_shapelet = shapelets.transform(flat_curves[test_index])
            for use_scalars in (False, True):
                name = "rdst_shapelet_curves_scalars" if use_scalars else "rdst_shapelet_curves"
                train_x = np.concatenate([train_shapelet, train_scalars], axis=1) if use_scalars else train_shapelet
                test_x = np.concatenate([test_shapelet, test_scalars], axis=1) if use_scalars else test_shapelet
                classifier = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=3000,
                        random_state=run_seed,
                    ),
                )
                classifier.fit(train_x, y_train)
                score = classifier.predict_proba(test_x)[:, 1]
                local_scores[name] = score
                accumulators[name]["sum"][seed_index, test_index] += score
                accumulators[name]["count"][seed_index, test_index] += 1
            logs.append(
                {
                    "family": "classical",
                    "model": "rdst_shapelet_curves_and_scalars",
                    "seed_index": seed_index,
                    "repeat": split["repeat"],
                    "fold": split["fold"],
                    "best_epoch": np.nan,
                    "best_val_loss": np.nan,
                    "parameters": train_shapelet.shape[1],
                    "seconds": time.time() - started,
                }
            )

            for use_scalars in (False, True):
                started = time.time()
                name = "catch22_xgb_curves_scalars" if use_scalars else "catch22_xgb_curves"
                train_x = np.concatenate([catch_features[train_index], train_scalars], axis=1) if use_scalars else catch_features[train_index]
                test_x = np.concatenate([catch_features[test_index], test_scalars], axis=1) if use_scalars else catch_features[test_index]
                classifier = XGBClassifier(
                    n_estimators=20 if smoke else 300,
                    max_depth=2,
                    learning_rate=0.03,
                    min_child_weight=3,
                    subsample=0.80,
                    colsample_bytree=0.70,
                    reg_alpha=1.0,
                    reg_lambda=5.0,
                    scale_pos_weight=positive_weight,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=8,
                    random_state=run_seed,
                )
                classifier.fit(train_x, y_train)
                score = classifier.predict_proba(test_x)[:, 1]
                local_scores[name] = score
                accumulators[name]["sum"][seed_index, test_index] += score
                accumulators[name]["count"][seed_index, test_index] += 1
                logs.append(
                    {
                        "family": "classical",
                        "model": name,
                        "seed_index": seed_index,
                        "repeat": split["repeat"],
                        "fold": split["fold"],
                        "best_epoch": np.nan,
                        "best_val_loss": np.nan,
                        "parameters": train_x.shape[1],
                        "seconds": time.time() - started,
                    }
                )
            np.savez_compressed(
                cache_file,
                test_index=test_index,
                logs=np.asarray(json.dumps(logs[log_start:]), dtype=object),
                **local_scores,
            )
    predictions, seed_predictions = accumulators_to_predictions(
        transitions, labels, primary_mask, accumulators
    )
    return predictions, seed_predictions, pd.DataFrame(logs), catch_names


def fixed_ensemble_rows(base: pd.DataFrame, combinations: dict[str, list[str]]) -> pd.DataFrame:
    template = base[base["model"].eq("current_cnn")].copy()
    pivot = base.pivot(index="transition_id", columns="model", values="score")
    rows = []
    for name, models in combinations.items():
        score = pivot[models].mean(axis=1)
        frame = template.drop(columns="score").merge(
            score.rename("score"), left_on="transition_id", right_index=True, how="left"
        )
        frame["model"] = name
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def seed_metrics(seed_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed_index), group in seed_predictions.groupby(["model", "seed_index"]):
        rows.append(
            {
                "model": model,
                "seed_index": seed_index,
                "roc_auc": roc_auc_score(group["label"], group["score"]),
                "average_precision": average_precision_score(group["label"], group["score"]),
            }
        )
    return pd.DataFrame(rows)


def make_figure(output: Path, predictions: pd.DataFrame, metrics: pd.DataFrame, best_new: str, best_ensemble: str):
    selected = [
        "clinical_ridge",
        "current_cnn",
        "moment_small_frozen_curves_scalars",
        best_new,
        best_ensemble,
    ]
    labels = {
        "clinical_ridge": "Clinical ridge",
        "current_cnn": "Current CNN",
        "moment_small_frozen_curves_scalars": "MOMENT-small",
        best_new: best_new.replace("_", " "),
        best_ensemble: "Best Mantis-free ensemble",
    }
    lookup = metrics.set_index("model")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for model in selected:
        group = predictions[predictions["model"].eq(model)]
        y = group["label"].to_numpy(int)
        score = group["score"].to_numpy(float)
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        axes[0].plot(fpr, tpr, linewidth=2, label=f"{labels[model]} ({lookup.loc[model, 'roc_auc']:.3f})")
        axes[1].plot(recall, precision, linewidth=2, label=f"{labels[model]} ({lookup.loc[model, 'average_precision']:.3f})")
    prevalence = predictions[predictions["model"].eq("current_cnn")]["label"].mean()
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].axhline(prevalence, color="k", linestyle="--", alpha=0.5)
    axes[0].set(title="ROC curves", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].set(title="Precision-recall curves", xlabel="Recall", ylabel="Precision")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="best")
    figure.suptitle("Round 4: dedicated time-series classifiers")
    figure.tight_layout()
    figure.savefig(output / "round4_roc_pr_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(
    output: Path,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    key_comparisons: pd.DataFrame,
    stability: pd.DataFrame,
    best_new: str,
    best_ensemble: str,
) -> None:
    names = {
        "clinical_ridge": "Clinical ridge",
        "current_cnn": "Current CNN",
        "moment_small_frozen_curves_scalars": "MOMENT-small + scalars",
        "timemil_curves_scalars": "TimeMIL attention + scalars",
        "scalar_mlp": "Scalar-only MLP",
        "inception_whole_curves": "InceptionTime whole-heart: curves",
        "inception_whole_curves_scalars": "InceptionTime whole-heart: curves + scalars",
        "inception_segment_curves": "InceptionTime segment-structured: curves",
        "inception_segment_curves_scalars": "InceptionTime segment-structured: curves + scalars",
        "convtran_small_curves": "ConvTran-small: curves",
        "convtran_small_curves_scalars": "ConvTran-small: curves + scalars",
        "drcif_curves": "DrCIF: curves",
        "drcif_curves_scalar_blend": "DrCIF + fixed scalar blend",
        "rdst_shapelet_curves": "RDST shapelets: curves",
        "rdst_shapelet_curves_scalars": "RDST shapelets: curves + scalars",
        "catch22_xgb_curves": "Catch22-XGBoost: curves",
        "catch22_xgb_curves_scalars": "Catch22-XGBoost: curves + scalars",
        "ensemble_cnn_moment": "Equal CNN + MOMENT",
        "ensemble_cnn_moment_timemil": "Equal CNN + MOMENT + TimeMIL",
    }
    for model in metrics["model"]:
        if model.startswith("ensemble_cnn_moment_") and model not in names:
            names[model] = "Equal CNN + MOMENT + " + model.removeprefix("ensemble_cnn_moment_").replace("_", " ")
    table = metrics.copy()
    table["Model"] = table["model"].map(names).fillna(table["model"])
    for source, target in [
        ("roc_auc", "AUC"), ("roc_auc_ci_low", "AUC CI low"), ("roc_auc_ci_high", "AUC CI high"),
        ("average_precision", "AP"), ("average_precision_ci_low", "AP CI low"), ("average_precision_ci_high", "AP CI high"),
    ]:
        table[target] = table[source].map(lambda value: f"{value:.3f}")
    table = table.sort_values("average_precision", ascending=False)

    delta_table = deltas.copy()
    delta_table["Model"] = delta_table["candidate_model"].map(names).fillna(delta_table["candidate_model"])
    for source, target in [
        ("delta_roc_auc", "delta AUC"), ("delta_roc_auc_ci_low", "delta AUC CI low"), ("delta_roc_auc_ci_high", "delta AUC CI high"),
        ("delta_average_precision", "delta AP"), ("delta_average_precision_ci_low", "delta AP CI low"), ("delta_average_precision_ci_high", "delta AP CI high"),
    ]:
        delta_table[target] = delta_table[source].map(lambda value: f"{value:.3f}")

    comparison_names = {
        "best_mantis_free_vs_clinical": "Best Mantis-free ensemble vs clinical ridge",
        "catch22_addition_to_cnn_moment": "Add Catch22 to CNN + MOMENT",
        "best_mantis_free_vs_mantis_ensemble": "Best Mantis-free vs Round 3 Mantis ensemble",
        "shapelet_ensemble_vs_catch22_ensemble": "Shapelet ensemble vs Catch22 ensemble",
    }
    key_table = key_comparisons.copy()
    key_table["Comparison"] = key_table["comparison"].map(comparison_names)
    for source, target in [
        ("delta_roc_auc", "delta AUC"), ("delta_roc_auc_ci_low", "delta AUC CI low"), ("delta_roc_auc_ci_high", "delta AUC CI high"),
        ("delta_average_precision", "delta AP"), ("delta_average_precision_ci_low", "delta AP CI low"), ("delta_average_precision_ci_high", "delta AP CI high"),
    ]:
        key_table[target] = key_table[source].map(lambda value: f"{value:.3f}")

    stable = stability.groupby("model", as_index=False).agg(
        auc_mean=("roc_auc", "mean"), auc_min=("roc_auc", "min"), auc_max=("roc_auc", "max"),
        ap_mean=("average_precision", "mean"), ap_min=("average_precision", "min"), ap_max=("average_precision", "max"),
    )
    stable["Model"] = stable["model"].map(names).fillna(stable["model"])
    stable["AUC mean [range]"] = stable.apply(lambda row: f"{row.auc_mean:.3f} [{row.auc_min:.3f}-{row.auc_max:.3f}]", axis=1)
    stable["AP mean [range]"] = stable.apply(lambda row: f"{row.ap_mean:.3f} [{row.ap_min:.3f}-{row.ap_max:.3f}]", axis=1)

    report = f"""# Round 4: dedicated time-series classifiers

## Locked protocol

Predict whether the immediately following visit is the first visit with at least
15% relative Mid-GLS deterioration from first-visit baseline. The evaluation has
238 transitions, 49 events, and 103 patients. All results use the same three
repeated five-fold patient-held-out splits and 2,000 patient-cluster bootstraps.
Randomized classifiers are averages of three fixed seeds.

## Results

{round1.markdown_table(table, ['Model', 'AUC', 'AUC CI low', 'AUC CI high', 'AP', 'AP CI low', 'AP CI high'])}

Random guessing has AUC 0.500 and expected AP 0.206. The best new classifier was
`{best_new}`. The best prespecified Mantis-free ensemble was `{best_ensemble}`.

## Paired changes from current CNN

{round1.markdown_table(delta_table, ['Model', 'delta AUC', 'delta AUC CI low', 'delta AUC CI high', 'delta AP', 'delta AP CI low', 'delta AP CI high'])}

## Key comparisons

Positive values favor the first named candidate in each comparison.

{round1.markdown_table(key_table, ['Comparison', 'delta AUC', 'delta AUC CI low', 'delta AUC CI high', 'delta AP', 'delta AP CI low', 'delta AP CI high'])}

## Seed stability

{round1.markdown_table(stable, ['Model', 'AUC mean [range]', 'AP mean [range]'])}

## Model definitions

- InceptionTime whole-heart processes all 108 segment-channel curves jointly with
  six multiscale inception blocks using temporal kernels 9, 19, and 39.
- Segment-structured InceptionTime applies the same six-channel encoder to every
  segment, then pools segment mean, standard deviation, and maximum.
- ConvTran-small uses a 108-to-64 temporal convolution followed by two four-head
  transformer layers and a classification token.
- DrCIF averages three 60-tree multivariate diverse-representation interval
  forests (180 trees total). Its
  scalar result is a fixed equal probability blend with the scalar-only MLP.
- RDST uses 1,200 multivariate dilated shapelets and balanced logistic regression.
- Catch22-XGBoost summarizes 22 curve characteristics across segments using mean,
  standard deviation, minimum, maximum, and median, then fits regularized shallow
  boosted trees.
"""
    (output / "round4_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for neural Round 4 models")
    output = SMOKE_OUTPUT if args.smoke else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    transitions, curves, scalar_columns, active_tasks, all_splits = round1.load_inputs()
    splits = all_splits[:1] if args.smoke else all_splits
    seeds = 1 if args.smoke else MODEL_SEEDS
    epochs = 2 if args.smoke else EPOCHS
    patience = 1 if args.smoke else PATIENCE
    started = time.time()

    prediction_parts = []
    seed_parts = []
    log_parts = []
    catch_names: list[str] = []
    if not args.classical_only:
        neural_paths = [
            output / "round4_neural_oof_predictions.parquet",
            output / "round4_neural_seed_oof_predictions.parquet",
            output / "round4_neural_training_log.csv",
        ]
        if args.resume and all(path.exists() for path in neural_paths):
            neural_predictions = pd.read_parquet(neural_paths[0])
            neural_seeds = pd.read_parquet(neural_paths[1])
            neural_logs = pd.read_csv(neural_paths[2])
        else:
            neural_predictions, neural_seeds, neural_logs = neural_oof(
                transitions, curves, scalar_columns, active_tasks, splits, epochs, patience, seeds
            )
            neural_predictions.to_parquet(neural_paths[0], index=False)
            neural_seeds.to_parquet(neural_paths[1], index=False)
            neural_logs.to_csv(neural_paths[2], index=False)
        prediction_parts.append(neural_predictions)
        seed_parts.append(neural_seeds)
        log_parts.append(neural_logs)
    if not args.neural_only:
        classical_paths = [
            output / "round4_classical_oof_predictions.parquet",
            output / "round4_classical_seed_oof_predictions.parquet",
            output / "round4_classical_training_log.csv",
            output / "catch22_feature_names.csv",
        ]
        if args.resume and all(path.exists() for path in classical_paths):
            classical_predictions = pd.read_parquet(classical_paths[0])
            classical_seeds = pd.read_parquet(classical_paths[1])
            classical_logs = pd.read_csv(classical_paths[2])
            catch_names = pd.read_csv(classical_paths[3])["feature"].tolist()
        else:
            classical_predictions, classical_seeds, classical_logs, catch_names = classical_oof(
                transitions, curves, scalar_columns, splits, seeds, args.smoke, output
            )
            classical_predictions.to_parquet(classical_paths[0], index=False)
            classical_seeds.to_parquet(classical_paths[1], index=False)
            classical_logs.to_csv(classical_paths[2], index=False)
            pd.DataFrame({"feature": catch_names}).to_csv(classical_paths[3], index=False)
        prediction_parts.append(classical_predictions)
        seed_parts.append(classical_seeds)
        log_parts.append(classical_logs)
    candidates = pd.concat(prediction_parts, ignore_index=True)
    all_seed_predictions = pd.concat(seed_parts, ignore_index=True)
    training_logs = pd.concat(log_parts, ignore_index=True)
    candidates.to_parquet(output / "round4_new_model_oof_predictions.parquet", index=False)
    all_seed_predictions.to_parquet(output / "round4_seed_oof_predictions.parquet", index=False)
    training_logs.to_csv(output / "round4_training_log.csv", index=False)
    if args.smoke or args.neural_only or args.classical_only:
        print(candidates.groupby("model").size())
        print(training_logs.groupby("model").seconds.sum().to_string())
        return 0

    prior = pd.read_parquet(ROUND3_RESULTS / "round3_oof_predictions.parquet")
    keep = [
        "clinical_ridge", "current_cnn", "moment_small_frozen_curves_scalars",
        "timemil_curves_scalars", "scalar_mlp",
    ]
    base = pd.concat([prior[prior["model"].isin(keep)], candidates], ignore_index=True)
    pivot = base.pivot(index="transition_id", columns="model", values="score")
    template = base[base["model"].eq("current_cnn")].copy()
    blend_score = 0.5 * pivot["drcif_curves"] + 0.5 * pivot["scalar_mlp"]
    blend = template.drop(columns="score").merge(
        blend_score.rename("score"), left_on="transition_id", right_index=True, how="left"
    )
    blend["model"] = "drcif_curves_scalar_blend"
    base = pd.concat([base, blend], ignore_index=True)

    new_scalar_models = [
        "inception_whole_curves_scalars", "inception_segment_curves_scalars",
        "convtran_small_curves_scalars", "drcif_curves_scalar_blend",
        "rdst_shapelet_curves_scalars", "catch22_xgb_curves_scalars",
    ]
    combinations = {
        "ensemble_cnn_moment": ["current_cnn", "moment_small_frozen_curves_scalars"],
        "ensemble_cnn_moment_timemil": ["current_cnn", "moment_small_frozen_curves_scalars", "timemil_curves_scalars"],
    }
    for model in new_scalar_models:
        combinations[f"ensemble_cnn_moment_{model}"] = [
            "current_cnn", "moment_small_frozen_curves_scalars", model
        ]
    ensembles = fixed_ensemble_rows(base, combinations)
    predictions = pd.concat([base, ensembles], ignore_index=True)
    completeness = predictions.groupby("model")["transition_id"].nunique()
    if not (completeness == 238).all():
        raise RuntimeError(f"Incomplete predictions: {completeness.to_dict()}")

    metrics, _ = core.evaluate_predictions(predictions, BOOTSTRAPS, SEED + 3010000)
    candidate_names = candidates["model"].unique().tolist() + ["drcif_curves_scalar_blend"] + list(combinations)
    comparisons = [(f"{model}_vs_cnn", "current_cnn", model) for model in candidate_names]
    deltas = qc.paired_variant_deltas(predictions, comparisons, BOOTSTRAPS, SEED + 3020000)
    round3 = pd.read_parquet(ROUND3_RESULTS / "round3_oof_predictions.parquet")
    mantis_ensemble = round3[round3["model"].eq("ensemble_equal_cnn_mantis_timemil")]
    comparison_predictions = pd.concat([predictions, mantis_ensemble], ignore_index=True)
    key_specs = [
        ("best_mantis_free_vs_clinical", "clinical_ridge", "ensemble_cnn_moment_catch22_xgb_curves_scalars"),
        ("catch22_addition_to_cnn_moment", "ensemble_cnn_moment", "ensemble_cnn_moment_catch22_xgb_curves_scalars"),
        ("best_mantis_free_vs_mantis_ensemble", "ensemble_equal_cnn_mantis_timemil", "ensemble_cnn_moment_catch22_xgb_curves_scalars"),
        ("shapelet_ensemble_vs_catch22_ensemble", "ensemble_cnn_moment_catch22_xgb_curves_scalars", "ensemble_cnn_moment_rdst_shapelet_curves_scalars"),
    ]
    key_comparisons = qc.paired_variant_deltas(
        comparison_predictions, key_specs, BOOTSTRAPS, SEED + 3030000
    )
    stability = seed_metrics(all_seed_predictions)

    new_metrics = metrics[metrics["model"].isin(candidates["model"].unique())]
    best_new = new_metrics.sort_values(["average_precision", "roc_auc"], ascending=False).iloc[0]["model"]
    ensemble_metrics = metrics[metrics["model"].isin(combinations)]
    best_ensemble = ensemble_metrics.sort_values(["average_precision", "roc_auc"], ascending=False).iloc[0]["model"]

    predictions.to_parquet(output / "round4_oof_predictions.parquet", index=False)
    metrics.to_csv(output / "round4_metrics.csv", index=False)
    deltas.to_csv(output / "round4_paired_deltas_vs_cnn.csv", index=False)
    key_comparisons.to_csv(output / "round4_key_comparisons.csv", index=False)
    stability.to_csv(output / "round4_seed_metrics.csv", index=False)
    pd.DataFrame({"feature": catch_names}).to_csv(output / "catch22_feature_names.csv", index=False)
    make_figure(output, predictions, metrics, best_new, best_ensemble)
    write_report(output, metrics, deltas, key_comparisons, stability, best_new, best_ensemble)
    postprocessing_seconds = time.time() - started
    training_seconds = float(pd.to_numeric(training_logs["seconds"], errors="coerce").sum())
    metadata = {
        "task": PRIMARY_TASK,
        "patients": int(transitions["patient_id"].nunique()),
        "eligible_transitions": 238,
        "events": 49,
        "event_rate": 49 / 238,
        "curve_shape": list(curves.shape),
        "cv": "3 repeated 5-fold patient-held-out",
        "model_seeds": MODEL_SEEDS,
        "bootstraps": BOOTSTRAPS,
        "device": torch.cuda.get_device_name(0),
        "training_seconds_from_logs": training_seconds,
        "postprocessing_seconds": postprocessing_seconds,
        "successful_pipeline_seconds_approx": training_seconds + postprocessing_seconds,
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "best_new_model": best_new,
        "best_mantis_free_ensemble": best_ensemble,
    }
    (output / "round4_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics.sort_values("average_precision", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
