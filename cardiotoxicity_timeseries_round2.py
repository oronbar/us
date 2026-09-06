from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch import nn

import cardiotoxicity_cnn_channel_ablation as channel
import cardiotoxicity_next_visit_gpu as core
import cardiotoxicity_nonapical_qc as qc
import cardiotoxicity_timeseries_round1 as round1


ROOT = Path(r"D:\us")
OUTPUT = ROOT / "cardiotoxicity_timeseries_round2_results"
SMOKE_OUTPUT = ROOT / "cardiotoxicity_timeseries_round2_smoke"
ROUND1_RESULTS = ROOT / "cardiotoxicity_timeseries_round1_results"
TS2VEC_SOURCE = Path(r"C:\Users\Oron\.cache\codex-external\ts2vec")
PRIMARY_TASK = "mid_first_rel15"
SEED = 20260722
BOOTSTRAPS = 2000
EPOCHS = 180
PATIENCE = 25
TS2VEC_ITERS = 200
PCA_COMPONENTS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round 2 strain-curve time-series models")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SinusoidalPosition(nn.Module):
    def __init__(self, length: int, dimensions: int):
        super().__init__()
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimensions, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dimensions)
        )
        encoding = torch.zeros(length, dimensions)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoding[: x.shape[1]].unsqueeze(0)


class PureSelectiveSSM(nn.Module):
    """Pure-PyTorch selective scan matching the single Mamba block equations.

    The official fused selective_scan CUDA extension is Linux-only. The sequence
    is only 96 samples, so an explicit recurrent scan is practical here.
    """

    def __init__(self, d_model: int = 32, d_state: int = 8, expand: int = 2, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = max(1, math.ceil(d_model / 16))
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        nn.init.constant_(self.dt_proj.bias, -3.0)
        state = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(state).repeat(self.d_inner, 1))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        x_branch, gate = self.in_proj(x).chunk(2, dim=-1)
        x_branch = self.conv(x_branch.transpose(1, 2))[..., :length].transpose(1, 2)
        x_branch = F.silu(x_branch)
        selective = self.x_proj(x_branch)
        dt_raw, b_value, c_value = torch.split(
            selective, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt_raw))
        transition = -torch.exp(self.A_log.float())
        hidden = torch.zeros(
            batch, self.d_inner, self.d_state, device=x.device, dtype=torch.float32
        )
        outputs: list[torch.Tensor] = []
        for time_index in range(length):
            dt = delta[:, time_index].float()
            decay = torch.exp(dt.unsqueeze(-1) * transition.unsqueeze(0))
            drive = (
                dt.unsqueeze(-1)
                * b_value[:, time_index].float().unsqueeze(1)
                * x_branch[:, time_index].float().unsqueeze(-1)
            )
            hidden = decay * hidden + drive
            y = torch.sum(hidden * c_value[:, time_index].float().unsqueeze(1), dim=-1)
            outputs.append(y)
        stacked = torch.stack(outputs, dim=1).to(x.dtype)
        stacked = stacked * F.silu(gate)
        return self.out_proj(stacked)


class ScalarBranch(nn.Module):
    def __init__(self, scalar_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(scalar_features, 48),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(48, 32),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ScalarOnlyCardio(nn.Module):
    def __init__(self, scalar_features: int, tasks: int, use_scalars: bool = True):
        super().__init__()
        self.scalar_branch = ScalarBranch(scalar_features)
        self.shared = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Dropout(0.35))
        self.head = nn.Linear(64, tasks)

    def forward(self, curves: torch.Tensor, scalars: torch.Tensor | None):
        return self.head(self.shared(self.scalar_branch(scalars))), torch.empty(0, device=scalars.device)


class MambaSLCardio(nn.Module):
    def __init__(self, scalar_features: int, tasks: int, use_scalars: bool):
        super().__init__()
        self.use_scalars = use_scalars
        self.input_projection = nn.Conv1d(
            18 * 6, 32, kernel_size=7, padding=3, padding_mode="replicate", bias=False
        )
        self.position = SinusoidalPosition(96, 32)
        self.mamba = PureSelectiveSSM(d_model=32, d_state=8, expand=2, d_conv=4)
        self.norm = nn.LayerNorm(32)
        self.activation = nn.SiLU()
        self.pool_heads = nn.Linear(32, 4)
        nn.init.zeros_(self.pool_heads.weight)
        nn.init.ones_(self.pool_heads.bias)
        if use_scalars:
            self.scalar_branch = ScalarBranch(scalar_features)
        shared_input = 32 + (32 if use_scalars else 0)
        self.shared = nn.Sequential(
            nn.Linear(shared_input, 64), nn.GELU(), nn.Dropout(0.35)
        )
        self.head = nn.Linear(64, tasks)

    def forward(self, curves: torch.Tensor, scalars: torch.Tensor | None):
        batch = curves.shape[0]
        sequence = curves.reshape(batch, 18 * 6, 96)
        hidden = self.input_projection(sequence).transpose(1, 2)
        hidden = hidden + self.position(hidden)
        hidden = self.activation(self.norm(self.mamba(hidden)))
        scores = self.pool_heads(hidden).amax(dim=-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        if self.use_scalars:
            pooled = torch.cat([pooled, self.scalar_branch(scalars)], dim=1)
        return self.head(self.shared(pooled)), weights


class TimeMILCardio(nn.Module):
    def __init__(self, scalar_features: int, tasks: int, use_scalars: bool):
        super().__init__()
        self.use_scalars = use_scalars
        self.patch_size = 8
        self.patches = 12
        self.patch_encoder = nn.Sequential(
            nn.LayerNorm(6 * self.patch_size),
            nn.Linear(6 * self.patch_size, 32),
            nn.GELU(),
        )
        self.segment_embedding = nn.Parameter(torch.zeros(18, 32))
        self.time_embedding = nn.Parameter(torch.zeros(self.patches, 32))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 32))
        nn.init.normal_(self.segment_embedding, std=0.02)
        # Multiscale wave-like initialization; remains learnable.
        positions = torch.linspace(0, 1, self.patches).unsqueeze(1)
        frequencies = torch.arange(1, 17, dtype=torch.float32).unsqueeze(0)
        wave = torch.cat(
            [torch.sin(2 * math.pi * positions * frequencies),
             torch.cos(2 * math.pi * positions * frequencies)],
            dim=1,
        )[:, :32]
        with torch.no_grad():
            self.time_embedding.copy_(wave)
        layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.instance_context = nn.TransformerEncoder(layer, num_layers=1)
        self.instance_score = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
        if use_scalars:
            self.scalar_branch = ScalarBranch(scalar_features)
        shared_input = 32 + (32 if use_scalars else 0)
        self.shared = nn.Sequential(
            nn.Linear(shared_input, 64), nn.GELU(), nn.Dropout(0.35)
        )
        self.head = nn.Linear(64, tasks)

    def forward(self, curves: torch.Tensor, scalars: torch.Tensor | None):
        batch = curves.shape[0]
        patches = curves.unfold(-1, self.patch_size, self.patch_size)
        # B x segment x channel x patch x within-patch -> B x segment x patch x 48
        patches = patches.permute(0, 1, 3, 2, 4).reshape(batch, 18, self.patches, -1)
        tokens = self.patch_encoder(patches)
        tokens = (
            tokens
            + self.segment_embedding[None, :, None, :]
            + self.time_embedding[None, None, :, :]
        ).reshape(batch, 18 * self.patches, 32)
        if self.training:
            drop = torch.rand(batch, 18 * self.patches, 1, device=tokens.device) < 0.10
            tokens = torch.where(drop, self.mask_token.expand_as(tokens), tokens)
        tokens = self.instance_context(tokens)
        weights = torch.softmax(self.instance_score(tokens).squeeze(-1), dim=1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        if self.use_scalars:
            pooled = torch.cat([pooled, self.scalar_branch(scalars)], dim=1)
        return self.head(self.shared(pooled)), weights.reshape(batch, 18, self.patches)


def supervised_oof(
    transitions: pd.DataFrame,
    curves: np.ndarray,
    scalar_columns: list[str],
    active_tasks: list[core.TaskSpec],
    splits: list[dict[str, object]],
    epochs: int,
    patience: int,
    specs_override: list[tuple[str, type[nn.Module], bool]] | None = None,
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
    specs = specs_override or [
        ("scalar_mlp", ScalarOnlyCardio, True),
        ("mambasl_curves", MambaSLCardio, False),
        ("mambasl_curves_scalars", MambaSLCardio, True),
        ("timemil_curves", TimeMILCardio, False),
        ("timemil_curves_scalars", TimeMILCardio, True),
    ]
    accumulators = {
        name: [np.zeros(len(transitions)), np.zeros(len(transitions), dtype=int)]
        for name, _, _ in specs
    }
    attention_sum = {
        name: np.zeros((len(transitions), 18, 12), dtype=np.float32)
        for name, _, _ in specs
        if name.startswith("timemil")
    }
    attention_count = {
        name: np.zeros(len(transitions), dtype=int) for name in attention_sum
    }
    logs: list[dict[str, object]] = []

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
        fit_scalars, val_scalars = round1.prepare_scalars(
            transitions, scalar_columns, fit_index, val_index
        )
        _, test_scalars = round1.prepare_scalars(
            transitions, scalar_columns, fit_index, test_index
        )
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
        for model_name, model_class, use_scalars in specs:
            set_seed(split_seed)
            model = model_class(len(scalar_columns), len(active_tasks), use_scalars).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-3)
            best_state = copy.deepcopy(model.state_dict())
            best_loss = math.inf
            best_epoch = 0
            stale = 0
            started = time.time()
            for epoch in range(epochs):
                model.train()
                order = np.random.default_rng(split_seed + epoch).permutation(len(fit_index))
                for start in range(0, len(order), 32):
                    local = order[start : start + 32]
                    global_index = fit_index[local]
                    batch_curves = torch.as_tensor(curves[global_index], device=device)
                    batch_scalars = (
                        torch.as_tensor(fit_scalars[local], device=device) if use_scalars else None
                    )
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
                logits, weights = model(
                    torch.as_tensor(curves[test_index], device=device),
                    torch.as_tensor(test_scalars, device=device) if use_scalars else None,
                )
                score = torch.sigmoid(logits[:, primary_index]).cpu().numpy()
                weights_np = weights.float().cpu().numpy()
            eligible = primary_mask[test_index]
            selected = test_index[eligible]
            accumulators[model_name][0][selected] += score[eligible]
            accumulators[model_name][1][selected] += 1
            if model_name in attention_sum:
                attention_sum[model_name][selected] += weights_np[eligible]
                attention_count[model_name][selected] += 1
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
            del model
            torch.cuda.empty_cache()

    attention_rows: list[dict[str, object]] = []
    for model_name in attention_sum:
        valid = primary_mask & (attention_count[model_name] > 0)
        for index in np.flatnonzero(valid):
            values = attention_sum[model_name][index] / attention_count[model_name][index]
            for segment in range(18):
                for patch in range(12):
                    attention_rows.append(
                        {
                            "model": model_name,
                            "transition_id": transitions.iloc[index]["transition_id"],
                            "patient_id": transitions.iloc[index]["patient_id"],
                            "label": int(primary_labels[index]),
                            "segment": segment + 1,
                            "temporal_patch": patch + 1,
                            "weight": float(values[segment, patch]),
                        }
                    )
    return (
        round1.prediction_rows(transitions, primary_labels, primary_mask, accumulators),
        pd.DataFrame(logs),
        pd.DataFrame(attention_rows),
    )


def aggregate_ts2vec(embeddings: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [embeddings.mean(axis=1), embeddings.std(axis=1), embeddings.max(axis=1)], axis=1
    ).astype(np.float32)


def ts2vec_split_embeddings(
    curves: np.ndarray,
    transitions: pd.DataFrame,
    split: dict[str, object],
    cache_dir: Path,
    n_iters: int,
) -> tuple[np.ndarray, np.ndarray, list[float], float]:
    cache_path = cache_dir / f"ts2vec_split_{int(split['split_index']):02d}_iter{n_iters}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return (
            cached["trained"],
            cached["random"],
            cached["loss"].tolist(),
            float(cached["seconds"]),
        )
    if str(TS2VEC_SOURCE) not in sys.path:
        sys.path.insert(0, str(TS2VEC_SOURCE))
    from ts2vec import TS2Vec

    split_seed = SEED + int(split["split_index"]) * 101 + 50000
    set_seed(split_seed)
    all_segments = curves.transpose(0, 1, 3, 2).reshape(-1, 96, 6).astype(np.float32)
    train_transition = np.flatnonzero(
        transitions["patient_id"].isin(split["train_patients"]).to_numpy()
    )
    train_segments = curves[train_transition].transpose(0, 1, 3, 2).reshape(-1, 96, 6)
    mean = train_segments.mean(axis=(0, 1), keepdims=True)
    std = train_segments.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-4)
    all_scaled = ((all_segments - mean) / std).astype(np.float32)
    train_scaled = (
        (train_segments.astype(np.float32) - mean) / std
    ).astype(np.float32)
    model = TS2Vec(
        input_dims=6,
        output_dims=128,
        hidden_dims=64,
        depth=6,
        device="cuda",
        lr=1e-3,
        batch_size=32,
    )
    started = time.time()
    random_embedding = model.encode(
        all_scaled, encoding_window="full_series", batch_size=128
    ).reshape(len(curves), 18, 128)
    loss = model.fit(train_scaled, n_iters=n_iters, verbose=False)
    trained_embedding = model.encode(
        all_scaled, encoding_window="full_series", batch_size=128
    ).reshape(len(curves), 18, 128)
    seconds = time.time() - started
    np.savez_compressed(
        cache_path,
        trained=trained_embedding.astype(np.float32),
        random=random_embedding.astype(np.float32),
        loss=np.asarray(loss, dtype=np.float32),
        seconds=np.asarray(seconds),
        channel_mean=mean,
        channel_std=std,
    )
    del model
    torch.cuda.empty_cache()
    return trained_embedding, random_embedding, loss, seconds


def probe_split(
    transitions: pd.DataFrame,
    features: np.ndarray,
    scalar_columns: list[str],
    split: dict[str, object],
    prefix: str,
    accumulators: dict[str, list[np.ndarray]],
) -> list[dict[str, object]]:
    mask = transitions[f"mask__{PRIMARY_TASK}"].astype(bool).to_numpy()
    labels = transitions[f"label__{PRIMARY_TASK}"].to_numpy(int)
    train_index = np.flatnonzero(
        transitions["patient_id"].isin(split["train_patients"]).to_numpy() & mask
    )
    test_index = np.flatnonzero(
        transitions["patient_id"].isin(split["test_patients"]).to_numpy() & mask
    )
    components = min(PCA_COMPONENTS, len(train_index) - 1, features.shape[1])
    pca = PCA(n_components=components, whiten=True, svd_solver="randomized", random_state=SEED)
    started = time.time()
    train_curve = pca.fit_transform(features[train_index]).astype(np.float32)
    test_curve = pca.transform(features[test_index]).astype(np.float32)
    train_scalar, test_scalar = round1.prepare_scalars(
        transitions, scalar_columns, train_index, test_index
    )
    logs = []
    for with_scalars in (False, True):
        model_name = f"{prefix}_curves_scalars" if with_scalars else f"{prefix}_curves"
        x_train = (
            np.concatenate([train_curve, train_scalar], axis=1) if with_scalars else train_curve
        )
        x_test = (
            np.concatenate([test_curve, test_scalar], axis=1) if with_scalars else test_curve
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
                "pca_components": components,
                "parameters": x_train.shape[1] + 1,
                "seconds": time.time() - started,
            }
        )
    return logs


def ts2vec_oof(
    transitions: pd.DataFrame,
    curves: np.ndarray,
    scalar_columns: list[str],
    splits: list[dict[str, object]],
    cache_dir: Path,
    n_iters: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = transitions[f"mask__{PRIMARY_TASK}"].astype(bool).to_numpy()
    labels = transitions[f"label__{PRIMARY_TASK}"].to_numpy(int)
    names = [
        "ts2vec_ssl_curves",
        "ts2vec_ssl_curves_scalars",
        "ts2vec_random_curves",
        "ts2vec_random_curves_scalars",
    ]
    accumulators = {
        name: [np.zeros(len(transitions)), np.zeros(len(transitions), dtype=int)] for name in names
    }
    logs: list[dict[str, object]] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        trained, random_embedding, loss, seconds = ts2vec_split_embeddings(
            curves, transitions, split, cache_dir, n_iters
        )
        logs.append(
            {
                "model": "ts2vec_ssl_pretraining",
                "repeat": split["repeat"],
                "fold": split["fold"],
                "iterations": n_iters,
                "loss_first": loss[0] if loss else np.nan,
                "loss_last": loss[-1] if loss else np.nan,
                "seconds": seconds,
            }
        )
        logs.extend(
            probe_split(
                transitions,
                aggregate_ts2vec(trained),
                scalar_columns,
                split,
                "ts2vec_ssl",
                accumulators,
            )
        )
        logs.extend(
            probe_split(
                transitions,
                aggregate_ts2vec(random_embedding),
                scalar_columns,
                split,
                "ts2vec_random",
                accumulators,
            )
        )
    return round1.prediction_rows(transitions, labels, mask, accumulators), pd.DataFrame(logs)


def add_references(predictions: pd.DataFrame) -> pd.DataFrame:
    prior = pd.read_parquet(ROUND1_RESULTS / "round1_oof_predictions.parquet")
    keep = [
        "clinical_ridge",
        "current_cnn",
        "mantis_random_frozen_curves_scalars",
        "moment_small_frozen_curves_scalars",
    ]
    prior = prior[prior["model"].isin(keep)].copy()
    return pd.concat([prior, predictions], ignore_index=True)


def make_figure(output: Path, predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    selected = [
        "clinical_ridge",
        "current_cnn",
        "moment_small_frozen_curves_scalars",
        "scalar_mlp",
        "mambasl_curves_scalars",
        "timemil_curves_scalars",
        "ts2vec_ssl_curves_scalars",
        "ts2vec_random_curves_scalars",
    ]
    labels = {
        "clinical_ridge": "Clinical ridge",
        "current_cnn": "Current CNN",
        "moment_small_frozen_curves_scalars": "MOMENT-small",
        "scalar_mlp": "Scalar-only MLP",
        "mambasl_curves_scalars": "MambaSL",
        "timemil_curves_scalars": "TimeMIL-lite",
        "ts2vec_ssl_curves_scalars": "TS2Vec SSL",
        "ts2vec_random_curves_scalars": "Random TS2Vec control",
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
        axes[0].plot(
            fpr,
            tpr,
            lw=2,
            color=color,
            label=f"{labels[model]} ({metric_index.loc[model, 'roc_auc']:.3f})",
        )
        axes[1].plot(
            recall,
            precision,
            lw=2,
            color=color,
            label=f"{labels[model]} ({metric_index.loc[model, 'average_precision']:.3f})",
        )
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
    figure.suptitle("Round 2 time-series models: next-visit 15% relative Mid-GLS deterioration")
    figure.tight_layout()
    figure.savefig(output / "round2_roc_pr_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(
    output: Path,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    ablations: pd.DataFrame,
    attention: pd.DataFrame,
) -> None:
    names = {
        "clinical_ridge": "Clinical ridge",
        "current_cnn": "Current CNN",
        "mantis_random_frozen_curves_scalars": "Round 1 random-Mantis control + scalars",
        "moment_small_frozen_curves_scalars": "Round 1 MOMENT-small + scalars",
        "scalar_mlp": "Scalar-only multitask MLP",
        "mambasl_curves": "MambaSL: curves",
        "mambasl_curves_scalars": "MambaSL: curves + scalars",
        "timemil_curves": "TimeMIL-lite: curves",
        "timemil_curves_scalars": "TimeMIL-lite: curves + scalars",
        "ts2vec_ssl_curves": "TS2Vec SSL: curves",
        "ts2vec_ssl_curves_scalars": "TS2Vec SSL: curves + scalars",
        "ts2vec_random_curves": "Random TS2Vec: curves",
        "ts2vec_random_curves_scalars": "Random TS2Vec: curves + scalars",
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
    ablation_names = {
        "mambasl_scalars": "Add scalars to MambaSL",
        "timemil_scalars": "Add scalars to TimeMIL-lite",
        "ts2vec_scalars": "Add scalars to TS2Vec SSL",
        "ts2vec_random_scalars": "Add scalars to random TS2Vec",
        "ts2vec_ssl_value_curves": "TS2Vec SSL vs random initialization, curves",
        "ts2vec_ssl_value_scalars": "TS2Vec SSL vs random initialization, curves + scalars",
        "timemil_vs_mambasl_curves": "TimeMIL-lite vs MambaSL, curves",
        "timemil_vs_mambasl_scalars": "TimeMIL-lite vs MambaSL, curves + scalars",
        "timemil_vs_scalar": "TimeMIL-lite + scalars vs scalar-only MLP",
        "mambasl_vs_scalar": "MambaSL + scalars vs scalar-only MLP",
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
    attention_summary = (
        attention.groupby(["model", "transition_id", "segment"], as_index=False)["weight"]
        .sum()
        .groupby(["model", "segment"], as_index=False)["weight"]
        .mean()
        if not attention.empty
        else pd.DataFrame(columns=["model", "segment", "weight"])
    )
    top_attention = (
        attention_summary.sort_values(["model", "weight"], ascending=[True, False])
        .groupby("model")
        .head(5)
        .rename(columns={"model": "Model", "segment": "Segment", "weight": "Mean segment mass"})
    )
    entropy_rows = []
    if not attention.empty:
        for (model, transition_id), group in attention.groupby(["model", "transition_id"]):
            values = group["weight"].to_numpy(float)
            entropy_rows.append(
                {
                    "Model": model,
                    "normalized_entropy": float(
                        -np.sum(values * np.log(np.maximum(values, 1e-12))) / np.log(len(values))
                    ),
                }
            )
    entropy = (
        pd.DataFrame(entropy_rows)
        .groupby("Model", as_index=False)["normalized_entropy"]
        .mean()
        .rename(columns={"normalized_entropy": "Mean normalized entropy"})
        if entropy_rows
        else pd.DataFrame(columns=["Model", "Mean normalized entropy"])
    )
    best = table.iloc[0]
    report = f"""# Round 2: specialized time-series models

## Task and protocol

Predict whether the immediately following visit will be the first visit with at
least 15% relative Mid-GLS deterioration from the first-visit baseline. The
locked evaluation contains 238 eligible transitions, 49 events, and 103
patients. All results use the same three repeated five-fold patient-held-out
splits as Rounds 1 and the current CNN. Confidence intervals use 2,000
patient-cluster bootstrap samples.

## Results

{round1.markdown_table(table, ['Model', 'AUC', 'AUC CI low', 'AUC CI high', 'AP', 'AP CI low', 'AP CI high'])}

Random guessing has AUC 0.500 and expected AP 0.206. The highest AP in the full
comparison was **{best['Model']}**, with AUC {best['AUC']:.3f} and AP {best['AP']:.3f}.

## Paired change from the current CNN

{round1.markdown_table(delta_table, ['Model', 'delta AUC', 'delta AUC CI low', 'delta AUC CI high', 'delta AP', 'delta AP CI low', 'delta AP CI high'])}

## Controlled ablations

Positive values favor the named change.

{round1.markdown_table(ablation_table, ['Ablation', 'delta AUC', 'delta AUC CI low', 'delta AUC CI high', 'delta AP', 'delta AP CI low', 'delta AP CI high'])}

## Model definitions

- MambaSL: 108 simultaneous segment-channel variables across 96 samples, a
  seven-sample input projection, one 32-dimensional selective state-space layer,
  and four-head adaptive temporal pooling. The official fused selective-scan
  extension is Linux-only, so the same recurrence was evaluated with an explicit
  pure-PyTorch scan on Windows.
- TimeMIL-lite: every eight-sample patch from every segment is an instance, giving
  18 x 12 = 216 instances per transition. It uses a 32-dimensional patch encoder,
  segment embeddings, multiscale temporal positional embeddings, one four-head
  transformer layer, 10% patch masking, and MIL attention pooling.
- TS2Vec: official hierarchical contrastive training for 200 iterations in each
  outer fold using only curves from that fold's training patients. Six-channel
  segment series are encoded into 128-dimensional embeddings and pooled across
  segments. A random-initialization control uses the identical encoder before
  contrastive training.
- All trainable supervised models use the same multitask labels, class weighting,
  internal patient validation, and early stopping as the retained CNN.
- The scalar-only MLP uses the identical 96-to-32 scalar branch and multitask
  head, but receives no curves. It isolates the incremental value of each curve
  architecture.

## TimeMIL segment attention

{round1.markdown_table(top_attention, ['Model', 'Segment', 'Mean segment mass'])}

Uniform segment mass is 1/18 = 0.0556. Attention concentration is summarized
below; normalized entropy 1.0 means completely uniform attention across all 216
instances.

{round1.markdown_table(entropy, ['Model', 'Mean normalized entropy'])}

Attention values are descriptive, not causal feature importance. Stable segment
preferences should be interpreted only if the model itself generalizes.
"""
    (output / "round2_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Round 2")
    output = SMOKE_OUTPUT if args.smoke else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "ts2vec_cache"
    transitions, curves, scalar_columns, active_tasks, all_splits = round1.load_inputs()
    splits = all_splits[:1] if args.smoke else all_splits
    epochs = 2 if args.smoke else EPOCHS
    patience = 1 if args.smoke else PATIENCE
    ts2vec_iters = 2 if args.smoke else TS2VEC_ITERS
    bootstraps = 20 if args.smoke else BOOTSTRAPS

    supervised_predictions, supervised_logs, attention = supervised_oof(
        transitions, curves, scalar_columns, active_tasks, splits, epochs, patience
    )
    ts_predictions, ts_logs = ts2vec_oof(
        transitions, curves, scalar_columns, splits, cache_dir, ts2vec_iters
    )
    candidate_predictions = pd.concat([supervised_predictions, ts_predictions], ignore_index=True)
    if args.smoke:
        candidate_predictions.to_parquet(output / "round2_smoke_predictions.parquet", index=False)
        pd.concat([supervised_logs, ts_logs], ignore_index=True).to_csv(
            output / "round2_smoke_log.csv", index=False
        )
        attention.to_parquet(output / "round2_smoke_attention.parquet", index=False)
        print(candidate_predictions.groupby("model").size())
        return 0

    predictions = add_references(candidate_predictions)
    completeness = predictions.groupby("model")["transition_id"].nunique()
    if not (completeness == 238).all():
        raise RuntimeError(f"Incomplete OOF predictions: {completeness.to_dict()}")
    metrics, _ = core.evaluate_predictions(predictions, bootstraps, SEED + 1010000)
    candidate_models = [
        model
        for model in predictions["model"].unique()
        if model not in {
            "clinical_ridge",
            "current_cnn",
            "mantis_random_frozen_curves_scalars",
            "moment_small_frozen_curves_scalars",
        }
    ]
    comparisons = [
        (f"{model}_vs_current_cnn", "current_cnn", model) for model in candidate_models
    ]
    deltas = qc.paired_variant_deltas(
        predictions, comparisons, bootstraps, SEED + 1020000
    )
    ablation_comparisons = [
        ("mambasl_scalars", "mambasl_curves", "mambasl_curves_scalars"),
        ("timemil_scalars", "timemil_curves", "timemil_curves_scalars"),
        ("ts2vec_scalars", "ts2vec_ssl_curves", "ts2vec_ssl_curves_scalars"),
        ("ts2vec_random_scalars", "ts2vec_random_curves", "ts2vec_random_curves_scalars"),
        ("ts2vec_ssl_value_curves", "ts2vec_random_curves", "ts2vec_ssl_curves"),
        ("ts2vec_ssl_value_scalars", "ts2vec_random_curves_scalars", "ts2vec_ssl_curves_scalars"),
        ("timemil_vs_mambasl_curves", "mambasl_curves", "timemil_curves"),
        ("timemil_vs_mambasl_scalars", "mambasl_curves_scalars", "timemil_curves_scalars"),
        ("timemil_vs_scalar", "scalar_mlp", "timemil_curves_scalars"),
        ("mambasl_vs_scalar", "scalar_mlp", "mambasl_curves_scalars"),
    ]
    ablations = qc.paired_variant_deltas(
        predictions, ablation_comparisons, bootstraps, SEED + 1030000
    )

    predictions.to_parquet(output / "round2_oof_predictions.parquet", index=False)
    metrics.to_csv(output / "round2_metrics.csv", index=False)
    deltas.to_csv(output / "round2_paired_deltas_vs_cnn.csv", index=False)
    ablations.to_csv(output / "round2_ablation_deltas.csv", index=False)
    pd.concat([supervised_logs, ts_logs], ignore_index=True).to_csv(
        output / "round2_training_log.csv", index=False
    )
    attention.to_parquet(output / "timemil_oof_attention.parquet", index=False)
    (
        attention.groupby(["model", "segment", "temporal_patch"], as_index=False)["weight"]
        .agg(["mean", "std"])
        .reset_index()
        .to_csv(output / "timemil_attention_summary.csv", index=False)
    )
    make_figure(output, predictions, metrics)
    write_report(output, metrics, deltas, ablations, attention)
    metadata = {
        "task": PRIMARY_TASK,
        "patients": int(transitions["patient_id"].nunique()),
        "eligible_transitions": 238,
        "events": 49,
        "event_rate": 49 / 238,
        "curve_shape": list(curves.shape),
        "scalar_features": len(scalar_columns),
        "cv": "3 repeated 5-fold patient-held-out",
        "bootstraps": bootstraps,
        "ts2vec_iterations_per_fold": ts2vec_iters,
        "ts2vec_output_dimensions": 128,
        "device": torch.cuda.get_device_name(0),
        "mambasl_scan": "pure PyTorch selective-state recurrence; official fused kernel unavailable on Windows",
    }
    (output / "round2_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics.sort_values("average_precision", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
