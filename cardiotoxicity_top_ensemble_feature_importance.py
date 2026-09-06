from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from xgboost import DMatrix, XGBClassifier

import cardiotoxicity_cnn_channel_ablation as channel
import cardiotoxicity_next_visit_gpu as core
import cardiotoxicity_timeseries_round1 as round1
import cardiotoxicity_timeseries_round4 as round4


ROOT = Path(r"D:\us")
ROUND4 = ROOT / "cardiotoxicity_timeseries_round4_results"
OUTPUT = ROOT / "cardiotoxicity_top_ensemble_feature_importance_results"
PRIMARY_TASK = "mid_first_rel15"
SEED = 20260722
PERMUTATIONS = 5
BOOTSTRAPS = 1000
CHANNEL_NAMES = [
    "current_endo",
    "current_mid",
    "current_endo_minus_mid",
    "change_endo",
    "change_mid",
    "change_endo_minus_mid",
]
RDST_TYPES = ["minimum_distance", "best_match_location", "shapelet_occurrence"]
ENSEMBLES = {
    "CNN + MOMENT + RDST": [
        "current_cnn",
        "moment_small_frozen_curves_scalars",
        "rdst_shapelet_curves_scalars",
    ],
    "CNN + MOMENT + Catch22": [
        "current_cnn",
        "moment_small_frozen_curves_scalars",
        "catch22_xgb_curves_scalars",
    ],
}
STAGE_MODEL = {
    "cnn": "current_cnn",
    "moment": "moment_small_frozen_curves_scalars",
    "rdst": "rdst_shapelet_curves_scalars",
    "catch22": "catch22_xgb_curves_scalars",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpret the two best Round-4 ensembles")
    parser.add_argument(
        "--stage",
        choices=["all", "cnn", "moment", "catch22", "rdst", "ensemble"],
        default="all",
    )
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metric_values(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def valid_data(transitions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.flatnonzero(transitions[f"mask__{PRIMARY_TASK}"].astype(bool).to_numpy())
    return (
        valid,
        transitions.iloc[valid][f"label__{PRIMARY_TASK}"].to_numpy(int),
        transitions.iloc[valid]["patient_id"].astype(str).to_numpy(),
        transitions.iloc[valid]["transition_id"].astype(str).to_numpy(),
    )


def cluster_drop_ci(
    y: np.ndarray,
    patient: np.ndarray,
    base: np.ndarray,
    altered: np.ndarray,
    bootstraps: int,
    seed: int,
) -> dict[str, float]:
    base_auc, base_ap = metric_values(y, base)
    alt_auc, alt_ap = metric_values(y, altered)
    rng = np.random.default_rng(seed)
    auc_values: list[float] = []
    ap_values: list[float] = []
    # Precompute patient blocks once. Calling np.flatnonzero for every sampled
    # patient inside every bootstrap dominates runtime for this repeated analysis.
    unique_patients = np.unique(patient)
    patient_blocks = [np.flatnonzero(patient == value) for value in unique_patients]
    for _ in range(bootstraps):
        sampled = rng.integers(0, len(patient_blocks), size=len(patient_blocks))
        index = np.concatenate([patient_blocks[value] for value in sampled])
        if np.unique(y[index]).size < 2:
            continue
        b_auc, b_ap = metric_values(y[index], base[index])
        a_auc, a_ap = metric_values(y[index], altered[index])
        auc_values.append(b_auc - a_auc)
        ap_values.append(b_ap - a_ap)
    return {
        "base_auc": base_auc,
        "base_ap": base_ap,
        "auc_drop": base_auc - alt_auc,
        "auc_ci_low": float(np.quantile(auc_values, 0.025)),
        "auc_ci_high": float(np.quantile(auc_values, 0.975)),
        "ap_drop": base_ap - alt_ap,
        "ap_ci_low": float(np.quantile(ap_values, 0.025)),
        "ap_ci_high": float(np.quantile(ap_values, 0.975)),
    }


class PermutationAccumulator:
    def __init__(
        self,
        transitions: pd.DataFrame,
        items: list[str],
        permutations: int,
    ) -> None:
        self.transitions = transitions
        self.items = items
        self.permutations = permutations
        self.base_sum = np.zeros(len(transitions), dtype=np.float64)
        self.count = np.zeros(len(transitions), dtype=np.int16)
        self.permuted_sum = np.zeros(
            (len(items), permutations, len(transitions)), dtype=np.float64
        )
        self.item_index = {item: index for index, item in enumerate(items)}

    def add_base(self, index: np.ndarray, score: np.ndarray) -> None:
        self.base_sum[index] += score
        self.count[index] += 1

    def add(self, item: str, index: np.ndarray, scores: np.ndarray) -> None:
        # Index the item first so NumPy does not move the advanced index axis
        # ahead of the permutation axis.
        self.permuted_sum[self.item_index[item]][:, index] += scores

    def finish(
        self,
        stage: str,
        bootstraps: int,
        top_ci: int = 25,
    ) -> pd.DataFrame:
        valid, y, patient, transition_id = valid_data(self.transitions)
        if np.any(self.count[valid] == 0):
            raise RuntimeError(f"{stage}: some eligible transitions have no OOF score")
        base = self.base_sum[valid] / self.count[valid]
        perturbed_repeats = self.permuted_sum[:, :, valid] / self.count[valid][None, None, :]
        perturbed = perturbed_repeats.mean(axis=1)
        base_auc, base_ap = metric_values(y, base)
        rows = []
        for item_index, item in enumerate(self.items):
            perm_auc = np.asarray(
                [roc_auc_score(y, x) for x in perturbed_repeats[item_index]], dtype=float
            )
            perm_ap = np.asarray(
                [average_precision_score(y, x) for x in perturbed_repeats[item_index]], dtype=float
            )
            alt_auc, alt_ap = metric_values(y, perturbed[item_index])
            rows.append(
                {
                    "stage": stage,
                    "item": item,
                    "base_auc": base_auc,
                    "base_ap": base_ap,
                    "auc_drop": base_auc - alt_auc,
                    "auc_drop_permutation_mean": base_auc - float(perm_auc.mean()),
                    "auc_drop_permutation_low": base_auc - float(np.quantile(perm_auc, 0.975)),
                    "auc_drop_permutation_high": base_auc - float(np.quantile(perm_auc, 0.025)),
                    "ap_drop": base_ap - alt_ap,
                    "ap_drop_permutation_mean": base_ap - float(perm_ap.mean()),
                    "ap_drop_permutation_low": base_ap - float(np.quantile(perm_ap, 0.975)),
                    "ap_drop_permutation_high": base_ap - float(np.quantile(perm_ap, 0.025)),
                }
            )
        result = pd.DataFrame(rows)
        candidates = set(result.nlargest(top_ci, "auc_drop")["item"]) | set(
            result.nlargest(top_ci, "ap_drop")["item"]
        )
        for row_index, row in result.iterrows():
            if row["item"] not in candidates:
                continue
            ci = cluster_drop_ci(
                y,
                patient,
                base,
                perturbed[self.item_index[row["item"]]],
                bootstraps,
                stable_seed(SEED, stage, row["item"], "bootstrap"),
            )
            for key in ["auc_ci_low", "auc_ci_high", "ap_ci_low", "ap_ci_high"]:
                result.loc[row_index, key] = ci[key]
        result["rank_auc"] = result["auc_drop"].rank(method="min", ascending=False).astype(int)
        result["rank_ap"] = result["ap_drop"].rank(method="min", ascending=False).astype(int)
        result.to_csv(OUTPUT / f"{stage}_heldout_permutation_importance.csv", index=False)
        np.savez_compressed(
            OUTPUT / f"{stage}_permuted_oof.npz",
            valid_index=valid,
            transition_id=transition_id,
            labels=y,
            patient_id=patient,
            base_score=base,
            items=np.asarray(self.items, dtype=object),
            perturbed_score=perturbed,
        )
        return result


def permuted_predictions(
    predict,
    x: np.ndarray,
    columns: np.ndarray,
    permutations: int,
    seed_parts: tuple[object, ...],
) -> np.ndarray:
    scores = []
    for repeat in range(permutations):
        rng = np.random.default_rng(stable_seed(*seed_parts, repeat))
        order = rng.permutation(len(x))
        altered = x.copy()
        altered[:, columns] = x[order][:, columns]
        scores.append(predict(altered))
    return np.asarray(scores)


def load_all():
    transitions, curves, scalar_columns, active_tasks, splits = round1.load_inputs()
    return transitions, curves, scalar_columns, active_tasks, splits


def run_moment(permutations: int, bootstraps: int) -> None:
    transitions, _, scalar_columns, _, splits = load_all()
    embeddings = np.load(
        round1.CACHE / "moment_small_embeddings.npz", allow_pickle=False
    )["embeddings"]
    features = round1.aggregate_segment_embeddings(embeddings)
    dimension = embeddings.shape[-1]
    items = ["curve_all", "scalar_all"]
    items += [f"curve_channel::{name}" for name in CHANNEL_NAMES]
    items += [f"embedding_pool::{name}" for name in ["segment_mean", "segment_std", "segment_max"]]
    items += [f"scalar::{name}" for name in scalar_columns]
    accumulator = PermutationAccumulator(transitions, items, permutations)
    mask = transitions[f"mask__{PRIMARY_TASK}"].astype(bool).to_numpy()
    labels = transitions[f"label__{PRIMARY_TASK}"].to_numpy(int)
    all_curve_columns = np.arange(features.shape[1])
    curve_groups: dict[str, np.ndarray] = {}
    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        columns = []
        for statistic_index in range(3):
            start = statistic_index * 6 * dimension + channel_index * dimension
            columns.extend(range(start, start + dimension))
        curve_groups[f"curve_channel::{channel_name}"] = np.asarray(columns)
    for statistic_index, statistic in enumerate(["segment_mean", "segment_std", "segment_max"]):
        curve_groups[f"embedding_pool::{statistic}"] = np.arange(
            statistic_index * 6 * dimension, (statistic_index + 1) * 6 * dimension
        )
    started = time.time()
    for split in splits:
        train = np.flatnonzero(
            transitions["patient_id"].isin(split["train_patients"]).to_numpy() & mask
        )
        test = np.flatnonzero(
            transitions["patient_id"].isin(split["test_patients"]).to_numpy() & mask
        )
        pca = PCA(
            n_components=min(round1.PCA_COMPONENTS, len(train) - 1, features.shape[1]),
            whiten=True,
            svd_solver="randomized",
            random_state=SEED,
        )
        train_curve = pca.fit_transform(features[train]).astype(np.float32)
        test_curve = pca.transform(features[test]).astype(np.float32)
        train_scalar, test_scalar = round1.prepare_scalars(
            transitions, scalar_columns, train, test
        )
        model = LogisticRegression(
            penalty="l2",
            C=0.3,
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED + int(split["split_index"]),
        )
        model.fit(np.concatenate([train_curve, train_scalar], axis=1), labels[train])

        def predict_parts(curve_raw: np.ndarray, scalar: np.ndarray) -> np.ndarray:
            return model.predict_proba(
                np.concatenate([pca.transform(curve_raw), scalar], axis=1)
            )[:, 1]

        base = model.predict_proba(np.concatenate([test_curve, test_scalar], axis=1))[:, 1]
        accumulator.add_base(test, base)
        for item, columns in {"curve_all": all_curve_columns, **curve_groups}.items():
            values = []
            for repeat in range(permutations):
                order = np.random.default_rng(
                    stable_seed(SEED, "moment", split["split_index"], item, repeat)
                ).permutation(len(test))
                altered = features[test].copy()
                altered[:, columns] = features[test][order][:, columns]
                values.append(predict_parts(altered, test_scalar))
            accumulator.add(item, test, np.asarray(values))
        scalar_item_columns = {"scalar_all": np.arange(len(scalar_columns))}
        scalar_item_columns.update(
            {f"scalar::{name}": np.asarray([i]) for i, name in enumerate(scalar_columns)}
        )
        for item, columns in scalar_item_columns.items():
            values = []
            for repeat in range(permutations):
                order = np.random.default_rng(
                    stable_seed(SEED, "moment", split["split_index"], item, repeat)
                ).permutation(len(test))
                altered = test_scalar.copy()
                altered[:, columns] = test_scalar[order][:, columns]
                values.append(predict_parts(features[test], altered))
            accumulator.add(item, test, np.asarray(values))
        print(f"moment split={split['split_index']} elapsed={time.time()-started:.1f}s", flush=True)
    result = accumulator.finish("moment", bootstraps)
    print(result.nlargest(10, "auc_drop")[["item", "auc_drop", "ap_drop"]].to_string(index=False))


def catch22_group_specs(names: list[str], scalar_columns: list[str]) -> dict[str, np.ndarray]:
    specs: dict[str, list[int]] = {
        "curve_all": list(range(len(names))),
        "scalar_all": list(range(len(names), len(names) + len(scalar_columns))),
    }
    for channel_index, channel_name in enumerate(CHANNEL_NAMES, start=1):
        specs[f"curve_channel::{channel_name}"] = [
            i for i, name in enumerate(names) if name.startswith(f"c{channel_index}_")
        ]
    for statistic in ["mean", "std", "min", "max", "median"]:
        specs[f"segment_aggregation::{statistic}"] = [
            i for i, name in enumerate(names) if f"_{statistic}_" in name
        ]
    descriptor_names = sorted({name.split("_", 2)[2] for name in names})
    for descriptor in descriptor_names:
        specs[f"catch22_descriptor::{descriptor}"] = [
            i for i, name in enumerate(names) if name.endswith(f"_{descriptor}")
        ]
    return {name: np.asarray(columns, dtype=int) for name, columns in specs.items()}


def run_catch22(permutations: int, bootstraps: int) -> None:
    transitions, curves, scalar_columns, _, splits = load_all()
    features, names = round4.catch22_features(
        curves, ROUND4 / "catch22_structured_features.npz"
    )
    mask = transitions[f"mask__{PRIMARY_TASK}"].astype(bool).to_numpy()
    labels = transitions[f"label__{PRIMARY_TASK}"].to_numpy(int)
    all_names = names + scalar_columns
    native_sum = np.zeros(len(all_names), dtype=float)
    native_count = np.zeros(len(all_names), dtype=int)
    artifacts = []
    started = time.time()
    for split in splits:
        train = np.flatnonzero(
            transitions["patient_id"].isin(split["train_patients"]).to_numpy() & mask
        )
        test = np.flatnonzero(
            transitions["patient_id"].isin(split["test_patients"]).to_numpy() & mask
        )
        train_scalar, test_scalar = round1.prepare_scalars(
            transitions, scalar_columns, train, test
        )
        train_x = np.concatenate([features[train], train_scalar], axis=1)
        test_x = np.concatenate([features[test], test_scalar], axis=1)
        weight = float((len(train) - labels[train].sum()) / max(labels[train].sum(), 1))
        for seed_index in range(3):
            run_seed = SEED + int(split["split_index"]) * 101 + seed_index * 10007
            model = XGBClassifier(
                n_estimators=300,
                max_depth=2,
                learning_rate=0.03,
                min_child_weight=3,
                subsample=0.80,
                colsample_bytree=0.70,
                reg_alpha=1.0,
                reg_lambda=5.0,
                scale_pos_weight=weight,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=8,
                random_state=run_seed,
            )
            model.fit(train_x, labels[train])
            base = model.predict_proba(test_x)[:, 1]
            shap = model.get_booster().predict(DMatrix(test_x), pred_contribs=True)[:, :-1]
            native_sum += np.abs(shap).sum(axis=0)
            native_count += len(test)
            artifacts.append(
                {
                    "split": int(split["split_index"]),
                    "seed": seed_index,
                    "test": test,
                    "x": test_x,
                    "model": model,
                    "base": base,
                }
            )
        print(f"catch22 split={split['split_index']} elapsed={time.time()-started:.1f}s", flush=True)
    native = pd.DataFrame(
        {
            "feature": all_names,
            "feature_source": ["catch22" if i < len(names) else "scalar" for i in range(len(all_names))],
            "mean_abs_oof_tree_shap": native_sum / np.maximum(native_count, 1),
        }
    ).sort_values("mean_abs_oof_tree_shap", ascending=False)
    native["native_rank"] = np.arange(1, len(native) + 1)
    native.to_csv(OUTPUT / "catch22_native_tree_shap.csv", index=False)
    top_features = native.head(30)["feature"].tolist()
    specs = catch22_group_specs(names, scalar_columns)
    for feature in top_features:
        specs[f"feature::{feature}"] = np.asarray([all_names.index(feature)])
    items = list(specs)
    accumulator = PermutationAccumulator(transitions, items, permutations)
    for artifact in artifacts:
        accumulator.add_base(artifact["test"], artifact["base"])
        altered_batches = []
        for item, columns in specs.items():
            for repeat in range(permutations):
                order = np.random.default_rng(
                    stable_seed(
                        SEED,
                        "catch22",
                        artifact["split"],
                        artifact["seed"],
                        item,
                        repeat,
                    )
                ).permutation(len(artifact["x"]))
                altered = artifact["x"].copy()
                altered[:, columns] = artifact["x"][order][:, columns]
                altered_batches.append(altered)
        # A single batched XGBoost call avoids thousands of small threaded
        # predict calls and gives exactly the same scores.
        all_scores = artifact["model"].predict_proba(np.concatenate(altered_batches))[:, 1]
        all_scores = all_scores.reshape(len(items), permutations, len(artifact["test"]))
        for item_index, item in enumerate(items):
            score = all_scores[item_index]
            accumulator.add(item, artifact["test"], score)
    result = accumulator.finish("catch22", bootstraps, top_ci=30)
    print(result.nlargest(12, "auc_drop")[["item", "auc_drop", "ap_drop"]].to_string(index=False))


def prepare_rdst_curves(curves: np.ndarray) -> np.ndarray:
    flat = curves.reshape(len(curves), 18 * 6, 96).copy()
    neutral = np.random.default_rng(0).standard_normal(96).astype(np.float32)
    neutral = neutral / neutral.std() * 1e-3
    flat += neutral[None, None, :]
    return (flat * 100.0).astype(np.float64, copy=False)


def run_rdst(permutations: int, bootstraps: int) -> None:
    transitions, curves, scalar_columns, _, splits = load_all()
    flat = prepare_rdst_curves(curves)
    mask = transitions[f"mask__{PRIMARY_TASK}"].astype(bool).to_numpy()
    labels = transitions[f"label__{PRIMARY_TASK}"].to_numpy(int)
    items = ["shapelet_all", "scalar_all"]
    items += [f"shapelet_output::{name}" for name in RDST_TYPES]
    items += [f"scalar::{name}" for name in scalar_columns]
    accumulator = PermutationAccumulator(transitions, items, permutations)
    scalar_native_rows = []
    shapelet_rows = []
    attribution_sum = np.zeros(18 * 6, dtype=float)
    attribution_runs = 0
    started = time.time()
    for split in splits:
        train = np.flatnonzero(
            transitions["patient_id"].isin(split["train_patients"]).to_numpy() & mask
        )
        test = np.flatnonzero(
            transitions["patient_id"].isin(split["test_patients"]).to_numpy() & mask
        )
        train_scalar, test_scalar = round1.prepare_scalars(
            transitions, scalar_columns, train, test
        )
        for seed_index in range(3):
            run_seed = SEED + int(split["split_index"]) * 101 + seed_index * 10007
            transform = RandomDilatedShapeletTransform(
                max_shapelets=1200, random_state=run_seed, n_jobs=-1
            )
            train_shapelet = transform.fit_transform(flat[train], labels[train])
            test_shapelet = transform.transform(flat[test])
            train_x = np.concatenate([train_shapelet, train_scalar], axis=1)
            test_x = np.concatenate([test_shapelet, test_scalar], axis=1)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=3000,
                    random_state=run_seed,
                ),
            )
            model.fit(train_x, labels[train])
            base = model.predict_proba(test_x)[:, 1]
            accumulator.add_base(test, base)
            n_shapelets = transform.n_shapelets_
            group_specs = {
                "shapelet_all": np.arange(3 * n_shapelets),
                "scalar_all": np.arange(3 * n_shapelets, 3 * n_shapelets + len(scalar_columns)),
            }
            for type_index, type_name in enumerate(RDST_TYPES):
                group_specs[f"shapelet_output::{type_name}"] = np.arange(type_index, 3 * n_shapelets, 3)
            for scalar_index, scalar_name in enumerate(scalar_columns):
                group_specs[f"scalar::{scalar_name}"] = np.asarray([3 * n_shapelets + scalar_index])
            spec_items = list(group_specs.items())
            for chunk_start in range(0, len(spec_items), 10):
                chunk = spec_items[chunk_start : chunk_start + 10]
                batches = []
                for item, columns in chunk:
                    for repeat in range(permutations):
                        order = np.random.default_rng(
                            stable_seed(
                                SEED,
                                "rdst",
                                split["split_index"],
                                seed_index,
                                item,
                                repeat,
                            )
                        ).permutation(len(test_x))
                        altered = test_x.copy()
                        altered[:, columns] = test_x[order][:, columns]
                        batches.append(altered)
                scores = model.predict_proba(np.concatenate(batches))[:, 1]
                scores = scores.reshape(len(chunk), permutations, len(test))
                for local_index, (item, _) in enumerate(chunk):
                    accumulator.add(item, test, scores[local_index])

            coef = model.named_steps["logisticregression"].coef_[0]
            for scalar_index, scalar_name in enumerate(scalar_columns):
                value = float(coef[3 * n_shapelets + scalar_index])
                scalar_native_rows.append(
                    {
                        "split": int(split["split_index"]),
                        "seed_index": seed_index,
                        "feature": scalar_name,
                        "coefficient": value,
                        "absolute_coefficient": abs(value),
                    }
                )
            values, starts, lengths, dilations, thresholds, normalizes, _, _, classes = transform.shapelets_
            run_attribution = np.zeros(18 * 6, dtype=float)
            for shapelet_index in range(n_shapelets):
                coefficients = coef[3 * shapelet_index : 3 * shapelet_index + 3]
                importance = float(np.abs(coefficients).sum())
                active_values = np.abs(values[shapelet_index, :, : int(lengths[shapelet_index])])
                energy = active_values.mean(axis=1)
                energy_sum = float(energy.sum())
                fraction = energy / energy_sum if energy_sum > 0 else np.full(108, 1 / 108)
                dominant = np.argsort(fraction)[-5:][::-1]
                shapelet_rows.append(
                    {
                        "split": int(split["split_index"]),
                        "seed_index": seed_index,
                        "shapelet_index": shapelet_index,
                        "importance_abs_standardized_coefficient_sum": importance,
                        "coef_minimum_distance": float(coefficients[0]),
                        "coef_best_match_location": float(coefficients[1]),
                        "coef_shapelet_occurrence": float(coefficients[2]),
                        "startpoint": int(starts[shapelet_index]),
                        "length_points": int(lengths[shapelet_index]),
                        "dilation": int(dilations[shapelet_index]),
                        "effective_span": int((lengths[shapelet_index] - 1) * dilations[shapelet_index] + 1),
                        "normalized": bool(normalizes[shapelet_index]),
                        "threshold": float(thresholds[shapelet_index]),
                        "source_class": int(classes[shapelet_index]),
                        "dominant_inputs": ";".join(
                            f"S{index // 6 + 1:02d}:{CHANNEL_NAMES[index % 6]}"
                            for index in dominant
                        ),
                    }
                )
                run_attribution += importance * fraction
            attribution_sum += run_attribution
            attribution_runs += 1
        print(f"rdst split={split['split_index']} elapsed={time.time()-started:.1f}s", flush=True)
    pd.DataFrame(scalar_native_rows).to_csv(OUTPUT / "rdst_scalar_native_coefficients.csv", index=False)
    pd.DataFrame(shapelet_rows).sort_values(
        "importance_abs_standardized_coefficient_sum", ascending=False
    ).to_csv(OUTPUT / "rdst_shapelet_native_importance.csv", index=False)
    attribution = pd.DataFrame(
        {
            "segment": np.arange(18 * 6) // 6 + 1,
            "curve_channel": [CHANNEL_NAMES[index % 6] for index in range(18 * 6)],
            "coefficient_energy_attribution": attribution_sum / max(attribution_runs, 1),
        }
    )
    attribution.groupby("curve_channel", as_index=False)["coefficient_energy_attribution"].mean().sort_values(
        "coefficient_energy_attribution", ascending=False
    ).to_csv(OUTPUT / "rdst_curve_channel_attribution.csv", index=False)
    attribution.groupby("segment", as_index=False)["coefficient_energy_attribution"].mean().sort_values(
        "coefficient_energy_attribution", ascending=False
    ).to_csv(OUTPUT / "rdst_segment_attribution.csv", index=False)
    result = accumulator.finish("rdst", bootstraps, top_ci=30)
    print(result.nlargest(12, "auc_drop")[["item", "auc_drop", "ap_drop"]].to_string(index=False))


def train_current_cnn_and_permute(permutations: int, bootstraps: int) -> None:
    transitions, curves, scalar_columns, active_tasks, splits = load_all()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CNN importance stage")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    labels = np.column_stack(
        [transitions[f"label__{task.name}"].to_numpy(np.float32) for task in active_tasks]
    )
    masks = np.column_stack(
        [transitions[f"mask__{task.name}"].to_numpy(np.float32) for task in active_tasks]
    )
    primary_index = [task.name for task in active_tasks].index(PRIMARY_TASK)
    patient_primary = transitions.groupby("patient_id")[f"label__{PRIMARY_TASK}"].max().astype(int).to_dict()
    items = ["curve_all", "scalar_all"]
    items += [f"curve_channel::{name}" for name in CHANNEL_NAMES]
    items += [f"segment::{index:02d}" for index in range(1, 19)]
    items += [f"time_window::{start+1:02d}-{start+8:02d}" for start in range(0, 96, 8)]
    items += [f"scalar::{name}" for name in scalar_columns]
    accumulator = PermutationAccumulator(transitions, items, permutations)
    attention_rows = []
    started = time.time()
    from sklearn.model_selection import train_test_split

    for split in splits:
        split_seed = SEED + int(split["split_index"]) * 101
        train_patients = sorted(split["train_patients"])
        fit_patients, val_patients = train_test_split(
            train_patients,
            test_size=0.2,
            random_state=split_seed,
            stratify=[patient_primary.get(patient, 0) for patient in train_patients],
        )
        fit = np.flatnonzero(transitions["patient_id"].isin(fit_patients).to_numpy())
        val = np.flatnonzero(transitions["patient_id"].isin(val_patients).to_numpy())
        test = np.flatnonzero(transitions["patient_id"].isin(split["test_patients"]).to_numpy())
        fit_scalar, val_scalar = round1.prepare_scalars(transitions, scalar_columns, fit, val)
        _, test_scalar = round1.prepare_scalars(transitions, scalar_columns, fit, test)
        fit_masks = masks[fit].copy()
        for task_index in range(len(active_tasks)):
            task_y = labels[fit][fit_masks[:, task_index] > 0, task_index]
            if task_y.sum() < 2 or len(task_y) - task_y.sum() < 2:
                fit_masks[:, task_index] = 0
        positives = (labels[fit] * fit_masks).sum(axis=0)
        negatives = ((1 - labels[fit]) * fit_masks).sum(axis=0)
        pos_weight = torch.as_tensor(
            np.clip(negatives / np.maximum(positives, 1), 1, 20).astype(np.float32),
            device=device,
        )
        set_seed(split_seed)
        model = channel.ChannelAttentionNet(6, len(scalar_columns), len(active_tasks), 18).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-3)
        amp_scaler = torch.amp.GradScaler("cuda", enabled=True)
        best_state = copy.deepcopy(model.state_dict())
        best_loss = math.inf
        stale = 0
        best_epoch = 0
        for epoch in range(channel.EPOCHS):
            model.train()
            order = np.random.default_rng(split_seed + epoch).permutation(len(fit))
            for start in range(0, len(order), 64):
                local = order[start : start + 64]
                global_index = fit[local]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    logits, _ = model(
                        torch.as_tensor(curves[global_index], device=device),
                        torch.as_tensor(fit_scalar[local], device=device),
                    )
                    loss = channel.binary_loss(
                        logits,
                        torch.as_tensor(labels[global_index], device=device),
                        torch.as_tensor(fit_masks[local], device=device),
                        pos_weight,
                    )
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
            model.eval()
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True):
                logits, _ = model(
                    torch.as_tensor(curves[val], device=device),
                    torch.as_tensor(val_scalar, device=device),
                )
                val_loss = float(
                    channel.binary_loss(
                        logits,
                        torch.as_tensor(labels[val], device=device),
                        torch.as_tensor(masks[val], device=device),
                        pos_weight,
                    ).item()
                )
            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= channel.PATIENCE:
                break
        model.load_state_dict(best_state)
        model.eval()

        def predict(curve_values: np.ndarray, scalar_values: np.ndarray):
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True):
                logits, weights = model(
                    torch.as_tensor(curve_values, device=device),
                    torch.as_tensor(scalar_values, device=device),
                )
            return torch.sigmoid(logits[:, primary_index]).float().cpu().numpy(), weights.float().cpu().numpy()

        primary_valid = masks[test, primary_index] > 0
        selected = test[primary_valid]
        base_all, weights = predict(curves[test], test_scalar)
        accumulator.add_base(selected, base_all[primary_valid])
        for local_index, global_index in enumerate(test):
            if primary_valid[local_index]:
                for segment, weight in enumerate(weights[local_index], start=1):
                    attention_rows.append(
                        {"split": int(split["split_index"]), "transition_index": int(global_index), "segment": segment, "attention_weight": float(weight)}
                    )

        curve_specs: dict[str, tuple[str, object]] = {"curve_all": ("all", None)}
        curve_specs.update({f"curve_channel::{name}": ("channel", i) for i, name in enumerate(CHANNEL_NAMES)})
        curve_specs.update({f"segment::{i+1:02d}": ("segment", i) for i in range(18)})
        curve_specs.update({f"time_window::{s+1:02d}-{s+8:02d}": ("time", slice(s, s + 8)) for s in range(0, 96, 8)})
        for item, (kind, target) in curve_specs.items():
            values = []
            for repeat in range(permutations):
                order = np.random.default_rng(stable_seed(SEED, "cnn", split["split_index"], item, repeat)).permutation(len(test))
                altered = curves[test].copy()
                if kind == "all":
                    altered = curves[test][order].copy()
                elif kind == "channel":
                    altered[:, :, int(target), :] = curves[test][order, :, int(target), :]
                elif kind == "segment":
                    altered[:, int(target), :, :] = curves[test][order, int(target), :, :]
                else:
                    altered[:, :, :, target] = curves[test][order, :, :, target]
                score, _ = predict(altered, test_scalar)
                values.append(score[primary_valid])
            accumulator.add(item, selected, np.asarray(values))
        scalar_specs = {"scalar_all": np.arange(len(scalar_columns))}
        scalar_specs.update({f"scalar::{name}": np.asarray([i]) for i, name in enumerate(scalar_columns)})
        for item, columns in scalar_specs.items():
            values = []
            for repeat in range(permutations):
                order = np.random.default_rng(stable_seed(SEED, "cnn", split["split_index"], item, repeat)).permutation(len(test))
                altered = test_scalar.copy()
                altered[:, columns] = test_scalar[order][:, columns]
                score, _ = predict(curves[test], altered)
                values.append(score[primary_valid])
            accumulator.add(item, selected, np.asarray(values))
        print(f"cnn split={split['split_index']} epoch={best_epoch} elapsed={time.time()-started:.1f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(attention_rows).to_csv(OUTPUT / "cnn_oof_segment_attention.csv", index=False)
    result = accumulator.finish("cnn", bootstraps, top_ci=30)
    print(result.nlargest(12, "auc_drop")[["item", "auc_drop", "ap_drop"]].to_string(index=False))


def subset_value(y: np.ndarray, score_frame: pd.DataFrame, members: tuple[str, ...]) -> tuple[float, float]:
    if not members:
        return 0.5, float(y.mean())
    return metric_values(y, score_frame[list(members)].mean(axis=1).to_numpy())


def exact_shapley(y: np.ndarray, scores: pd.DataFrame, members: list[str]) -> dict[str, tuple[float, float]]:
    n = len(members)
    output = {member: [0.0, 0.0] for member in members}
    for member in members:
        others = [x for x in members if x != member]
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for subset in combinations(others, size):
                before = subset_value(y, scores, tuple(subset))
                after = subset_value(y, scores, tuple((*subset, member)))
                output[member][0] += weight * (after[0] - before[0])
                output[member][1] += weight * (after[1] - before[1])
    return {key: (float(value[0]), float(value[1])) for key, value in output.items()}


def load_stage_npz(stage: str) -> dict[str, object]:
    payload = np.load(OUTPUT / f"{stage}_permuted_oof.npz", allow_pickle=True)
    return {key: payload[key] for key in payload.files}


def ensemble_analysis(bootstraps: int) -> None:
    predictions = pd.read_parquet(ROUND4 / "round4_oof_predictions.parquet")
    template = predictions[predictions["model"].eq("current_cnn")].sort_values("transition_id")
    transition_id = template["transition_id"].astype(str).to_numpy()
    y = template["label"].to_numpy(int)
    patient = template["patient_id"].astype(str).to_numpy()
    pivot = predictions.pivot(index="transition_id", columns="model", values="score").loc[transition_id]
    contribution_rows = []
    bootstrap_records: dict[tuple[str, str], dict[str, list[float]]] = {}
    for ensemble_name, members in ENSEMBLES.items():
        full_auc, full_ap = subset_value(y, pivot, tuple(members))
        shapley = exact_shapley(y, pivot, members)
        for member in members:
            pair = [x for x in members if x != member]
            pair_auc, pair_ap = subset_value(y, pivot, tuple(pair))
            contribution_rows.append(
                {
                    "ensemble": ensemble_name,
                    "component": member,
                    "full_auc": full_auc,
                    "full_ap": full_ap,
                    "leave_one_out_auc_drop": full_auc - pair_auc,
                    "leave_one_out_ap_drop": full_ap - pair_ap,
                    "shapley_auc": shapley[member][0],
                    "shapley_ap": shapley[member][1],
                }
            )
            bootstrap_records[(ensemble_name, member)] = {
                "loo_auc": [], "loo_ap": [], "shap_auc": [], "shap_ap": []
            }
    rng = np.random.default_rng(stable_seed(SEED, "ensemble", "bootstrap"))
    for _ in range(bootstraps):
        index = core.cluster_sample_indices(patient, rng)
        if np.unique(y[index]).size < 2:
            continue
        boot_y = y[index]
        boot_scores = pivot.iloc[index].reset_index(drop=True)
        for ensemble_name, members in ENSEMBLES.items():
            full = subset_value(boot_y, boot_scores, tuple(members))
            shapley = exact_shapley(boot_y, boot_scores, members)
            for member in members:
                pair = subset_value(boot_y, boot_scores, tuple(x for x in members if x != member))
                record = bootstrap_records[(ensemble_name, member)]
                record["loo_auc"].append(full[0] - pair[0])
                record["loo_ap"].append(full[1] - pair[1])
                record["shap_auc"].append(shapley[member][0])
                record["shap_ap"].append(shapley[member][1])
    contributions = pd.DataFrame(contribution_rows)
    for row_index, row in contributions.iterrows():
        record = bootstrap_records[(row["ensemble"], row["component"])]
        for key, output_name in [
            ("loo_auc", "leave_one_out_auc"), ("loo_ap", "leave_one_out_ap"),
            ("shap_auc", "shapley_auc"), ("shap_ap", "shapley_ap"),
        ]:
            contributions.loc[row_index, f"{output_name}_ci_low"] = np.quantile(record[key], 0.025)
            contributions.loc[row_index, f"{output_name}_ci_high"] = np.quantile(record[key], 0.975)
    contributions.to_csv(OUTPUT / "ensemble_component_importance.csv", index=False)

    stage_payload = {stage: load_stage_npz(stage) for stage in STAGE_MODEL}
    reconstruction_rows = []
    for stage, model_name in STAGE_MODEL.items():
        payload = stage_payload[stage]
        order = pd.Series(np.arange(len(payload["transition_id"])), index=payload["transition_id"].astype(str)).loc[transition_id].to_numpy()
        reconstructed = payload["base_score"][order]
        saved = pivot[model_name].to_numpy()
        reconstruction_rows.append(
            {
                "stage": stage,
                "model": model_name,
                "saved_auc": roc_auc_score(y, saved),
                "reconstructed_auc": roc_auc_score(y, reconstructed),
                "saved_ap": average_precision_score(y, saved),
                "reconstructed_ap": average_precision_score(y, reconstructed),
                "score_correlation": np.corrcoef(saved, reconstructed)[0, 1],
                "mean_absolute_score_difference": np.abs(saved - reconstructed).mean(),
            }
        )
        payload["aligned_order"] = order
    pd.DataFrame(reconstruction_rows).to_csv(OUTPUT / "constituent_reproduction_check.csv", index=False)

    ensemble_feature_rows = []
    ensemble_perturbed: dict[tuple[str, str], np.ndarray] = {}
    for ensemble_name, members in ENSEMBLES.items():
        full = pivot[members].mean(axis=1).to_numpy()
        for stage, model_name in STAGE_MODEL.items():
            if model_name not in members:
                continue
            payload = stage_payload[stage]
            order = payload["aligned_order"]
            base_rebuilt = payload["base_score"][order]
            for item_index, item in enumerate(payload["items"].astype(str)):
                altered_component = pivot[model_name].to_numpy() + payload["perturbed_score"][item_index, order] - base_rebuilt
                altered_ensemble = full + (altered_component - pivot[model_name].to_numpy()) / 3.0
                auc, ap = metric_values(y, altered_ensemble)
                full_auc, full_ap = metric_values(y, full)
                key = f"{stage}::{item}"
                ensemble_feature_rows.append(
                    {
                        "ensemble": ensemble_name,
                        "importance_scope": "one_component",
                        "component": model_name,
                        "item": item,
                        "display_item": key,
                        "auc_drop": full_auc - auc,
                        "ap_drop": full_ap - ap,
                    }
                )
                ensemble_perturbed[(ensemble_name, key)] = altered_ensemble

        # Joint scalar permutations provide the closest analogue to a single
        # engineered feature used simultaneously by all three constituent models.
        for scalar_name in round1.load_inputs()[2]:
            deltas = np.zeros(len(y), dtype=float)
            available = True
            for stage, model_name in STAGE_MODEL.items():
                if model_name not in members:
                    continue
                payload = stage_payload[stage]
                item_name = f"scalar::{scalar_name}"
                lookup = {name: i for i, name in enumerate(payload["items"].astype(str))}
                if item_name not in lookup:
                    available = False
                    break
                order = payload["aligned_order"]
                deltas += payload["perturbed_score"][lookup[item_name], order] - payload["base_score"][order]
            if available:
                altered = full + deltas / 3.0
                full_auc, full_ap = metric_values(y, full)
                auc, ap = metric_values(y, altered)
                key = f"joint_scalar::{scalar_name}"
                ensemble_feature_rows.append(
                    {
                        "ensemble": ensemble_name,
                        "importance_scope": "joint_across_components",
                        "component": "all three",
                        "item": scalar_name,
                        "display_item": key,
                        "auc_drop": full_auc - auc,
                        "ap_drop": full_ap - ap,
                    }
                )
                ensemble_perturbed[(ensemble_name, key)] = altered
    importance = pd.DataFrame(ensemble_feature_rows)
    importance["rank_auc"] = importance.groupby("ensemble")["auc_drop"].rank(method="min", ascending=False).astype(int)
    importance["rank_ap"] = importance.groupby("ensemble")["ap_drop"].rank(method="min", ascending=False).astype(int)
    ci_keys = set()
    for ensemble_name, group in importance.groupby("ensemble"):
        ci_keys |= {(ensemble_name, key) for key in group.nlargest(25, "auc_drop")["display_item"]}
        ci_keys |= {(ensemble_name, key) for key in group.nlargest(25, "ap_drop")["display_item"]}
    for row_index, row in importance.iterrows():
        key = (row["ensemble"], row["display_item"])
        if key not in ci_keys:
            continue
        full = pivot[ENSEMBLES[row["ensemble"]]].mean(axis=1).to_numpy()
        ci = cluster_drop_ci(
            y, patient, full, ensemble_perturbed[key], bootstraps,
            stable_seed(SEED, "ensemble_feature", *key),
        )
        for name in ["auc_ci_low", "auc_ci_high", "ap_ci_low", "ap_ci_high"]:
            importance.loc[row_index, name] = ci[name]
    importance.to_csv(OUTPUT / "ensemble_feature_importance.csv", index=False)
    make_outputs(contributions, importance)


def make_outputs(contributions: pd.DataFrame, importance: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, metric, title in zip(
        axes,
        ["shapley_auc", "shapley_ap"],
        ["Exact model-level Shapley value: AUC", "Exact model-level Shapley value: AP"],
    ):
        pivot = contributions.pivot(index="component", columns="ensemble", values=metric).fillna(0)
        pivot.plot(kind="barh", ax=axis)
        axis.set_title(title)
        axis.set_xlabel("Contribution above random baseline")
        axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT / "ensemble_component_shapley.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(15, 7))
    for axis, (ensemble_name, group) in zip(axes, importance.groupby("ensemble", sort=False)):
        top = group.nlargest(12, "auc_drop").sort_values("auc_drop")
        axis.barh(top["display_item"], top["auc_drop"], color="#277da1")
        axis.set_title(ensemble_name)
        axis.set_xlabel("Held-out AUC decrease after permutation")
        axis.grid(axis="x", alpha=0.2)
        axis.tick_params(axis="y", labelsize=7)
    figure.tight_layout()
    figure.savefig(OUTPUT / "top_ensemble_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    lines = [
        "# Feature importance of the two best Round-4 ensembles",
        "",
        "Importance was estimated strictly from patient-held-out predictions. Individual inputs were shuffled in the held-out fold; the reported decrease is performance lost after shuffling. Model-level contributions use both leave-one-model-out ablation and exact three-player Shapley values. Confidence intervals are patient-cluster bootstraps.",
        "",
        "## Model-level contribution",
        "",
        round1.markdown_table(
            contributions,
            [
                "ensemble",
                "component",
                "full_auc",
                "full_ap",
                "leave_one_out_auc_drop",
                "leave_one_out_ap_drop",
                "shapley_auc",
                "shapley_ap",
            ],
        ),
    ]
    for ensemble_name, group in importance.groupby("ensemble", sort=False):
        columns = ["display_item", "importance_scope", "auc_drop", "auc_ci_low", "auc_ci_high", "ap_drop", "ap_ci_low", "ap_ci_high"]
        lines += [
            "",
            f"## {ensemble_name}: most valuable inputs",
            "",
            round1.markdown_table(group.nlargest(15, "auc_drop"), columns),
        ]
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Permutation importance measures reliance of this fitted model, not causality.",
        "- Correlated inputs share or mask importance; a low individual value does not prove that the physiological variable is irrelevant.",
        "- RDST shapelets are whole-heart multivariate motifs. Their segment/channel attribution is coefficient-weighted shapelet energy, a descriptive native attribution rather than a held-out permutation result.",
        "- Deep representations do not expose named coefficients. CNN importance is grouped raw-input permutation; MOMENT importance is permutation of frozen embedding groups before fold-specific PCA.",
    ]
    (OUTPUT / "top_ensemble_feature_importance_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stages = ["cnn", "moment", "catch22", "rdst"] if args.stage == "all" else [args.stage]
    if "cnn" in stages:
        train_current_cnn_and_permute(args.permutations, args.bootstraps)
    if "moment" in stages:
        run_moment(args.permutations, args.bootstraps)
    if "catch22" in stages:
        run_catch22(args.permutations, args.bootstraps)
    if "rdst" in stages:
        run_rdst(args.permutations, args.bootstraps)
    if args.stage in {"all", "ensemble"}:
        ensemble_analysis(args.bootstraps)
    metadata = {
        "task": PRIMARY_TASK,
        "permutations": args.permutations,
        "patient_cluster_bootstraps": args.bootstraps,
        "seed": SEED,
        "stage": args.stage,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
