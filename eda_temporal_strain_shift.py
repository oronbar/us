"""
Exploratory analysis: how do strain curves shift from a patient's earliest scan
to their later scans?

Given the preprocessed parquet (same columns used by patient_conditioned_strain_model.py),
the script:
 1) Builds per-cycle curves (averaging the four required segments after resampling).
 2) For each patient (+ optional view), compares the earliest scan date vs all later scans.
 3) Computes simple metrics (MAE, RMSE, Pearson r) between the early-mean and late-mean curves.
 4) (Optional) Extracts embeddings (mean curve or model visit embedding), runs t-SNE/UMAP, a classifier probe, and distance-to-earliest diagnostics for separability.
 5) Saves a summary CSV and plots (histograms, overlays, projections).

Usage (PowerShell / bash):
python eda_temporal_strain_shift.py --parquet path/to/strain_dataset.parquet --outdir eda_out

Requirements: pandas, numpy, matplotlib, scikit-learn
Optional: torch + patient_conditioned_strain_model checkpoint (for model embeddings), umap-learn (for UMAP)
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Segment order constants reused from the model script
SEG_ORDER_4C = [
    "basal_inferoseptal",
    "mid_inferoseptal",
    "mid_anterolateral",
    "basal_anterolateral",
]
SEG_ORDER_2C = [
    "basal_inferior",
    "mid_inferior",
    "mid_anterior",
    "basal_anterior",
]
VIEW_TO_SEGS = {"4C": SEG_ORDER_4C, "2C": SEG_ORDER_2C}


def resample_curve(y: List[float], T: int) -> np.ndarray:
    """Linearly resample a 1D list/array to length T."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return np.zeros(T, dtype=float)
    if n == T:
        return y.copy()
    x_old = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, T)
    return np.interp(x_new, x_old, y)


def build_cycle_curves(df: pd.DataFrame, T: int, view_filter: str | None) -> List[dict]:
    """Return per-cycle curves (per-segment + mean) with metadata."""
    if view_filter is not None:
        df = df[df["view"] == view_filter]

    req_cols = ["short_id", "view", "study_instance_uid", "cycle_index", "segment", "delta", "study_datetime"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in parquet")

    df = df.copy()
    df["study_datetime"] = pd.to_datetime(df["study_datetime"])
    df.sort_values(by=["short_id", "study_datetime", "view", "study_instance_uid", "cycle_index"], inplace=True)

    samples: List[dict] = []
    group_cols = ["short_id", "view", "study_instance_uid", "cycle_index"]
    for (sid, view, study_uid, cycle_idx), sub in df.groupby(group_cols, sort=False):
        seg_order = VIEW_TO_SEGS.get(view, SEG_ORDER_4C)
        present = set(sub["segment"].tolist())
        if not all(seg in present for seg in seg_order):
            continue  # need all required segments

        curves = []
        for seg in seg_order:
            row = sub[sub["segment"] == seg].iloc[0]
            curves.append(resample_curve(row["delta"], T))
        curves = np.stack(curves, axis=0)  # [S, T]
        mean_curve = curves.mean(axis=0)   # [T]

        samples.append(
            {
                "short_id": sid,
                "view": view,
                "study_uid": study_uid,
                "cycle_index": int(cycle_idx),
                "dt": sub["study_datetime"].iloc[0],
                "curve": mean_curve,
                "curves_full": curves,  # [S, T]
            }
        )
    return samples


def split_early_late(samples: List[dict], min_gap_days: int) -> List[dict]:
    """Group by patient/view and return early vs late aggregates with metrics."""
    out = []
    by_patient: Dict[Tuple[str, str], List[dict]] = {}
    for s in samples:
        key = (s["short_id"], s["view"])
        by_patient.setdefault(key, []).append(s)

    gap = pd.Timedelta(days=min_gap_days)
    for (sid, view), items in by_patient.items():
        items = sorted(items, key=lambda x: x["dt"])
        if not items:
            continue
        first_dt = items[0]["dt"]
        early_curves = [s["curve"] for s in items if s["dt"] == first_dt]
        late_curves = [s["curve"] for s in items if s["dt"] > (first_dt + gap)]
        if not late_curves:
            continue  # nothing to compare against

        early_mean = np.stack(early_curves, axis=0).mean(axis=0)
        late_mean = np.stack(late_curves, axis=0).mean(axis=0)

        diff = late_mean - early_mean
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        if np.allclose(np.std(early_mean), 0) or np.allclose(np.std(late_mean), 0):
            corr = np.nan
        else:
            corr = float(np.corrcoef(early_mean, late_mean)[0, 1])

        out.append(
            {
                "short_id": sid,
                "view": view,
                "n_early_cycles": len(early_curves),
                "n_late_cycles": len(late_curves),
                "mae": mae,
                "rmse": rmse,
                "pearson_r": corr,
                "early_dt": first_dt,
                "latest_dt": max(s["dt"] for s in items),
                "early_mean_curve": early_mean,
                "late_mean_curve": late_mean,
            }
        )
    return out


def label_samples_for_classification(samples: List[dict], min_gap_days: int) -> List[dict]:
    """Attach early/late labels per cycle with optional gap; drop unlabeled ones."""
    by_patient: Dict[str, List[dict]] = {}
    for s in samples:
        by_patient.setdefault(s["short_id"], []).append(s)

    labeled: List[dict] = []
    gap = pd.Timedelta(days=min_gap_days)
    for sid, items in by_patient.items():
        items = sorted(items, key=lambda x: x["dt"])
        if not items:
            continue
        first_dt = items[0]["dt"]
        for s in items:
            if s["dt"] == first_dt:
                s_labeled = dict(s)
                s_labeled["label"] = 0  # early
                labeled.append(s_labeled)
            elif s["dt"] > first_dt + gap:
                s_labeled = dict(s)
                s_labeled["label"] = 1  # late
                labeled.append(s_labeled)
            else:
                continue  # within gap, skip
    return labeled


def build_embeddings(
    labeled_samples: List[dict],
    use_model: bool,
    model_ckpt: Optional[str],
    device: str,
    batch_size: int,
    T: int,
):
    """
    Return (embeddings, meta, repr_name).
    - If model_ckpt is provided, use the visit embedding from patient_conditioned_strain_model.Model.
    - Otherwise use the mean curve (length T) as the feature.
    """
    if not labeled_samples:
        return None, [], "none"

    if use_model and model_ckpt:
        try:
            import torch
            from torch.utils.data import DataLoader
            from patient_conditioned_strain_model import Model, SEG_ID_MAP, VIEW_ID_MAP
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Falling back to mean curves because loading model failed: {exc}")
            use_model = False

    if use_model and model_ckpt:
        import torch

        # Define a tiny dataset to reuse the curves we already built
        class CurveDs(torch.utils.data.Dataset):
            def __init__(self, items):
                self.items = items

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                s = self.items[idx]
                X = torch.tensor(s["curves_full"], dtype=torch.float32)  # [S, T]
                return X, s

        def collate(batch):
            X_list, meta = zip(*batch)
            X = torch.stack(X_list, dim=0)  # [B, S, T]
            seg_ids = []
            view_ids = []
            for m in meta:
                seg_order = VIEW_TO_SEGS.get(m["view"], SEG_ORDER_4C)
                seg_ids.append(torch.tensor([SEG_ID_MAP[s] for s in seg_order], dtype=torch.long))
                view_ids.append(torch.tensor(VIEW_ID_MAP.get(m["view"], 1), dtype=torch.long))
            seg_ids = torch.stack(seg_ids, dim=0)
            view_ids = torch.stack(view_ids, dim=0)
            return X, seg_ids, view_ids, meta

        dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        model = Model(T=T)
        model.load_state_dict(torch.load(model_ckpt, map_location="cpu"))
        model.to(dev)
        model.eval()

        loader = DataLoader(CurveDs(labeled_samples), batch_size=batch_size, shuffle=False, collate_fn=collate)
        feats = []
        meta_rows = []
        with torch.no_grad():
            for X, seg_ids, view_ids, meta in loader:
                X = X.to(dev)
                seg_ids = seg_ids.to(dev)
                view_ids = view_ids.to(dev)
                v, _, _, _, _ = model(X, seg_ids, view_ids)
                feats.append(v.cpu().numpy())
                meta_rows.extend(meta)
        embeddings = np.concatenate(feats, axis=0)
        return embeddings, meta_rows, "model_visit_embedding"

    # Fallback: use mean curve as feature
    embeddings = np.stack([s["curve"] for s in labeled_samples], axis=0)
    meta_rows = labeled_samples
    return embeddings, meta_rows, "mean_curve"


def plot_2d_projection(
    embeddings: np.ndarray,
    meta: List[dict],
    outdir: Path,
    method: str = "tsne",
    perplexity: int = 30,
    max_points: int = 4000,
):
    n = embeddings.shape[0]
    if n < 2:
        return
    idx = np.arange(n)
    if n > max_points:
        idx = np.random.choice(idx, size=max_points, replace=False)
    X = embeddings[idx]
    meta_sel = [meta[i] for i in idx]

    patient_ids = [m["short_id"] for m in meta_sel]
    unique_patients = list(dict.fromkeys(patient_ids))  # preserve order
    patient_times: Dict[str, List[pd.Timestamp]] = {}
    for m in meta_sel:
        patient_times.setdefault(m["short_id"], []).append(m.get("dt"))

    fixed_colors = ["red", "blue", "green", "yellow", "black"]

    def color_for_pid(pid_idx: int):
        return fixed_colors[pid_idx % len(fixed_colors)]

    # Compute projection once on all points
    if method == "tsne":
        if X.shape[0] < 3:
            return
        max_perp = max(2, X.shape[0] - 1)
        perp = max(2, min(perplexity, max_perp))
        proj_all = TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto", random_state=42).fit_transform(X)
        base_fname = "tsne"
        base_title = "t-SNE"
    elif method == "umap":
        try:
            import umap  # type: ignore
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"UMAP not installed, skipping projection: {exc}")
            return
        reducer = umap.UMAP(n_components=2, random_state=42)
        proj_all = reducer.fit_transform(X)
        base_fname = "umap"
        base_title = "UMAP"
    else:
        return

    # Build sizes based on time per patient
    def norm_time(m, pid):
        times = patient_times.get(pid, [])
        times = [t for t in times if pd.notna(t)]
        if times:
            tmin, tmax = min(times), max(times)
            if tmax == tmin:
                return 0.0
            if pd.notna(m.get("dt")):
                nt = (m["dt"] - tmin) / (tmax - tmin)
                return float(np.clip(nt, 0.0, 1.0))
        return 0.5

    base_size = 18.0
    size_scale = 50.0

    # Chunk patients 5 per plot
    chunk_size = 5
    patient_chunks = [unique_patients[i:i + chunk_size] for i in range(0, len(unique_patients), chunk_size)]
    for chunk_idx, chunk in enumerate(patient_chunks, start=1):
        mask = [i for i, pid in enumerate(patient_ids) if pid in chunk]
        if len(mask) < 2:
            continue

        proj = proj_all[mask]
        meta_c = [meta_sel[i] for i in mask]
        pids_c = [patient_ids[i] for i in mask]

        colors = [color_for_pid(unique_patients.index(pid)) for pid in pids_c]
        sizes = [base_size + size_scale * norm_time(m, pid) for m, pid in zip(meta_c, pids_c)]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=sizes, alpha=0.85, edgecolor="none")
        ax.set_title(f"{base_title} (patients chunk {chunk_idx})")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")

        handles = []
        for pid in chunk:
            c = color_for_pid(unique_patients.index(pid))
            handles.append(plt.Line2D([], [], marker="o", linestyle="", color=c, label=pid))
        ax.legend(
            handles=handles,
            title="Patient color\nSize ↑ = later",
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
        )

        fig.tight_layout()
        fig.savefig(outdir / f"{base_fname}_chunk_{chunk_idx}.png", dpi=200)
        plt.close(fig)


def run_probe(embeddings: np.ndarray, labels: np.ndarray, outdir: Path, repr_name: str):
    """Train a simple early/late probe; save metrics."""
    if len(np.unique(labels)) < 2:
        warnings.warn("Only one class present; skipping probe.")
        return
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.3, random_state=42, stratify=labels
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_val)[:, 1]
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    try:
        auc = roc_auc_score(y_val, probs)
    except ValueError:
        auc = float("nan")
    out_text = outdir / "probe_metrics.txt"
    with out_text.open("w", encoding="utf-8") as f:
        f.write(f"representation: {repr_name}\n")
        f.write(f"n_train={len(X_train)}, n_val={len(X_val)}\n")
        f.write(f"accuracy={acc:.4f}\n")
        f.write(f"roc_auc={auc:.4f}\n")
    print(f"Probe metrics saved to {out_text}")


def plot_distance_to_earliest(embeddings: np.ndarray, meta: List[dict], outdir: Path):
    """Plot distribution of distances from each patient's earliest embedding."""
    # Organize by patient
    by_patient: Dict[str, List[Tuple[pd.Timestamp, int, np.ndarray]]] = {}
    for emb, m in zip(embeddings, meta):
        by_patient.setdefault(m["short_id"], []).append((m["dt"], m["label"], emb))

    l2_all = []
    cos_all = []
    for sid, items in by_patient.items():
        items = sorted(items, key=lambda x: x[0])
        if not items:
            continue
        base_emb = items[0][2]
        later = [it for it in items[1:] if it[1] == 1]  # only labeled late samples
        if not later:
            continue
        cur_l2 = [np.linalg.norm(it[2] - base_emb) for it in later]
        cur_cos = cosine_distances(np.stack([base_emb]), np.stack([it[2] for it in later]))[0]
        l2_all.extend(cur_l2)
        cos_all.extend(cur_cos.tolist())

    if not l2_all:
        print("No late samples to compute distance-to-earliest plots.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(l2_all, bins=30, color="#1f77b4", alpha=0.8)
    axes[0].set_title("L2 distance to earliest (late samples)")
    axes[0].set_xlabel("L2 distance")
    axes[0].set_ylabel("Counts")

    axes[1].hist(cos_all, bins=30, color="#ff7f0e", alpha=0.8)
    axes[1].set_title("Cosine distance to earliest (late samples)")
    axes[1].set_xlabel("Cosine distance (1 - cos sim)")

    fig.tight_layout()
    fig.savefig(outdir / "distance_to_earliest.png", dpi=200)
    plt.close(fig)


def plot_histograms(summary: pd.DataFrame, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(summary["mae"].dropna(), bins=30, color="#1f77b4", alpha=0.8)
    axes[0].set_title("MAE: early vs late")
    axes[0].set_xlabel("MAE")
    axes[0].set_ylabel("Patients")

    axes[1].hist(summary["pearson_r"].dropna(), bins=30, color="#ff7f0e", alpha=0.8)
    axes[1].set_title("Pearson r: early vs late")
    axes[1].set_xlabel("Correlation")

    fig.tight_layout()
    fig.savefig(outdir / "histograms.png", dpi=200)
    plt.close(fig)


def plot_examples(records: List[dict], outdir: Path, max_examples: int):
    # pick the top-N patients with largest MAE
    records = sorted(records, key=lambda x: x["mae"], reverse=True)[:max_examples]
    if not records:
        return

    n = len(records)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, rec in zip(axes, records):
        ax.plot(rec["early_mean_curve"], label="Early mean", color="#1f77b4")
        ax.plot(rec["late_mean_curve"], label="Late mean", color="#d62728")
        ax.set_title(f"{rec['short_id']} ({rec['view']})  MAE={rec['mae']:.3f}  r={rec['pearson_r']:.2f}")
        ax.legend()
        ax.set_ylabel("Strain")
    axes[-1].set_xlabel("Resampled time (T bins)")
    fig.tight_layout()
    fig.savefig(outdir / "examples.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="EDA: early vs late strain curve shift per patient")
    ap.add_argument("--parquet", required=True, help="Path to preprocessed strain parquet")
    ap.add_argument("--outdir", default="eda_output", help="Directory to save CSV + plots")
    ap.add_argument("--view", default=None, help="Optional view filter, e.g., 4C or 2C")
    ap.add_argument("--T", type=int, default=64, help="Resample length")
    ap.add_argument("--min_gap_days", type=int, default=0, help="Require at least this gap between early and late")
    ap.add_argument("--max_examples", type=int, default=8, help="How many overlay examples to save")
    ap.add_argument("--model-ckpt", default=None, help="Optional checkpoint to get model visit embeddings")
    ap.add_argument("--device", default="auto", help='Device for model embeddings ("auto", "cpu", or "cuda")')
    ap.add_argument("--batch", type=int, default=64, help="Batch size for embedding extraction")
    ap.add_argument("--no_tsne", action="store_true", help="Skip t-SNE projection")
    ap.add_argument("--umap", action="store_true", help="Also run UMAP projection (requires umap-learn)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    samples = build_cycle_curves(df, T=args.T, view_filter=args.view)
    records = split_early_late(samples, min_gap_days=args.min_gap_days)

    if not records:
        print("No patients with both early and later scans found under current filters.")
        return

    # Flatten summary (drop curve arrays) for CSV
    summary_rows = [
        {
            "short_id": r["short_id"],
            "view": r["view"],
            "n_early_cycles": r["n_early_cycles"],
            "n_late_cycles": r["n_late_cycles"],
            "mae": r["mae"],
            "rmse": r["rmse"],
            "pearson_r": r["pearson_r"],
            "early_dt": r["early_dt"],
            "latest_dt": r["latest_dt"],
        }
        for r in records
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "summary.csv", index=False)

    plot_histograms(summary_df, outdir)
    plot_examples(records, outdir, max_examples=args.max_examples)

    # Separability analysis
    labeled = label_samples_for_classification(samples, min_gap_days=args.min_gap_days)
    embeddings, meta_rows, repr_name = build_embeddings(
        labeled_samples=labeled,
        use_model=bool(args.model_ckpt),
        model_ckpt=args.model_ckpt,
        device=args.device,
        batch_size=args.batch,
        T=args.T,
    )
    if embeddings is not None and len(meta_rows):
        labels = np.array([m["label"] for m in meta_rows])
        if not args.no_tsne:
            plot_2d_projection(embeddings, meta_rows, outdir, method="tsne")
        if args.umap:
            plot_2d_projection(embeddings, meta_rows, outdir, method="umap")
        run_probe(embeddings, labels, outdir, repr_name)
        plot_distance_to_earliest(embeddings, meta_rows, outdir)

    print(f"Saved summary for {len(records)} patients to {outdir}")
    print(f"Examples, histograms, and separability outputs written to {outdir}")


if __name__ == "__main__":
    main()
