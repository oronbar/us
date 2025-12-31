"""
Plot a t-SNE of embeddings colored by GLS with marker size by time from first visit.

Defaults:
  Input:  ~/OneDrive - Technion/DS/Ichilov_GLS_embeddings.parquet
  Output: ~/OneDrive - Technion/models/Ichilov_GLS_models/tsne_plots/<input_stem>_tsne_gls_<timestamp>.png

Usage (PowerShell):
  .venv\\Scripts\\python plot_gls_tsne.py
"""
from __future__ import annotations

import argparse
import ast
import inspect
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("plot_gls_tsne")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _find_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = False) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    for cand in candidates:
        key = cand.lower()
        for col in df.columns:
            if key in str(col).lower():
                return col
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def _parse_embedding(val: object) -> Optional[np.ndarray]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, np.ndarray):
        return val.astype(np.float32)
    if isinstance(val, list):
        return np.asarray(val, dtype=np.float32)
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return np.asarray(parsed, dtype=np.float32)
        except Exception:
            return None
    return None


def _load_embeddings(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)


def _add_gls_from_report(df: pd.DataFrame, report_xlsx: Path) -> pd.DataFrame:
    rep = pd.read_excel(report_xlsx, engine="openpyxl")
    rep.columns = [str(c).strip() for c in rep.columns]

    mapping = {}
    for view in ("A2C", "A3C", "A4C"):
        gls_col = _find_column(rep, [f"{view}_GLS"], required=False)
        src_col = _find_column(rep, [f"{view}_GLS_SOURCE_DICOM"], required=False)
        if not gls_col or not src_col:
            continue
        for _, row in rep.iterrows():
            src = row.get(src_col)
            gls = row.get(gls_col)
            if pd.isna(src) or pd.isna(gls):
                continue
            try:
                gls_val = float(gls)
            except Exception:
                continue
            mapping[Path(str(src)).name.lower()] = gls_val

    if "source_dicom" not in df.columns:
        raise ValueError("Embeddings dataframe missing 'source_dicom' column needed for GLS join.")

    gls_vals = []
    for _, row in df.iterrows():
        src = row.get("source_dicom")
        base = Path(str(src)).name.lower()
        gls_vals.append(mapping.get(base))

    out = df.copy()
    out["gls"] = gls_vals
    return out


def main() -> None:
    user_home = Path.home()
    if user_home.name == "oronbar.RF":
        user_home = Path("F:\\")
    default_embeddings = user_home / "OneDrive - Technion" / "DS" / "Ichilov_GLS_embeddings_full.parquet"
    default_out_dir = user_home / "OneDrive - Technion" / "models" / "Ichilov_GLS_models" / "tsne_plots"

    parser = argparse.ArgumentParser(description="t-SNE visualization of embeddings colored by GLS.")
    parser.add_argument(
        "--input-embeddings",
        type=Path,
        default=default_embeddings,
        help=f"Embedding dataframe (parquet/csv) (default: {default_embeddings})",
    )
    parser.add_argument(
        "--report-xlsx",
        type=Path,
        default=user_home / "OneDrive - Technion" / "DS" / "Report_Ichilov_GLS_oron.xlsx",
        help="Optional report Excel to supply GLS targets if missing",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="",
        help="Optional target column name (defaults to 'gls' if present)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Optional embedding dimension to select (e.g., 768 for MAE max-pooled, 1536 for STF fusion).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1500,
        help="Randomly subsample to this many points before t-SNE (0 to disable).",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output PNG path (default: {default_out_dir}/<input_stem>_tsne_gls_<timestamp>.png).",
    )
    args = parser.parse_args()

    df = _load_embeddings(args.input_embeddings)
    df.columns = [str(c).strip() for c in df.columns]

    if args.target_col:
        if args.target_col not in df.columns:
            raise ValueError(f"Target column '{args.target_col}' not found in embeddings df.")
        df["gls"] = df[args.target_col]
    elif "gls" not in df.columns and "GLS" in df.columns:
        df["gls"] = df["GLS"]

    if "gls" not in df.columns:
        if args.report_xlsx:
            df = _add_gls_from_report(df, args.report_xlsx)
        else:
            raise ValueError("No GLS target found. Provide --report-xlsx or --target-col.")

    emb_col = _find_column(df, ["embedding"], required=True)
    df["embedding_vec"] = df[emb_col].apply(_parse_embedding)
    df = df[df["embedding_vec"].notna()].copy()
    df["gls"] = pd.to_numeric(df["gls"], errors="coerce")
    df = df[df["gls"].notna()].copy()

    if df.empty:
        raise ValueError("No valid rows with embeddings and GLS targets.")

    if args.embedding_dim is not None:
        if "embedding_dim" in df.columns:
            df = df[df["embedding_dim"] == args.embedding_dim].copy()
            logger.info("Filtered to embedding_dim=%s; remaining rows=%d", args.embedding_dim, len(df))
        else:
            logger.warning("--embedding-dim provided but 'embedding_dim' column missing; proceeding without filtering.")

    emb_lengths = df["embedding_vec"].apply(lambda x: x.shape[0]).value_counts()
    if len(emb_lengths) > 1:
        target_dim = int(emb_lengths.idxmax())
        df = df[df["embedding_vec"].apply(lambda x: x.shape[0] == target_dim)].copy()
        logger.warning(
            "Mixed embedding dimensions detected; auto-selected dim=%d with %d samples.",
            target_dim,
            len(df),
        )

    if len(df) == 0:
        raise ValueError("No samples left after filtering inconsistent embedding dimensions.")

    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.random_state).copy()
        logger.info("Subsampled to %d samples for t-SNE.", len(df))

    X = np.stack(df["embedding_vec"].to_list()).astype(np.float32)
    y = df["gls"].astype(float).values

    # Marker size: time (days) from first patient visit.
    patient_col = _find_column(df, ["patient_id", "patient_num", "patient"], required=False)
    date_col = _find_column(
        df,
        ["visit_date", "study_date", "exam_date", "acquisition_date", "acquisition_time", "scan_date", "date"],
        required=False,
    )
    sizes = np.full(len(df), 60.0, dtype=float)
    patients = None
    dates = None
    if patient_col is None or date_col is None:
        logger.warning("Missing patient/date columns; using uniform marker size.")
    else:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        patients = df[patient_col].astype(str)
        min_dates = dates.groupby(patients).transform("min")
        delta_days = (dates - min_dates).dt.total_seconds() / 86400.0
        delta_days = delta_days.fillna(0).clip(lower=0)
        dmin, dmax = delta_days.min(), delta_days.max()
        if pd.isna(dmin) or pd.isna(dmax):
            logger.warning("Could not parse dates; using uniform marker size.")
        elif dmax - dmin < 1e-6:
            sizes = np.full(len(df), 60.0, dtype=float)
        else:
            sizes = 40.0 + 200.0 * (delta_days - dmin) / (dmax - dmin)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tsne_sig = inspect.signature(TSNE)
    tsne_params = tsne_sig.parameters.keys()
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": args.perplexity,
        "random_state": args.random_state,
    }
    if "n_iter" in tsne_params:
        tsne_kwargs["n_iter"] = args.n_iter
    else:
        logger.warning("Installed scikit-learn TSNE does not support 'n_iter'; using library default.")
    if "init" in tsne_params:
        tsne_kwargs["init"] = "pca"
    if "learning_rate" in tsne_params:
        tsne_kwargs["learning_rate"] = 200.0

    tsne = TSNE(**tsne_kwargs)
    coords = tsne.fit_transform(X_scaled)

    # Output handling.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = args.output
    if out_base is None:
        out_path = default_out_dir / f"{args.input_embeddings.stem}_tsne_gls_{run_id}.png"
    else:
        out_base = Path(out_base)
        if out_base.suffix:
            out_path = out_base
        else:
            out_path = out_base / f"{args.input_embeddings.stem}_tsne_gls_{run_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=y,
        s=sizes,
        cmap="coolwarm",
        alpha=0.8,
        edgecolor="k",
        linewidth=0.2,
    )

    vectors = []
    vec_arr = None
    if patients is not None and dates is not None:
        prog_df = pd.DataFrame(
            {
                "patient": patients,
                "date": dates,
                "x": coords[:, 0],
                "y": coords[:, 1],
            }
        )
        prog_df = prog_df.dropna(subset=["date"])
        for patient_id, group in prog_df.sort_values("date").groupby("patient"):
            if len(group) < 2:
                continue
            diffs = group[["x", "y"]].diff().iloc[1:]
            total = diffs.sum()
            start = group.iloc[0][["x", "y"]]
            vectors.append((start["x"], start["y"], total["x"], total["y"]))
        if vectors:
            vec_arr = np.array(vectors, dtype=float)
            ax.quiver(
                vec_arr[:, 0],
                vec_arr[:, 1],
                vec_arr[:, 2],
                vec_arr[:, 3],
                angles="xy",
                scale_units="xy",
                scale=2.5,
                color="black",
                alpha=0.6,
                width=0.0015,
                headwidth=3.0,
                headlength=3.5,
            )
        else:
            logger.warning("Not enough dated visits per patient to plot progression vectors.")
    else:
        logger.warning("Patient/date columns missing; skipping progression vectors.")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("GLS")
    ax.set_title("t-SNE of embeddings with GLS coloring")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved t-SNE plot: %s", out_path)

    if vectors and vec_arr is not None:
        mean_dx = float(np.mean(vec_arr[:, 2]))
        mean_dy = float(np.mean(vec_arr[:, 3]))
        fig_sum, ax_sum = plt.subplots(figsize=(8, 6))
        ax_sum.scatter(
            coords[:, 0],
            coords[:, 1],
            c=y,
            s=sizes,
            cmap="coolwarm",
            alpha=0.35,
            edgecolor="none",
        )
        ax_sum.quiver(
            0.0,
            0.0,
            mean_dx,
            mean_dy,
            angles="xy",
            scale_units="xy",
            scale=2.5,
            color="black",
            alpha=0.9,
            width=0.003,
            headwidth=4.0,
            headlength=4.5,
        )
        cbar_sum = fig_sum.colorbar(ax_sum.collections[0], ax=ax_sum)
        cbar_sum.set_label("GLS")
        ax_sum.set_title("t-SNE with mean patient progression vector")
        ax_sum.set_xlabel("t-SNE 1")
        ax_sum.set_ylabel("t-SNE 2")
        fig_sum.tight_layout()

        sum_out_path = out_path.with_name(f"{out_path.stem}_sumvec{out_path.suffix}")
        fig_sum.savefig(sum_out_path, dpi=200)
        plt.close(fig_sum)
        logger.info("Saved summed progression plot: %s", sum_out_path)

        fig_hist, axes = plt.subplots(1, 2, figsize=(10, 4))
        dx_vals = vec_arr[:, 2]
        dy_vals = vec_arr[:, 3]
        axes[0].hist(dx_vals, bins=30, color="steelblue", alpha=0.8)
        axes[0].axvline(mean_dx, color="red", linestyle="--", linewidth=1.5, label=f"mean={mean_dx:.3f}")
        axes[0].set_title("Vector dx histogram")
        axes[0].set_xlabel("dx")
        axes[0].set_ylabel("count")
        axes[0].legend()

        axes[1].hist(dy_vals, bins=30, color="steelblue", alpha=0.8)
        axes[1].axvline(mean_dy, color="red", linestyle="--", linewidth=1.5, label=f"mean={mean_dy:.3f}")
        axes[1].set_title("Vector dy histogram")
        axes[1].set_xlabel("dy")
        axes[1].set_ylabel("count")
        axes[1].legend()

        fig_hist.tight_layout()
        hist_out_path = out_path.with_name(f"{out_path.stem}_vec_hist{out_path.suffix}")
        fig_hist.savefig(hist_out_path, dpi=200)
        plt.close(fig_hist)
        logger.info("Saved vector component histograms: %s", hist_out_path)


if __name__ == "__main__":
    main()
