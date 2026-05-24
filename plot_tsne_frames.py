import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D

REQUIRED_VIEWS = ["A2C", "A3C", "A4C"]


def pick_patients(df: pd.DataFrame, n_patients: int, seed: int) -> list[str]:
    view_ok = (
        df.groupby("patient_id")["view"]
        .apply(lambda x: set(REQUIRED_VIEWS).issubset(set(x.unique())))
    )
    eligible = view_ok[view_ok].index.to_numpy()
    if len(eligible) < n_patients:
        raise ValueError(
            f"Only {len(eligible)} patients contain all required views; requested {n_patients}."
        )
    rng = np.random.default_rng(seed)
    return rng.choice(eligible, size=n_patients, replace=False).tolist()


def add_visit_size(
    df: pd.DataFrame,
    min_marker_size: float,
    max_marker_size: float,
) -> pd.DataFrame:
    out = df.copy()
    out["study_datetime"] = pd.to_datetime(out["study_datetime"], errors="coerce")
    if out["study_datetime"].isna().any():
        raise ValueError("study_datetime contains invalid values; cannot rank visits.")

    out["visit_rank"] = (
        out.groupby("patient_id")["study_datetime"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    out["visit_count"] = out.groupby("patient_id")["visit_rank"].transform("max")
    denom = (out["visit_count"] - 1).replace(0, 1)
    out["marker_size"] = min_marker_size + (
        (out["visit_rank"] - 1) / denom
    ) * (max_marker_size - min_marker_size)
    return out


def compute_tsne(df: pd.DataFrame, random_state: int, perplexity: float) -> tuple[pd.DataFrame, float]:
    n_rows = len(df)
    emb = np.stack(df["embedding"].apply(np.asarray).to_numpy())
    max_perplexity = max(5.0, min(50.0, (n_rows - 1) / 3.0))
    use_perplexity = min(perplexity, max_perplexity)

    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=use_perplexity,
    )
    coords = tsne.fit_transform(emb)
    out = df.copy()
    out["tsne_x"] = coords[:, 0]
    out["tsne_y"] = coords[:, 1]
    return out, use_perplexity


def save_patient_view_plots(
    df: pd.DataFrame,
    output_path: Path,
    picked_patients: list[str],
    min_marker_size: float,
    max_marker_size: float,
) -> list[Path]:
    cmap = plt.colormaps["tab20"]
    output_paths: list[Path] = []

    for view in REQUIRED_VIEWS:
        view_df = df[df["view"] == view]
        if view_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))
        for i, pid in enumerate(picked_patients):
            g = view_df[view_df["patient_id"] == pid]
            if g.empty:
                continue
            ax.scatter(
                g["tsne_x"],
                g["tsne_y"],
                c=[cmap(i % 20)],
                marker="o",
                s=g["marker_size"].to_numpy(),
                alpha=0.7,
            )

        ax.set_title(
            f"t-SNE ({view}) - {len(picked_patients)} Patients, "
            "marker size: earliest -> latest visit"
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        patient_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(i % 20),
                markersize=7,
                label=str(pid),
            )
            for i, pid in enumerate(picked_patients)
        ]
        size_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                alpha=0.8,
                markersize=np.sqrt(min_marker_size),
                label="Earliest visit",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                alpha=0.8,
                markersize=np.sqrt(max_marker_size),
                label="Latest visit",
            ),
        ]

        patient_legend = ax.legend(
            handles=patient_handles,
            title="Patient ID",
            ncol=2,
            fontsize=8,
            frameon=True,
            loc="upper right",
        )
        ax.add_artist(patient_legend)
        ax.legend(handles=size_handles, title="Visit Order", frameon=True, loc="lower right")
        fig.tight_layout()

        view_out = output_path.with_name(f"{output_path.stem}_{view}{output_path.suffix}")
        view_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(view_out, dpi=220)
        plt.close(fig)
        output_paths.append(view_out)

    return output_paths


def build_plot(
    parquet_path: Path,
    output_path: Path,
    sample_size: int = 500,
    random_state: int = 42,
    perplexity: float = 30.0,
    n_patients: int = 0,
    min_marker_size: float = 12.0,
    max_marker_size: float = 64.0,
) -> None:
    df = pd.read_parquet(parquet_path)

    required_cols = {"embedding", "frame_index"}
    if n_patients > 0:
        required_cols |= {"patient_id", "view", "study_datetime"}
    else:
        required_cols |= {"end_diastole", "end_systole"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    picked_patients: list[str] = []
    if n_patients > 0:
        picked_patients = pick_patients(df, n_patients=n_patients, seed=random_state)
        sampled = df[df["patient_id"].isin(picked_patients)].copy()
        sample_n = len(sampled)
    else:
        sample_n = min(sample_size, len(df))
        sampled = df.sample(n=sample_n, random_state=random_state).copy()
    if n_patients > 0:
        sampled = add_visit_size(
            sampled,
            min_marker_size=min_marker_size,
            max_marker_size=max_marker_size,
        )
        per_view_rows: list[tuple[str, int, float]] = []
        tsne_views = []
        for view in REQUIRED_VIEWS:
            view_df = sampled[sampled["view"] == view].copy()
            if view_df.empty:
                continue
            tsne_view, view_perplexity = compute_tsne(
                view_df,
                random_state=random_state,
                perplexity=perplexity,
            )
            per_view_rows.append((view, len(tsne_view), view_perplexity))
            tsne_views.append(tsne_view)
        if not tsne_views:
            raise ValueError("No rows for REQUIRED_VIEWS after patient filtering.")

        tsne_all = pd.concat(tsne_views, ignore_index=True)
        saved = save_patient_view_plots(
            tsne_all,
            output_path=output_path,
            picked_patients=picked_patients,
            min_marker_size=min_marker_size,
            max_marker_size=max_marker_size,
        )
    else:
        sampled, use_perplexity = compute_tsne(
            sampled,
            random_state=random_state,
            perplexity=perplexity,
        )

        plt.figure(figsize=(10, 8))
        ed_mask = sampled["frame_index"] == sampled["end_diastole"]
        es_mask = sampled["frame_index"] == sampled["end_systole"]
        both_mask = ed_mask & es_mask
        normal_mask = ~(ed_mask | es_mask)
        ed_only_mask = ed_mask & ~both_mask
        es_only_mask = es_mask & ~both_mask

        plt.scatter(
            sampled.loc[normal_mask, "tsne_x"],
            sampled.loc[normal_mask, "tsne_y"],
            c="lightgray",
            marker="o",
            s=35,
            alpha=0.7,
            label="Other frames",
        )
        plt.scatter(
            sampled.loc[ed_only_mask, "tsne_x"],
            sampled.loc[ed_only_mask, "tsne_y"],
            c="#d62728",
            marker="^",
            s=70,
            alpha=0.95,
            label="End diastole",
        )
        plt.scatter(
            sampled.loc[es_only_mask, "tsne_x"],
            sampled.loc[es_only_mask, "tsne_y"],
            c="#1f77b4",
            marker="s",
            s=70,
            alpha=0.95,
            label="End systole",
        )

        if both_mask.any():
            plt.scatter(
                sampled.loc[both_mask, "tsne_x"],
                sampled.loc[both_mask, "tsne_y"],
                c="#2ca02c",
                marker="X",
                s=90,
                alpha=1.0,
                label="Both ED and ES",
            )

        plt.title(f"t-SNE of {sample_n} Random Frame Embeddings")
        plt.legend()
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=220)
        plt.close()

        print(f"Saved plot: {output_path}")
        print(f"Sample size: {sample_n}")
        print(f"Perplexity used: {use_perplexity:.2f}")
        print(f"ED frames in sample: {int(ed_only_mask.sum())}")
        print(f"ES frames in sample: {int(es_only_mask.sum())}")
        print(f"Both ED+ES in sample: {int(both_mask.sum())}")
        return

    print(f"Saved plots: {', '.join(map(str, saved))}")
    print(f"Sample size: {sample_n}")
    print(f"Patients ({len(picked_patients)}): {', '.join(map(str, picked_patients))}")
    for view, n_rows, view_perplexity in per_view_rows:
        print(f"{view}: rows={n_rows}, perplexity={view_perplexity:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample frame embeddings from parquet and visualize with t-SNE."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path(
            r"C:\Users\oronbar.RF\Downloads\Ichilov_frame_embeddings_DinoPipeline_16.parquet"
        ),
        help="Path to parquet input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tsne_frames_500.png"),
        help="Path to output image.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of random rows to sample.",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=0,
        help=(
            "If > 0, randomly choose this many patients (with A2C/A3C/A4C) "
            "and plot all of their frames with one color per patient."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and t-SNE.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Requested t-SNE perplexity (auto-clipped to valid range).",
    )
    parser.add_argument(
        "--min-marker-size",
        type=float,
        default=12.0,
        help="Marker size used for each patient's earliest visit in --n-patients mode.",
    )
    parser.add_argument(
        "--max-marker-size",
        type=float,
        default=64.0,
        help="Marker size used for each patient's latest visit in --n-patients mode.",
    )
    args = parser.parse_args()

    build_plot(
        parquet_path=args.parquet,
        output_path=args.output,
        sample_size=args.sample_size,
        random_state=args.seed,
        perplexity=args.perplexity,
        n_patients=args.n_patients,
        min_marker_size=args.min_marker_size,
        max_marker_size=args.max_marker_size,
    )


if __name__ == "__main__":
    main()
