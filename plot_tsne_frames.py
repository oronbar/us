import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def build_plot(
    parquet_path: Path,
    output_path: Path,
    sample_size: int = 500,
    random_state: int = 42,
    perplexity: float = 30.0,
) -> None:
    df = pd.read_parquet(parquet_path)

    required_cols = {"embedding", "frame_index", "end_diastole", "end_systole"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    sample_n = min(sample_size, len(df))
    sampled = df.sample(n=sample_n, random_state=random_state).copy()

    embeddings = np.stack(sampled["embedding"].apply(np.asarray).to_numpy())

    max_perplexity = max(5.0, min(50.0, (sample_n - 1) / 3.0))
    use_perplexity = min(perplexity, max_perplexity)

    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=use_perplexity,
    )
    coords = tsne.fit_transform(embeddings)

    sampled["tsne_x"] = coords[:, 0]
    sampled["tsne_y"] = coords[:, 1]

    ed_mask = sampled["frame_index"] == sampled["end_diastole"]
    es_mask = sampled["frame_index"] == sampled["end_systole"]
    both_mask = ed_mask & es_mask
    normal_mask = ~(ed_mask | es_mask)
    ed_only_mask = ed_mask & ~both_mask
    es_only_mask = es_mask & ~both_mask

    plt.figure(figsize=(10, 8))
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
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
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
    args = parser.parse_args()

    build_plot(
        parquet_path=args.parquet,
        output_path=args.output,
        sample_size=args.sample_size,
        random_state=args.seed,
        perplexity=args.perplexity,
    )


if __name__ == "__main__":
    main()
