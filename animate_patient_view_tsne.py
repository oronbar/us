import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


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


def pick_one_clip_per_view(df: pd.DataFrame, patient_ids: list[str]) -> pd.DataFrame:
    sub = df[df["patient_id"].isin(patient_ids)].copy()
    rows = []
    for pid in patient_ids:
        for view in REQUIRED_VIEWS:
            cur = sub[(sub["patient_id"] == pid) & (sub["view"] == view)]
            if cur.empty:
                continue
            clip = cur.groupby("source_dicom").size().idxmax()
            clip_df = cur[cur["source_dicom"] == clip].sort_values("frame_index").copy()
            clip_df["traj_key"] = f"{pid}::{view}"
            rows.append(clip_df)
    if not rows:
        raise ValueError("No trajectories selected.")
    return pd.concat(rows, ignore_index=True)


def compute_tsne(df: pd.DataFrame, seed: int, perplexity: float) -> pd.DataFrame:
    emb = np.stack(df["embedding"].apply(np.asarray).to_numpy())
    max_perplexity = max(5.0, min(50.0, (len(df) - 1) / 3.0))
    use_perplexity = min(perplexity, max_perplexity)
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=use_perplexity,
    )
    xy = tsne.fit_transform(emb)
    out = df.copy()
    out["x"] = xy[:, 0]
    out["y"] = xy[:, 1]
    return out


def animate(
    tsne_df: pd.DataFrame,
    out_mp4: Path,
    fps: int,
    n_frames: int,
    title: str,
) -> None:
    patient_ids = sorted(tsne_df["patient_id"].unique().tolist())
    cmap = plt.colormaps["tab20"]
    patient_colors = {pid: cmap(i % 20) for i, pid in enumerate(patient_ids)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    axes_by_view = {v: ax for v, ax in zip(REQUIRED_VIEWS, axes)}

    x_min, x_max = tsne_df["x"].min(), tsne_df["x"].max()
    y_min, y_max = tsne_df["y"].min(), tsne_df["y"].max()
    dx = (x_max - x_min) * 0.07
    dy = (y_max - y_min) * 0.07

    artists = {}
    trajs = {}
    for (pid, view), g in tsne_df.groupby(["patient_id", "view"]):
        g = g.sort_values("frame_index")
        trajs[(pid, view)] = (g["x"].to_numpy(), g["y"].to_numpy())

    for view, ax in axes_by_view.items():
        ax.set_title(view)
        ax.set_xlim(x_min - dx, x_max + dx)
        ax.set_ylim(y_min - dy, y_max + dy)
        ax.set_xlabel("t-SNE 1")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("t-SNE 2")

    for pid in patient_ids:
        for view in REQUIRED_VIEWS:
            ax = axes_by_view[view]
            line = ax.plot([], [], color=patient_colors[pid], alpha=0.85, lw=1.6)[0]
            dot = ax.scatter([], [], s=28, color=patient_colors[pid], alpha=1.0)
            artists[(pid, view)] = (line, dot)

    legend_lines = [
        plt.Line2D([0], [0], color=patient_colors[pid], lw=2, label=pid)
        for pid in patient_ids
    ]
    fig.legend(
        handles=legend_lines,
        loc="upper center",
        ncol=10,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, 1.05),
        title="Patient ID colors",
    )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_mp4.as_posix(), fps=fps, codec="libx264") as writer:
        for fi in range(n_frames):
            progress = fi / max(1, n_frames - 1)
            for key, (xs, ys) in trajs.items():
                n = len(xs)
                stop = int(progress * (n - 1)) + 1
                stop = max(1, min(stop, n))
                line, dot = artists[key]
                line.set_data(xs[:stop], ys[:stop])
                dot.set_offsets(np.array([[xs[stop - 1], ys[stop - 1]]]))

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)

    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Animate patient/view trajectories in t-SNE.")
    p.add_argument(
        "--parquet",
        type=Path,
        default=Path(
            r"C:\Users\oronbar.RF\Downloads\Ichilov_frame_embeddings_DinoPipeline_16.parquet"
        ),
    )
    p.add_argument("--output", type=Path, default=Path(r"c:\work\us\tsne_patient_tracks_20.mp4"))
    p.add_argument("--n-patients", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--frames", type=int, default=180)
    args = p.parse_args()

    use_cols = ["patient_id", "view", "source_dicom", "frame_index", "embedding"]
    df = pd.read_parquet(args.parquet, columns=use_cols)

    patients = pick_patients(df, n_patients=args.n_patients, seed=args.seed)
    selected = pick_one_clip_per_view(df, patient_ids=patients)
    tsne_df = compute_tsne(selected, seed=args.seed, perplexity=args.perplexity)

    title = (
        f"t-SNE trajectories for {args.n_patients} patients "
        f"(one clip per view, A2C/A3C/A4C)"
    )
    animate(tsne_df, args.output, fps=args.fps, n_frames=args.frames, title=title)

    print(f"Saved MP4: {args.output}")
    print(f"Patients ({len(patients)}): {', '.join(map(str, patients))}")
    print(f"Rows used: {len(tsne_df)}")


if __name__ == "__main__":
    main()
