"""Plot completed GE EchoPrime SSL fold training losses from runner logs."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"D:\DS\ichilov_ge_ssl_20260930")
OUT = Path(r"D:\us\output\ichilov_video_experiments_20260930\plots")
PATTERN = re.compile(r"SSL all r(\d+) f(\d+): (\d+)/(\d+) loss=([\d.]+)")


def smooth(values: np.ndarray, width: int = 7) -> np.ndarray:
    return np.array([
        values[max(0, i - width // 2):min(len(values), i + width // 2 + 1)].mean()
        for i in range(len(values))
    ])


def main() -> None:
    fig, axes = plt.subplots(3, 5, figsize=(16, 8.5), sharex=True, sharey=True)
    fig.patch.set_facecolor("#f8fafc")
    count = 0
    for repeat in range(3):
        for fold in range(5):
            ax = axes[repeat, fold]
            ax.set_facecolor("white")
            complete = ROOT / "all" / f"r{repeat}_f{fold}" / "run_complete.json"
            log = ROOT / "logs" / f"train_all_r{repeat}_f{fold}.log"
            ax.set_title(f"Repeat {repeat + 1} · fold {fold + 1}", fontsize=10, loc="left")
            ax.grid(alpha=0.18)
            if not complete.exists() or not log.exists():
                ax.text(0.5, 0.5, "Not finished", ha="center", va="center",
                        transform=ax.transAxes, color="#94a3b8")
                continue
            rows = [m for line in log.read_text(encoding="utf-8").splitlines()
                    if (m := PATTERN.search(line))]
            if not rows:
                continue
            steps = np.array([int(m.group(3)) for m in rows])
            losses = np.array([float(m.group(5)) for m in rows])
            ax.plot(steps, losses, color="#93c5fd", lw=0.75, alpha=0.65)
            ax.plot(steps, smooth(losses), color="#1d4ed8", lw=2.1)
            ax.set_xlim(0, max(3300, steps[-1]))
            ax.set_ylim(0, 2.8)
            ax.text(0.97, 0.92, f"{losses[-1]:.2f} last", ha="right", va="top",
                    transform=ax.transAxes, fontsize=9, color="#1e40af")
            count += 1
    for ax in axes[-1]:
        ax.set_xlabel("Training step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Self-supervised loss")
    fig.suptitle(f"EchoPrime GE self-supervised training · {count}/15 shared folds finished",
                 fontsize=16, fontweight="bold", x=0.05, ha="left")
    fig.text(0.05, 0.925,
             "Light: logged loss every 25 steps   ·   Dark: 7-point rolling mean   ·   Same y-axis in every panel",
             fontsize=10, color="#475569")
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.3, h_pad=1.8)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "shared_fold_training_loss.png"
    fig.savefig(target, dpi=180, facecolor=fig.get_facecolor())
    print(f"{target}\nfinished_folds={count}")


if __name__ == "__main__":
    main()
