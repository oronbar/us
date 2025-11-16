"""
Quick reconstruction QC:
  - loads a trained checkpoint of patient_conditioned_strain_model.Model
  - takes a small subset of samples from the parquet
  - plots original Δ (endo-epi) vs reconstructed Δ from the autoencoder

Usage:
  python reconstruct_qc.py \
    --parquet "C:\\path\\strain_dataset.parquet" \
    --ckpt "C:\\path\\models\\pretrain_best.pt" \
    --outdir "C:\\path\\models\\qc_recon" \
    --num 8 \
    --T 64

Outputs: PNG files per sample to outdir.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from patient_conditioned_strain_model import (
    Model,
    StrainParquetDataset,
    batchify,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt)")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--num", type=int, default=8, help="Number of samples to plot")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--T", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = StrainParquetDataset(__import__("pandas").read_parquet(args.parquet), T=args.T)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=batchify)

    model = Model(T=args.T)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.to(device)
    model.eval()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plotted = 0
    for X, y, seg_ids, view_ids, meta in loader:
        X = X.to(device)
        seg_ids = seg_ids.to(device)
        view_ids = view_ids.to(device)

        _, _, _, recon, _ = model(X, seg_ids, view_ids)  # recon: [B,S,T]
        X_np = X.cpu().numpy()
        R_np = recon.cpu().numpy()

        B, S, T = X_np.shape
        for i in range(B):
            if plotted >= args.num:
                return
            m = meta[i]
            fig, axes = plt.subplots(S, 1, figsize=(10, 2.5 * S), sharex=True)
            if S == 1:
                axes = [axes]
            for s in range(S):
                axes[s].plot(X_np[i, s], label="original Δ", color="C0")
                axes[s].plot(R_np[i, s], label="recon Δ", color="C1", linestyle="--")
                axes[s].set_ylabel(f"seg {s}")
                axes[s].legend(loc="upper right")
            axes[-1].set_xlabel("time (resampled)")
            fig.suptitle(f"{m.get('short_id','?')} | {m.get('view','?')} | {m.get('dicom_file','?')} | cycle {m.get('cycle_index','?')}")
            fig.tight_layout()
            outpath = outdir / f"recon_{plotted:03d}.png"
            fig.savefig(outpath, dpi=150)
            plt.close(fig)
            plotted += 1
    print(f"Saved {plotted} figures to {outdir}")


if __name__ == "__main__":
    main()
