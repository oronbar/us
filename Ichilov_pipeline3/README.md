# Ichilov Pipeline3 Diagnostics And Ablations

## Quick Start

Baseline run with diagnostics enabled:

```powershell
python Ichilov_pipeline3\train_pipeline3.py `
  --config Ichilov_pipeline3\config_pipeline3.yaml `
  --run-name pipeline3_baseline `
  --temporal-mode attn_pool `
  --view-fusion-mode attn `
  --longitudinal-ablation-mode inherit `
  --sampling-mode uniform `
  --delta-gls-convention latest_minus_earliest `
  --debug-delta
```

Example ablation run:

```powershell
python Ichilov_pipeline3\train_pipeline3.py `
  --config Ichilov_pipeline3\config_pipeline3.yaml `
  --run-name pipeline3_ablation_temporal_mean `
  --temporal-mode mean_pool `
  --view-fusion-mode attn `
  --longitudinal-ablation-mode inherit `
  --sampling-mode phase_aligned `
  --phase-aligned-strategy cycle `
  --phase-aligned-include-midpoints
```

Grid runner:

```powershell
python Ichilov_pipeline3\run_ablations.py `
  --config Ichilov_pipeline3\config_pipeline3.yaml `
  --output-root "C:\Users\Oron\OneDrive - Technion\Experiments\DinoPipeline_21\ablation_suite_pipeline3" `
  --epochs 8
```

Repeated patient-level CV (K splits):

```powershell
python Ichilov_pipeline3\run_cv.py `
  --config Ichilov_pipeline3\config_pipeline3.yaml `
  --output-root "C:\Users\Oron\OneDrive - Technion\Experiments\DinoPipeline_21\cv_suite_pipeline3" `
  --k-splits 3 `
  --epochs 6 `
  --variants baseline,no_time_encoding,long_last_visit_linear,temporal_mean_pool,view_mean `
  --sampling-modes uniform,phase_aligned `
  --delta-gls-convention latest_minus_earliest
```

## Delta Convention + Sign Checks

- Single source-of-truth convention:
  - visits are sorted earliest -> latest
  - `delta_gls_target = GLS_latest - GLS_earliest` when `delta_gls_convention=latest_minus_earliest`
- Per-epoch sign diagnostics (`--debug-delta`) logs:
  - sample patients: `(t_first,t_last)`, `(GLS_first,GLS_last)`, `target_delta`, `pred_delta`
  - `corr(pred,target)` and `corr(-pred,target)`
- Split metrics now include:
  - `corr_signflip_advantage = corr(-pred,target) - corr(pred,target)`
  - Warning is emitted if this is greater than `0.2`.

## Diagnostic Interpretation

- `diagnostics/final_frame_attention_hist.png`:
  - Healthy behavior: non-degenerate spread, not all mass on one frame.
  - Bug signal: `masked_attention_max` noticeably above ~`1e-5`.

- `diagnostics/final_view_attention_hist.png` and `dominance_counts`:
  - Healthy behavior: all views used when available.
  - Bug signal: one view dominates almost always, even when all 3 are valid.

- `diagnostics/final_within_cos_vs_delta_time.png`:
  - Healthy behavior: adjacent-visit cosine varies with trajectory/time.
  - Collapse signal: cosine is near 1.0 for almost all pairs.

- `summary.json -> probes.patient_id_probe`:
  - High train accuracy suggests strong identity encoding in visit embeddings.
  - Very high values are leakage risk indicators for non-patient-grouped splits.

- `summary.json -> probes.delta_gls_probe`:
  - Compare `baseline_*` vs `model_*`.
  - If baseline ~= model, longitudinal stack may not add signal.

- `summary.json -> split_metrics.val.corr_signflip_advantage`:
  - Near `0`: no obvious sign mismatch signal.
  - Large positive value (`>0.2`): likely sign mismatch or anti-correlated predictions.
