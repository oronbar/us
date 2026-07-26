# Non-apical and curve-quality ablation

## Bottom line

Removing the apical ring did not improve prediction. Curve QC produced a small controlled CNN ranking gain after apex removal, but did not improve the best overall model or materially raise AUC.

Segments 13–18 (the apical ring) were removed from every variant. Labels, clinical trajectory features, patient splits, and relative thresholds were unchanged.

## Filters

- `noapex`: retain all basal/mid curves (segments 1–12).
- `noapex_fixed_qc`: reject near-flat/extreme curves, peak outside 3–45%, time-to-peak outside 0.20–0.90 cycle, excessive normalized second-difference roughness, or positive-dominant morphology.
- `noapex_shape_qc`: fixed QC plus correlation below 0.75 with the within-analysis/layer median shape.
- Rejected tensor curves were replaced by the same-visit/layer median. Engineered summaries used accepted curves only and included retained-fraction features.

## Curve retention

| policy | raw_curves | accepted_curves | retained_fraction |
| --- | --- | --- | --- |
| noapex | 9984.000 | 9984.000 | 1.000 |
| noapex_fixed_qc | 9984.000 | 9856.000 | 0.987 |
| noapex_shape_qc | 9984.000 | 9096.000 | 0.911 |

## Primary task: first baseline, 15% relative Mid-GLS decline

| model | n | events | roc_auc | roc_auc_ci_low | roc_auc_ci_high | average_precision | average_precision_ci_low | average_precision_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clinical_ridge | 238.000 | 49.000 | 0.631 | 0.544 | 0.728 | 0.289 | 0.214 | 0.431 |
| original18__transmural_sparse | 238.000 | 49.000 | 0.644 | 0.564 | 0.725 | 0.309 | 0.227 | 0.455 |
| noapex__transmural_sparse | 238.000 | 49.000 | 0.618 | 0.527 | 0.705 | 0.290 | 0.215 | 0.405 |
| noapex_fixed_qc__transmural_sparse | 238.000 | 49.000 | 0.602 | 0.515 | 0.690 | 0.254 | 0.190 | 0.358 |
| noapex_shape_qc__transmural_sparse | 238.000 | 49.000 | 0.626 | 0.543 | 0.710 | 0.298 | 0.214 | 0.441 |
| original18__variability_sparse | 238.000 | 49.000 | 0.650 | 0.560 | 0.734 | 0.288 | 0.215 | 0.402 |
| noapex__variability_sparse | 238.000 | 49.000 | 0.610 | 0.510 | 0.709 | 0.253 | 0.191 | 0.369 |
| noapex_fixed_qc__variability_sparse | 238.000 | 49.000 | 0.590 | 0.495 | 0.689 | 0.241 | 0.178 | 0.342 |
| noapex_shape_qc__variability_sparse | 238.000 | 49.000 | 0.613 | 0.515 | 0.703 | 0.254 | 0.190 | 0.356 |
| original18__combined_trees | 238.000 | 49.000 | 0.672 | 0.591 | 0.752 | 0.313 | 0.234 | 0.449 |
| noapex__combined_trees | 238.000 | 49.000 | 0.611 | 0.511 | 0.701 | 0.286 | 0.207 | 0.413 |
| noapex_fixed_qc__combined_trees | 238.000 | 49.000 | 0.605 | 0.510 | 0.698 | 0.279 | 0.208 | 0.413 |
| noapex_shape_qc__combined_trees | 238.000 | 49.000 | 0.610 | 0.517 | 0.704 | 0.273 | 0.199 | 0.397 |
| original18__gpu_curve_net | 238.000 | 49.000 | 0.664 | 0.582 | 0.737 | 0.302 | 0.221 | 0.432 |
| noapex__gpu_curve_net | 238.000 | 49.000 | 0.556 | 0.467 | 0.639 | 0.233 | 0.173 | 0.350 |
| noapex_fixed_qc__gpu_curve_net | 238.000 | 49.000 | 0.567 | 0.477 | 0.662 | 0.248 | 0.181 | 0.365 |
| noapex_shape_qc__gpu_curve_net | 238.000 | 49.000 | 0.588 | 0.496 | 0.676 | 0.256 | 0.187 | 0.364 |

## Direct paired changes

Positive values favor the candidate. These are paired on identical patient-held-out predictions.

| comparison | delta_roc_auc | delta_roc_auc_ci_low | delta_roc_auc_ci_high | delta_average_precision | delta_average_precision_ci_low | delta_average_precision_ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| remove_apex__transmural_sparse | -0.025 | -0.100 | 0.040 | -0.018 | -0.113 | 0.073 |
| fixed_qc_vs_noapex__transmural_sparse | -0.017 | -0.055 | 0.025 | -0.036 | -0.099 | 0.003 |
| shape_qc_vs_noapex__transmural_sparse | 0.008 | -0.041 | 0.059 | 0.007 | -0.064 | 0.077 |
| remove_apex__variability_sparse | -0.040 | -0.090 | 0.011 | -0.034 | -0.094 | 0.021 |
| fixed_qc_vs_noapex__variability_sparse | -0.019 | -0.046 | 0.006 | -0.012 | -0.040 | 0.007 |
| shape_qc_vs_noapex__variability_sparse | 0.003 | -0.030 | 0.035 | 0.001 | -0.028 | 0.026 |
| remove_apex__combined_trees | -0.061 | -0.119 | -0.005 | -0.027 | -0.103 | 0.037 |
| fixed_qc_vs_noapex__combined_trees | -0.005 | -0.035 | 0.021 | -0.007 | -0.041 | 0.027 |
| shape_qc_vs_noapex__combined_trees | -0.001 | -0.032 | 0.033 | -0.013 | -0.052 | 0.024 |
| remove_apex__gpu_curve_net | -0.109 | -0.189 | -0.036 | -0.068 | -0.157 | 0.005 |
| fixed_qc_vs_noapex__gpu_curve_net | 0.012 | -0.034 | 0.061 | 0.014 | -0.020 | 0.058 |
| shape_qc_vs_noapex__gpu_curve_net | 0.032 | -0.020 | 0.087 | 0.022 | -0.026 | 0.080 |

## Notes

- The CNN used the NVIDIA GPU for all three variants.
- Shape-consensus filtering is deliberately a sensitivity analysis: a true regional abnormality can also look like a shape outlier.
- This is exploratory validation, not a clinical alert system.

## Controlled GPU isolation

Because the main engineered ablation also changed the CNN scalar branch, a second experiment held the 27 clinical scalar features, architecture, folds, and seeds identical. Only the curve tensor changed.

| curve input | AUC | AP |
| --- | ---: | ---: |
| All 18 segments, unfiltered | 0.619 | 0.271 |
| Segments 1–12, unfiltered | 0.616 | 0.264 |
| Segments 1–12, fixed QC | 0.624 | 0.280 |
| Segments 1–12, shape QC | 0.624 | 0.284 |
| All 18 segments, fixed QC | 0.615 | 0.272 |
| All 18 segments, shape QC | 0.620 | 0.275 |

- Apex removal alone: delta AUC -0.003 [-0.014, +0.009]; delta AP -0.007 [-0.026, +0.004]. This is effectively neutral.
- After apex removal, fixed QC: delta AUC +0.008 [-0.005, +0.020]; delta AP +0.016 [+0.001, +0.046]. The AP gain is small but bootstrap-stable on the primary task.
- Filtering all 18 segments was neutral. Fixed QC retained 98.5% of curves and shape QC retained 91.1%.
- Across neighboring thresholds, QC changes were generally small and positive, but not consistently significant. The best overall result remains the original 18-segment combined Extra Trees model: AUC 0.672, AP 0.313.

Controlled outputs: `controlled_gpu_metrics.csv`, `controlled_gpu_deltas.csv`, `controlled_gpu_oof_predictions.parquet`, and `controlled_gpu_training_log.csv`.
