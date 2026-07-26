# Next-visit cardiotoxicity alert study

## Bottom line

No Endo-Mid, variability, nonlinear, or GPU model showed a bootstrap-stable average-precision gain over the clinical trajectory across the Mid-GLS next-visit tasks.

Primary task—first visit as baseline and 15% relative Mid-GLS deterioration at the immediately following visit: clinical ridge AUC 0.631, AP 0.289; best model **combined_extra_trees** AUC 0.672, AP 0.313.

This is patient-held-out exploratory validation, not a clinical alert system.

## What was predicted

- One sample is **current visit → immediately next visit**. All 297 adjacent transitions from 103 patients were used when eligible.
- `first`: deterioration is relative to visit 1. Only transitions up to the first threshold crossing are eligible; the visit immediately before that crossing is the positive alert visit.
- `roll2` / `roll3`: the next visit is compared with the mean of the last 2 or 3 visits available at the current visit.
- Every endpoint is a **relative decline**. No absolute-decline labels were used.
- Mid-GLS thresholds: 10%, 12%, 15%, and 20%. Endo-GLS and relative EF decline are sensitivity analyses.

## Features

| family | features |
| --- | --- |
| Clinical trajectory (27) | Current/baseline Mid and Endo GLS, EF, relative change from first/rolling baselines, last-visit change, slopes, intervals, history length, Endo–Mid GLS gap. |
| Endo–Mid engineered (170) | Segment-paired amplitude and time-to-peak gaps, curve/shape distances, Endo–Mid coherence, phase lag, and early change in these quantities. |
| Inter-segment variability (72) | Robust peak-amplitude dispersion, circular time-to-peak dispersion, curve/shape incoherence, regional gradients, graph roughness, and change from the previous visit. |
| Raw GPU curves | 18 segments × 96 phase points with current Endo, current Mid, Endo−Mid, and the three corresponding changes from the previous visit. |

## Models

| model | implementation |
| --- | --- |
| `clinical_ridge` | L2 logistic regression on clinical trajectory features. |
| `clinical_plus_transmural_sparse` | L1 logistic regression adding engineered Endo–Mid features. |
| `clinical_plus_variability_sparse` | L1 logistic regression adding segment-variability features. |
| `combined_extra_trees` | Constrained nonlinear Extra Trees using all engineered features. |
| `gpu_segment_curve_net` | 16,746-parameter shared 1D segment CNN, mean/std/max segment pooling, plus clinical/variability scalars; trained on **NVIDIA GeForce RTX 4060 Ti**. |

All visits from a patient stay in the same fold. Results average 3 repeated five-fold patient splits. Confidence intervals use patient-cluster bootstrap.

## Relative-label audit

| task | baseline | relative_threshold | eligible_transitions | events | event_rate |
| --- | --- | --- | --- | --- | --- |
| mid_first_rel10 | first | 0.100 | 211.000 | 65.000 | 0.308 |
| mid_first_rel12 | first | 0.120 | 225.000 | 58.000 | 0.258 |
| mid_first_rel15 | first | 0.150 | 238.000 | 49.000 | 0.206 |
| mid_first_rel20 | first | 0.200 | 262.000 | 29.000 | 0.111 |
| mid_roll2_rel10 | roll2 | 0.100 | 194.000 | 51.000 | 0.263 |
| mid_roll2_rel12 | roll2 | 0.120 | 194.000 | 42.000 | 0.216 |
| mid_roll2_rel15 | roll2 | 0.150 | 194.000 | 29.000 | 0.149 |
| mid_roll2_rel20 | roll2 | 0.200 | 194.000 | 13.000 | 0.067 |
| mid_roll3_rel10 | roll3 | 0.100 | 93.000 | 19.000 | 0.204 |
| mid_roll3_rel12 | roll3 | 0.120 | 93.000 | 14.000 | 0.151 |
| mid_roll3_rel15 | roll3 | 0.150 | 93.000 | 9.000 | 0.097 |
| mid_roll3_rel20 | roll3 | 0.200 | 93.000 | 4.000 | 0.043 |

Rolling-3 strict thresholds have very few events and are sensitivity checks only.

## Mid-GLS result summary

| task | events/n | clinical_AP | best_model | best_AP | best_AUC | delta_AP_CI |
| --- | --- | --- | --- | --- | --- | --- |
| mid_first_rel10 | 65/211 | 0.482 | clinical_plus_variability_sparse | 0.484 | 0.682 | +0.002 [-0.050, +0.055] |
| mid_first_rel12 | 58/225 | 0.346 | clinical_plus_transmural_sparse | 0.404 | 0.661 | +0.058 [-0.037, +0.143] |
| mid_first_rel15 | 49/238 | 0.289 | combined_extra_trees | 0.313 | 0.672 | +0.024 [-0.086, +0.114] |
| mid_first_rel20 | 29/262 | 0.288 | clinical_ridge | 0.288 | 0.712 | reference |
| mid_roll2_rel10 | 51/194 | 0.404 | clinical_ridge | 0.404 | 0.668 | reference |
| mid_roll2_rel12 | 42/194 | 0.270 | clinical_ridge | 0.270 | 0.601 | reference |
| mid_roll2_rel15 | 29/194 | 0.212 | clinical_ridge | 0.212 | 0.580 | reference |
| mid_roll2_rel20 | 13/194 | 0.127 | clinical_ridge | 0.127 | 0.583 | reference |
| mid_roll3_rel10 | 19/93 | 0.274 | gpu_segment_curve_net | 0.297 | 0.474 | +0.024 [-0.172, +0.191] |
| mid_roll3_rel12 | 14/93 | 0.321 | gpu_segment_curve_net | 0.321 | 0.580 | +0.000 [-0.309, +0.260] |
| mid_roll3_rel15 | 9/93 | 0.211 | gpu_segment_curve_net | 0.290 | 0.620 | +0.080 [-0.219, +0.418] |

`delta_AP_CI` compares the best model with clinical ridge. An interval crossing zero is not a stable improvement.

## Primary 15% first-baseline task

| model | n | events | roc_auc | roc_auc_ci_low | roc_auc_ci_high | average_precision | average_precision_ci_low | average_precision_ci_high | sensitivity_top20pct | precision_top20pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combined_extra_trees | 238.000 | 49.000 | 0.672 | 0.586 | 0.752 | 0.313 | 0.228 | 0.453 | 0.306 | 0.312 |
| clinical_plus_transmural_sparse | 238.000 | 49.000 | 0.644 | 0.565 | 0.724 | 0.309 | 0.233 | 0.443 | 0.286 | 0.292 |
| gpu_segment_curve_net | 238.000 | 49.000 | 0.664 | 0.579 | 0.737 | 0.302 | 0.218 | 0.440 | 0.286 | 0.292 |
| clinical_ridge | 238.000 | 49.000 | 0.631 | 0.539 | 0.721 | 0.289 | 0.209 | 0.434 | 0.327 | 0.333 |
| clinical_plus_variability_sparse | 238.000 | 49.000 | 0.650 | 0.568 | 0.731 | 0.288 | 0.219 | 0.406 | 0.286 | 0.292 |

## Interpretation

- Fixed-first labels answer “will the next visit be the first threshold crossing?” and best match a surveillance alert.
- Rolling baselines adapt to drift but can label repeated episodes and have fewer eligible transitions, especially for `roll3`.
- A GPU model is worth retaining only if it improves patient-held-out ranking across neighboring thresholds or baseline definitions, not one isolated task.
- The [2022 ESC definitions](https://academic.oup.com/ehjcimaging/article/23/10/e333/6675075) use >15% relative GLS decline as one component of CTRCD. This dataset lacks biomarkers, treatment exposure, symptoms, and adjudication, so these are imaging-deterioration alerts only.

## Files

- `next_visit_transitions.parquet`: one row per current→next visit sample and every relative label.
- `label_audit.csv`: sample/event counts for all baseline and threshold definitions.
- `oof_predictions.parquet`, `model_metrics.csv`, `model_deltas_vs_clinical.csv`.
- `gpu_training_log.csv`, `patient_fold_assignments.csv`, `feature_manifest.csv`.
- `figures/`: event-rate, AP-delta, and alert-budget figures.
