# Early detection of final-visit cardiotoxicity deterioration

## Bottom line

Neither proposed feature family showed a bootstrap-stable incremental gain over the early clinical trajectory for the primary threshold; any apparent gains are exploratory.

For the primary final Mid-GLS relative decline threshold of 15%, the best repeated nested-CV model was **transmural_only_sparse** (ROC AUC 0.615, average precision 0.457) versus the early clinical ridge (ROC AUC 0.615, average precision 0.367).

Endo-Mid curve-difference features added to the clinical trajectory: delta AP +0.004 (95% bootstrap CI -0.135 to +0.130); delta ROC AUC -0.061 (-0.202 to +0.076).

Inter-segment variability features added to the clinical trajectory: delta AP +0.002 (95% bootstrap CI -0.125 to +0.126); delta ROC AUC -0.028 (-0.166 to +0.112).

This is a small, retrospective, internally validated exploratory analysis—not a deployable clinical model.

## Study design

- Source: 416 anonymized AutoStrainCap exports, reconstructed as 400 true visits in 103 patients.
- Landmark cohort: 101 patients with at least three true visits.
- Predictor window: visit 1 plus visit 2 only. The outcome is the last available true visit.
- Median visit-2 landmark: 93.8 days after baseline; median remaining prediction horizon: 192.6 days.
- “Incident” analyses exclude patients who already crossed the deterioration threshold by visit 2.
- Mid is the software's mid-myocardial layer; Mid-GLS is primary, with Endo-GLS and EF sensitivity endpoints.
- Technical reanalyses were averaged within Study UID before longitudinal modeling.
- Near-zero segment amplitudes were excluded from Endo/Mid ratios; all linear models used fold-local robust scaling and bounded transformed values so one tracking failure could not dominate a held-out prediction.

## Threshold audit

| outcome | eligible_patients | events | event_rate |
| --- | --- | --- | --- |
| mid_gls_relative_drop_10 | 101.000 | 46.000 | 0.455 |
| mid_gls_relative_drop_12 | 101.000 | 40.000 | 0.396 |
| mid_gls_relative_drop_15 | 101.000 | 30.000 | 0.297 |
| mid_gls_relative_drop_20 | 101.000 | 11.000 | 0.109 |
| ef_absolute_drop_5 | 95.000 | 26.000 | 0.274 |
| incident_mid_gls_relative_drop_15 | 76.000 | 17.000 | 0.224 |
| incident_ef_absolute_drop_5 | 68.000 | 18.000 | 0.265 |

The 20% Mid-GLS and 10-point EF thresholds have few events and should be read as stress tests, not stable model-development endpoints.

The 15% relative GLS threshold was chosen because it is used in the [2022 ESC cardio-oncology definitions](https://academic.oup.com/ehjcimaging/article/23/10/e333/6675075), but those definitions combine GLS with LVEF and/or biomarkers. This dataset lacks biomarkers, symptoms, treatment exposure, and adjudication, so every label here means **imaging deterioration**, not diagnosed cancer therapy-related cardiac dysfunction. The EF 5/7/10-point labels are exploratory; only 0 landmark patients had both an EF decline of at least 10 points and final EF below 50%.

## Signals worth independent follow-up

- For incident 15% Mid-GLS decline after visit 2, clinical plus Endo-Mid features reached ROC AUC 0.684 and AP 0.411, versus 0.539 and 0.282 for clinical ridge. Its paired ranking improvements did not exclude zero, so this is the most interesting Endo-Mid lead—not confirmation.
- For the 2.9-point absolute Mid-GLS drop, combined Extra Trees reached ROC AUC 0.777 and AP 0.590, versus 0.709 and 0.449. Ranking intervals also crossed zero; the Brier-score improvement was borderline/stable and needs replication.
- No candidate feature survived FDR at q<0.10 (0 discoveries). Every feature-extended model had worse point-estimate RMSE than clinical ridge for continuous future GLS and EF decline.

## Approaches tested

1. A low-dimensional clinical ridge using baseline and visit-2 Mid/Endo GLS, EF, early changes, and time to visit 2.
2. Sparse L1-logistic models using direct paired Endo-Mid segment-curve features: transmural amplitude and timing gaps, curve/shape distance, layer coherence, phase lag, and fixed DCT waveform coefficients.
3. Sparse L1-logistic models using inter-segment heterogeneity: robust peak dispersion, normalized time-to-peak dispersion, curve/shape incoherence, spatial roughness, regional gradients, and their visit-1 to visit-2 changes.
4. Combined sparse-logistic and constrained Extra Trees models to test multivariable and nonlinear signals.
5. Threshold-independent ridge/Extra Trees regression of continuous final and post-visit-2 GLS/EF decline.
6. Clinical-adjusted partial Spearman screens and bootstrap sparse-logistic coefficient stability for interpretation.

All classification scores are averages of repeated patient-held-out outer folds. Logistic regularization was chosen inside each training fold. Model comparison uses paired patient bootstrap intervals.

## Classification results

| outcome | model | n | events | roc_auc | average_precision | sensitivity_top20pct | precision_top20pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mid_gls_relative_drop_10 | clinical_ridge | 101.000 | 46.000 | 0.730 | 0.692 | 0.348 | 0.762 |
| mid_gls_relative_drop_10 | transmural_only_sparse | 101.000 | 46.000 | 0.577 | 0.517 | 0.239 | 0.524 |
| mid_gls_relative_drop_10 | variability_only_sparse | 101.000 | 46.000 | 0.693 | 0.626 | 0.283 | 0.619 |
| mid_gls_relative_drop_10 | clinical_plus_transmural | 101.000 | 46.000 | 0.607 | 0.587 | 0.261 | 0.571 |
| mid_gls_relative_drop_10 | clinical_plus_variability | 101.000 | 46.000 | 0.763 | 0.713 | 0.326 | 0.714 |
| mid_gls_relative_drop_10 | combined_sparse | 101.000 | 46.000 | 0.594 | 0.581 | 0.261 | 0.571 |
| mid_gls_relative_drop_10 | combined_extra_trees | 101.000 | 46.000 | 0.654 | 0.608 | 0.283 | 0.619 |
| mid_gls_relative_drop_12 | clinical_ridge | 101.000 | 40.000 | 0.682 | 0.548 | 0.325 | 0.619 |
| mid_gls_relative_drop_12 | transmural_only_sparse | 101.000 | 40.000 | 0.614 | 0.512 | 0.275 | 0.524 |
| mid_gls_relative_drop_12 | variability_only_sparse | 101.000 | 40.000 | 0.634 | 0.487 | 0.300 | 0.571 |
| mid_gls_relative_drop_12 | clinical_plus_transmural | 101.000 | 40.000 | 0.580 | 0.444 | 0.225 | 0.429 |
| mid_gls_relative_drop_12 | clinical_plus_variability | 101.000 | 40.000 | 0.662 | 0.583 | 0.300 | 0.571 |
| mid_gls_relative_drop_12 | combined_sparse | 101.000 | 40.000 | 0.577 | 0.454 | 0.275 | 0.524 |
| mid_gls_relative_drop_12 | combined_extra_trees | 101.000 | 40.000 | 0.625 | 0.512 | 0.250 | 0.476 |
| mid_gls_relative_drop_15 | clinical_ridge | 101.000 | 30.000 | 0.615 | 0.367 | 0.300 | 0.429 |
| mid_gls_relative_drop_15 | transmural_only_sparse | 101.000 | 30.000 | 0.615 | 0.457 | 0.267 | 0.381 |
| mid_gls_relative_drop_15 | variability_only_sparse | 101.000 | 30.000 | 0.653 | 0.449 | 0.333 | 0.476 |
| mid_gls_relative_drop_15 | clinical_plus_transmural | 101.000 | 30.000 | 0.554 | 0.370 | 0.233 | 0.333 |
| mid_gls_relative_drop_15 | clinical_plus_variability | 101.000 | 30.000 | 0.587 | 0.368 | 0.233 | 0.333 |
| mid_gls_relative_drop_15 | combined_sparse | 101.000 | 30.000 | 0.478 | 0.275 | 0.100 | 0.143 |
| mid_gls_relative_drop_15 | combined_extra_trees | 101.000 | 30.000 | 0.600 | 0.420 | 0.300 | 0.429 |
| mid_gls_relative_drop_20 | clinical_ridge | 101.000 | 11.000 | 0.497 | 0.118 | 0.091 | 0.048 |
| mid_gls_relative_drop_20 | transmural_only_sparse | 101.000 | 11.000 | 0.273 | 0.078 | 0.000 | 0.000 |
| mid_gls_relative_drop_20 | variability_only_sparse | 101.000 | 11.000 | 0.397 | 0.099 | 0.091 | 0.048 |
| mid_gls_relative_drop_20 | clinical_plus_transmural | 101.000 | 11.000 | 0.301 | 0.080 | 0.000 | 0.000 |
| mid_gls_relative_drop_20 | clinical_plus_variability | 101.000 | 11.000 | 0.358 | 0.090 | 0.000 | 0.000 |
| mid_gls_relative_drop_20 | combined_sparse | 101.000 | 11.000 | 0.379 | 0.091 | 0.091 | 0.048 |
| mid_gls_relative_drop_20 | combined_extra_trees | 101.000 | 11.000 | 0.494 | 0.113 | 0.091 | 0.048 |
| ef_absolute_drop_5 | clinical_ridge | 95.000 | 26.000 | 0.748 | 0.584 | 0.500 | 0.684 |
| ef_absolute_drop_5 | transmural_only_sparse | 95.000 | 26.000 | 0.412 | 0.281 | 0.154 | 0.211 |
| ef_absolute_drop_5 | variability_only_sparse | 95.000 | 26.000 | 0.613 | 0.361 | 0.269 | 0.368 |
| ef_absolute_drop_5 | clinical_plus_transmural | 95.000 | 26.000 | 0.615 | 0.393 | 0.308 | 0.421 |
| ef_absolute_drop_5 | clinical_plus_variability | 95.000 | 26.000 | 0.691 | 0.466 | 0.346 | 0.474 |
| ef_absolute_drop_5 | combined_sparse | 95.000 | 26.000 | 0.606 | 0.435 | 0.346 | 0.474 |
| ef_absolute_drop_5 | combined_extra_trees | 95.000 | 26.000 | 0.697 | 0.505 | 0.423 | 0.579 |
| incident_mid_gls_relative_drop_15 | clinical_ridge | 76.000 | 17.000 | 0.539 | 0.282 | 0.353 | 0.375 |
| incident_mid_gls_relative_drop_15 | transmural_only_sparse | 76.000 | 17.000 | 0.664 | 0.349 | 0.176 | 0.188 |
| incident_mid_gls_relative_drop_15 | variability_only_sparse | 76.000 | 17.000 | 0.590 | 0.292 | 0.294 | 0.312 |
| incident_mid_gls_relative_drop_15 | clinical_plus_transmural | 76.000 | 17.000 | 0.684 | 0.411 | 0.353 | 0.375 |
| incident_mid_gls_relative_drop_15 | clinical_plus_variability | 76.000 | 17.000 | 0.593 | 0.291 | 0.294 | 0.312 |
| incident_mid_gls_relative_drop_15 | combined_sparse | 76.000 | 17.000 | 0.655 | 0.294 | 0.235 | 0.250 |
| incident_mid_gls_relative_drop_15 | combined_extra_trees | 76.000 | 17.000 | 0.634 | 0.310 | 0.294 | 0.312 |
| incident_ef_absolute_drop_5 | clinical_ridge | 68.000 | 18.000 | 0.737 | 0.593 | 0.444 | 0.571 |
| incident_ef_absolute_drop_5 | transmural_only_sparse | 68.000 | 18.000 | 0.562 | 0.319 | 0.278 | 0.357 |
| incident_ef_absolute_drop_5 | variability_only_sparse | 68.000 | 18.000 | 0.636 | 0.383 | 0.278 | 0.357 |
| incident_ef_absolute_drop_5 | clinical_plus_transmural | 68.000 | 18.000 | 0.658 | 0.514 | 0.444 | 0.571 |
| incident_ef_absolute_drop_5 | clinical_plus_variability | 68.000 | 18.000 | 0.767 | 0.519 | 0.444 | 0.571 |
| incident_ef_absolute_drop_5 | combined_sparse | 68.000 | 18.000 | 0.722 | 0.516 | 0.389 | 0.500 |
| incident_ef_absolute_drop_5 | combined_extra_trees | 68.000 | 18.000 | 0.614 | 0.428 | 0.389 | 0.500 |

### Incremental performance at the most useful endpoints

| outcome | model | delta_roc_auc | delta_roc_auc_ci_low | delta_roc_auc_ci_high | delta_average_precision | delta_average_precision_ci_low | delta_average_precision_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mid_gls_relative_drop_15 | transmural_only_sparse | 0.000 | -0.148 | 0.147 | 0.090 | -0.059 | 0.231 |
| mid_gls_relative_drop_15 | variability_only_sparse | 0.038 | -0.104 | 0.180 | 0.083 | -0.088 | 0.225 |
| mid_gls_relative_drop_15 | clinical_plus_transmural | -0.061 | -0.202 | 0.076 | 0.004 | -0.135 | 0.130 |
| mid_gls_relative_drop_15 | clinical_plus_variability | -0.028 | -0.166 | 0.112 | 0.002 | -0.125 | 0.126 |
| mid_gls_relative_drop_15 | combined_sparse | -0.137 | -0.272 | -0.015 | -0.091 | -0.218 | -0.009 |
| mid_gls_relative_drop_15 | combined_extra_trees | -0.015 | -0.130 | 0.095 | 0.054 | -0.076 | 0.203 |
| ef_absolute_drop_5 | transmural_only_sparse | -0.336 | -0.501 | -0.168 | -0.304 | -0.505 | -0.099 |
| ef_absolute_drop_5 | variability_only_sparse | -0.135 | -0.274 | 0.009 | -0.223 | -0.406 | -0.004 |
| ef_absolute_drop_5 | clinical_plus_transmural | -0.133 | -0.275 | 0.014 | -0.191 | -0.361 | -0.003 |
| ef_absolute_drop_5 | clinical_plus_variability | -0.057 | -0.158 | 0.034 | -0.118 | -0.280 | 0.045 |
| ef_absolute_drop_5 | combined_sparse | -0.142 | -0.270 | 0.003 | -0.149 | -0.315 | 0.037 |
| ef_absolute_drop_5 | combined_extra_trees | -0.051 | -0.174 | 0.075 | -0.079 | -0.225 | 0.106 |
| incident_mid_gls_relative_drop_15 | transmural_only_sparse | 0.125 | -0.057 | 0.292 | 0.067 | -0.113 | 0.200 |
| incident_mid_gls_relative_drop_15 | variability_only_sparse | 0.051 | -0.125 | 0.238 | 0.010 | -0.144 | 0.185 |
| incident_mid_gls_relative_drop_15 | clinical_plus_transmural | 0.145 | -0.037 | 0.324 | 0.129 | -0.066 | 0.307 |
| incident_mid_gls_relative_drop_15 | clinical_plus_variability | 0.054 | -0.116 | 0.221 | 0.009 | -0.132 | 0.155 |
| incident_mid_gls_relative_drop_15 | combined_sparse | 0.116 | -0.073 | 0.293 | 0.012 | -0.144 | 0.133 |
| incident_mid_gls_relative_drop_15 | combined_extra_trees | 0.095 | -0.061 | 0.265 | 0.028 | -0.106 | 0.171 |

Positive deltas favor the proposed model except Brier-score deltas, where negative is better. Top-20% sensitivity/precision is an alert-budget summary, not a tuned clinical operating point.

## Threshold-independent prediction

| outcome | model | n | rmse | mae | spearman_rho |
| --- | --- | --- | --- | --- | --- |
| final_mid_relative_decline | clinical_ridge | 101.000 | 0.131 | 0.108 | 0.441 |
| final_mid_relative_decline | clinical_plus_transmural_ridge | 101.000 | 0.145 | 0.120 | 0.287 |
| final_mid_relative_decline | clinical_plus_variability_ridge | 101.000 | 0.138 | 0.117 | 0.358 |
| final_mid_relative_decline | combined_ridge | 101.000 | 0.143 | 0.118 | 0.298 |
| final_mid_relative_decline | combined_extra_trees | 101.000 | 0.133 | 0.110 | 0.417 |
| future_mid_relative_decline | clinical_ridge | 101.000 | 0.136 | 0.112 | 0.389 |
| future_mid_relative_decline | clinical_plus_transmural_ridge | 101.000 | 0.152 | 0.121 | 0.246 |
| future_mid_relative_decline | clinical_plus_variability_ridge | 101.000 | 0.145 | 0.119 | 0.312 |
| future_mid_relative_decline | combined_ridge | 101.000 | 0.149 | 0.118 | 0.265 |
| future_mid_relative_decline | combined_extra_trees | 101.000 | 0.140 | 0.114 | 0.342 |
| final_ef_absolute_decline | clinical_ridge | 95.000 | 5.353 | 4.026 | 0.509 |
| final_ef_absolute_decline | clinical_plus_transmural_ridge | 95.000 | 6.700 | 4.869 | -0.052 |
| final_ef_absolute_decline | clinical_plus_variability_ridge | 95.000 | 6.526 | 4.702 | 0.273 |
| final_ef_absolute_decline | combined_ridge | 95.000 | 6.854 | 4.947 | 0.026 |
| final_ef_absolute_decline | combined_extra_trees | 95.000 | 6.387 | 4.546 | 0.357 |
| future_ef_absolute_decline | clinical_ridge | 92.000 | 4.752 | 3.786 | 0.662 |
| future_ef_absolute_decline | clinical_plus_transmural_ridge | 92.000 | 6.519 | 5.050 | 0.355 |
| future_ef_absolute_decline | clinical_plus_variability_ridge | 92.000 | 6.105 | 4.803 | 0.515 |
| future_ef_absolute_decline | combined_ridge | 92.000 | 6.677 | 5.192 | 0.377 |
| future_ef_absolute_decline | combined_extra_trees | 92.000 | 5.562 | 4.288 | 0.563 |

### Incremental prediction after visit 2

| outcome | model | delta_rmse | delta_rmse_ci_low | delta_rmse_ci_high | delta_spearman_rho | delta_spearman_rho_ci_low | delta_spearman_rho_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| future_mid_relative_decline | clinical_plus_transmural_ridge | 0.015 | 0.003 | 0.028 | -0.144 | -0.278 | -0.017 |
| future_mid_relative_decline | clinical_plus_variability_ridge | 0.008 | -0.003 | 0.017 | -0.077 | -0.165 | 0.025 |
| future_mid_relative_decline | combined_ridge | 0.013 | -0.001 | 0.025 | -0.124 | -0.256 | 0.030 |
| future_mid_relative_decline | combined_extra_trees | 0.004 | -0.006 | 0.012 | -0.047 | -0.154 | 0.062 |
| future_ef_absolute_decline | clinical_plus_transmural_ridge | 1.766 | 0.966 | 2.562 | -0.307 | -0.463 | -0.156 |
| future_ef_absolute_decline | clinical_plus_variability_ridge | 1.353 | 0.696 | 2.036 | -0.147 | -0.256 | -0.036 |
| future_ef_absolute_decline | combined_ridge | 1.925 | 0.984 | 2.871 | -0.285 | -0.461 | -0.128 |
| future_ef_absolute_decline | combined_extra_trees | 0.810 | 0.193 | 1.412 | -0.100 | -0.208 | 0.008 |

Negative delta RMSE is better. These outcomes ask the hardest and cleanest question: after accounting for measurements already available at visit 2, do the curve hypotheses predict further deterioration?

## Clinical-adjusted feature screen

Each candidate feature and outcome was separately residualized for the full early clinical trajectory, then correlated. FDR is across all Endo-Mid and variability candidates for each outcome.

| outcome | feature_family | feature | n | partial_spearman_rho | ci_low | ci_high | p_value | fdr_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_ef_absolute_decline | transmural | d_tm_sd_gap_dct06 | 95.000 | 0.312 | 0.100 | 0.504 | 0.002 | 0.738 |
| future_ef_absolute_decline | transmural | d_tm_sd_gap_dct06 | 92.000 | 0.300 | 0.078 | 0.483 | 0.004 | 1.000 |
| final_mid_relative_decline | variability | v2_var__mid_vendor_peak_systolic_abs_robust_sd | 101.000 | -0.284 | -0.460 | -0.089 | 0.004 | 0.995 |
| future_mid_relative_decline | variability | v2_var__mid_vendor_peak_systolic_abs_robust_sd | 101.000 | -0.278 | -0.463 | -0.068 | 0.005 | 0.992 |
| final_mid_relative_decline | transmural | v2_tm_shape_rms_std | 101.000 | -0.260 | -0.441 | -0.048 | 0.009 | 0.995 |
| future_mid_relative_decline | transmural | v2_tm_shape_rms_std | 101.000 | -0.243 | -0.418 | -0.046 | 0.014 | 0.992 |
| future_ef_absolute_decline | transmural | v1_tm_mean_gap_dct04 | 92.000 | -0.249 | -0.453 | -0.041 | 0.017 | 1.000 |
| final_mid_relative_decline | transmural | v2_tm_mean_shape_gap_dct01 | 101.000 | 0.236 | 0.060 | 0.412 | 0.018 | 0.995 |
| future_ef_absolute_decline | transmural | v1_tm_vendor_peak_gap_robust_sd | 92.000 | -0.243 | -0.448 | -0.029 | 0.020 | 1.000 |
| final_mid_relative_decline | transmural | v2_tm_mean_shape_gap_dct03 | 101.000 | -0.230 | -0.404 | -0.037 | 0.021 | 0.995 |
| final_ef_absolute_decline | variability | d_var__mid_impaired_segment_fraction_lt15 | 95.000 | 0.237 | 0.046 | 0.416 | 0.021 | 1.000 |
| future_ef_absolute_decline | variability | d_var__mid_impaired_segment_fraction_lt15 | 92.000 | 0.240 | 0.044 | 0.414 | 0.021 | 1.000 |

## Primary-model coefficient stability

The combined sparse logistic model was refit in class-stratified bootstrap samples. Correlated features can substitute for one another, so selection fractions are more informative than any single full-cohort coefficient.

| feature_family | feature | selection_fraction | sign_consistency_when_selected | coefficient_median | importance_score |
| --- | --- | --- | --- | --- | --- |
| transmural | v2_tm_segment_curve_correlation_std | 0.567 | 1.000 | -0.050 | 0.029 |
| transmural | v1_tm_segment_curve_correlation_mean | 0.513 | 1.000 | 0.010 | 0.005 |
| clinical | early_mid_abs_decline | 0.043 | 1.000 | 0.000 | 0.000 |
| clinical | early_mid_rel_decline | 0.043 | 1.000 | 0.000 | 0.000 |
| clinical | v1_endo_gls | 0.030 | 1.000 | 0.000 | 0.000 |
| clinical | v2_endo_gls | 0.000 |  | 0.000 | 0.000 |
| clinical | early_endo_abs_decline | 0.007 | 1.000 | 0.000 | 0.000 |
| clinical | early_endo_rel_decline | 0.000 |  | 0.000 | 0.000 |
| clinical | v1_ef | 0.027 | 1.000 | 0.000 | 0.000 |
| clinical | v2_ef | 0.010 | 1.000 | 0.000 | 0.000 |
| clinical | early_ef_abs_decline | 0.147 | 1.000 | 0.000 | 0.000 |
| transmural | v1_tm_peak_layer_correlation | 0.000 |  | 0.000 | 0.000 |
| variability | v2_var__mid_curve_pairwise_rmse | 0.000 |  | 0.000 | 0.000 |
| transmural | v2_tm_peak_gap_std | 0.000 |  | 0.000 | 0.000 |
| variability | d_var__mid_within_view_peak_robust_sd_mean | 0.000 |  | 0.000 | 0.000 |

## Interpretation and next modeling direction

- A feature family is considered supported only if it improves patient-held-out ranking and continuous future-decline prediction, with paired intervals that do not comfortably include no improvement.
- If direct Endo-Mid waveform summaries help but scalar gaps do not, the next model should be a segment-aware functional mixed model rather than a larger black-box CNN.
- If variability helps mainly through visit-1 to visit-2 changes, the next design should model segment-specific random slopes and persistent abnormal segments, with technical-reanalysis variance explicitly separated from biological change.
- If neither survives the incident and continuous analyses, the dataset is likely too small/noisy for supervised curve modeling; the highest-value next step is more patients, reproducible reprocessing, and an independently adjudicated endpoint.

## Limitations

- Only 101 landmark patients and no external test cohort.
- Many correlated candidate features and several exploratory endpoints; FDR is applied only to the univariate screen, not to the model-comparison table.
- The final visit occurs at varying follow-up times; time to the final visit was deliberately not used as an early predictor because it is not generally known at the landmark.
- EF is missing for some visits and its largest-drop threshold has very few events.
- Strain and EF are analysis outputs, not adjudicated clinical cardiotoxicity. Treatment, dose, biomarkers, symptoms, and comorbidities are unavailable.
- Internal technical reanalysis variation is large enough to explain some moderate patient-level changes.

## Files

- `landmark_patient_features.parquet`: patient-level predictors/outcomes.
- `label_audit.csv`: threshold prevalence and eligible populations.
- `classification_metrics.csv`, `classification_predictions.parquet`, `classification_deltas_vs_clinical.csv`.
- `continuous_metrics.csv`, `continuous_predictions.parquet`, `continuous_deltas_vs_clinical.csv`.
- `partial_feature_associations.csv`, `primary_sparse_logistic_stability.csv`, `feature_manifest.csv`.
- `figures/`: threshold, model comparison, alert-budget, and continuous-outcome figures.
