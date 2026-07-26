# Focused plateau experiments

## Bottom line

Small segment attention produced the best point estimate (AUC 0.683, AP 0.333), but its paired gain over uniform pooling was not bootstrap-stable. The continuous next-visit GLS auxiliary target was only moderately learnable and did not improve classification. Measurement noise makes individual threshold-crossing labels unstable, but the simulated noise-only oracle remains well above the achieved AUC, so label noise is a contributor rather than the entire plateau.

## Primary task: first baseline, 15% relative Mid-GLS decline

All predictions are from three repeated five-fold patient-held-out validation. Confidence intervals use 500 patient-cluster bootstrap resamples.

| model | n | events | roc_auc | roc_auc_ci_low | roc_auc_ci_high | average_precision | average_precision_ci_low | average_precision_ci_high | sensitivity_top20pct | precision_top20pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention_binary | 238.000 | 49.000 | 0.683 | 0.607 | 0.756 | 0.333 | 0.248 | 0.456 | 0.265 | 0.271 |
| attention_gls_aux | 238.000 | 49.000 | 0.658 | 0.577 | 0.728 | 0.307 | 0.231 | 0.406 | 0.306 | 0.312 |
| uniform_binary | 238.000 | 49.000 | 0.664 | 0.576 | 0.731 | 0.302 | 0.224 | 0.428 | 0.286 | 0.292 |
| uniform_gls_aux | 238.000 | 49.000 | 0.659 | 0.578 | 0.740 | 0.291 | 0.211 | 0.430 | 0.286 | 0.292 |
| clinical_ridge | 238.000 | 49.000 | 0.631 | 0.544 | 0.727 | 0.289 | 0.214 | 0.435 | 0.327 | 0.333 |

Paired ablations:

| comparison | delta_roc_auc | delta_roc_auc_ci_low | delta_roc_auc_ci_high | delta_average_precision | delta_average_precision_ci_low | delta_average_precision_ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| attention_vs_uniform | 0.019 | -0.017 | 0.054 | 0.031 | -0.019 | 0.066 |
| uniform_aux_vs_binary | -0.005 | -0.037 | 0.028 | -0.010 | -0.054 | 0.029 |
| attention_aux_vs_binary | -0.025 | -0.065 | 0.016 | -0.025 | -0.079 | 0.022 |
| attention_aux_vs_uniform | -0.006 | -0.060 | 0.037 | 0.006 | -0.076 | 0.066 |

Attention used 17,491 parameters versus 16,746 for uniform pooling. Versus clinical ridge, its paired delta was +0.052 AUC (CI -0.044 to +0.139) and +0.043 AP (CI -0.084 to +0.143), also not stable.

Attention versus the existing combined Extra Trees model:

| comparison | delta_roc_auc | delta_roc_auc_ci_low | delta_roc_auc_ci_high | delta_average_precision | delta_average_precision_ci_low | delta_average_precision_ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| attention_vs_existing_trees | 0.011 | -0.068 | 0.093 | 0.019 | -0.110 | 0.126 |

### Threshold stability

| endpoint | attention AUC | clinical AUC | attention AP | clinical AP |
| --- | ---: | ---: | ---: | ---: |
| first baseline, 10% | 0.640 | 0.690 | 0.428 | 0.482 |
| first baseline, 12% | 0.641 | 0.635 | 0.365 | 0.346 |
| first baseline, 15% | 0.683 | 0.631 | 0.333 | 0.289 |
| first baseline, 20% | 0.621 | 0.712 | 0.158 | 0.288 |

The attention advantage is concentrated around the 12–15% first-baseline definitions. It does not generalize to 10%, 20%, or rolling-2 endpoints, where the clinical trajectory usually remains better.

## Segment attention

Uniform weight is 0.0556. The learned averages stayed close to uniform and event/non-event differences were not stable.

| segment | segment_name | mean_weight | mean_weight_ci_low | mean_weight_ci_high | event_minus_nonevent | difference_ci_low | difference_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9.0000 | mid inferoseptal | 0.0601 | 0.0594 | 0.0608 | -0.0004 | -0.0013 | 0.0007 |
| 18.0000 | apical anterolateral | 0.0600 | 0.0593 | 0.0606 | 0.0000 | -0.0010 | 0.0012 |
| 16.0000 | apical inferior | 0.0595 | 0.0584 | 0.0607 | -0.0005 | -0.0021 | 0.0016 |
| 8.0000 | mid anteroseptal | 0.0590 | 0.0581 | 0.0598 | -0.0008 | -0.0023 | 0.0006 |
| 13.0000 | apical anterior | 0.0589 | 0.0582 | 0.0597 | -0.0005 | -0.0017 | 0.0008 |
| 10.0000 | mid inferior | 0.0586 | 0.0579 | 0.0594 | -0.0002 | -0.0015 | 0.0012 |

Interpretation: the point-estimate gain is more consistent with mild adaptive reweighting/regularization than a strong anatomical segment signature.

## Continuous GLS auxiliary target

| variant | n | mae | mae_ci_low | mae_ci_high | r2 | pearson_r | pearson_r_ci_low | pearson_r_ci_high | spearman_r |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention_gls_aux | 297.000 | 0.106 | 0.095 | 0.117 | 0.142 | 0.387 | 0.313 | 0.462 | 0.359 |
| uniform_gls_aux | 297.000 | 0.105 | 0.095 | 0.116 | 0.164 | 0.407 | 0.322 | 0.489 | 0.375 |

The target was immediate relative Mid-GLS change. MAE 0.105 means roughly 10.5 relative percentage points, and R² was only 0.14–0.16. Adding its SmoothL1 loss reduced primary AUC/AP for both pooling methods.

## Error analysis

At the global top-20% alert threshold, the attention model produced 36 false negatives and 35 false positives. Despite higher overall AUC/AP, its primary sensitivity at this alert budget was 0.265, below the clinical ridge value of 0.327.

Two recurrent failure patterns were visible among visits after V1:

- False positives frequently represented near-threshold trajectories that did not persist: 54% already had a 10% or larger current decline, versus 11% of true negatives.
- False negatives were more abrupt: their median current decline was 6.6%, compared with 10.0% for true positives. The next visit therefore contained deterioration that was weakly expressed at the alert visit.

Subgroup performance was also unstable. Attention AUC was 0.714 with baseline GLS <18 but only 0.559 with baseline GLS ≥18; it was 0.713 after V1 but 0.607 when predicting directly from V1. The middle Endo–Mid-gap tertile was particularly poor (AUC 0.444), while the low and high tertiles performed much better, suggesting an unstable/U-shaped interaction rather than a robust monotonic transmural signal.

Weakest estimable subgroups:

| subgroup_variable | subgroup | n | events | roc_auc | roc_auc_ci_low | roc_auc_ci_high | average_precision | sensitivity_at_global_top20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| endo_mid_gap_group | gap middle | 78.000 | 17.000 | 0.444 | 0.305 | 0.597 | 0.244 | 0.059 |
| label_margin_group | 3-8% from threshold | 50.000 | 18.000 | 0.531 | 0.355 | 0.721 | 0.453 | 0.222 |
| current_decline_group | >=10% | 23.000 | 5.000 | 0.556 | 0.333 | 0.777 | 0.260 | 0.400 |
| baseline_gls_group | baseline GLS >=18 | 77.000 | 25.000 | 0.559 | 0.422 | 0.687 | 0.403 | 0.320 |
| variability_group | variability low | 81.000 | 24.000 | 0.593 | 0.485 | 0.693 | 0.370 | 0.250 |
| history_group | current=V1 | 103.000 | 26.000 | 0.607 | 0.476 | 0.723 | 0.359 | 0.346 |
| followup_interval_group | middle | 79.000 | 11.000 | 0.634 | 0.440 | 0.816 | 0.214 | 0.364 |
| variability_group | variability middle | 77.000 | 17.000 | 0.635 | 0.469 | 0.789 | 0.366 | 0.353 |

Calibration warning:

| model | quintile_ece |
| --- | --- |
| clinical_ridge | 0.256 |
| existing_combined_trees | 0.101 |
| attention_binary | 0.263 |

Scores come from class-weighted models and should be interpreted as rankings, not calibrated probabilities.

## Strain label noise and ceiling

There were only 16 paired strain analyses. Estimated within-visit Mid-GLS SD was 0.942 points (95% bootstrap CI 0.508–1.315); robust estimate 0.771. Mean absolute replicate difference was 0.915 points. The small replicate sample makes this uncertain.

15% endpoint simulation:

| noise_sd_gls_points | expected_observed_label_flip_fraction | observed_event_identity_reproducibility | ambiguous_observed_fraction_p10_p90 | oracle_auc_mean | oracle_auc_sim_low | oracle_auc_sim_high | oracle_ap_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.500 | 0.069 | 0.842 | 0.218 | 0.979 | 0.964 | 0.990 | 0.932 |
| 0.771 | 0.099 | 0.785 | 0.315 | 0.956 | 0.932 | 0.976 | 0.869 |
| 0.942 | 0.117 | 0.752 | 0.361 | 0.939 | 0.907 | 0.964 | 0.828 |
| 1.500 | 0.168 | 0.671 | 0.542 | 0.876 | 0.826 | 0.920 | 0.713 |

The oracle assumes perfect knowledge of the latent GLS trajectory and only random measurement error at observation. It is therefore an optimistic ceiling, not an expected model score. If the replicate SD near 0.8–0.9 points is realistic, a meaningful fraction of exact first-crossing identities is unstable; however, the oracle AUC remains above the achieved 0.683, showing that missing predictors and limited sample size still dominate.

## Recommendation

- Keep attention as an optional ensemble component, not a replacement justified by this cohort alone.
- Do not keep the GLS auxiliary loss at weight 0.25; it did not improve classification.
- Evaluate alerts over a confirmation window or require repeated deterioration to reduce threshold noise.
- Prioritize additional treatment/biomarker timing and more independent events before increasing model complexity.
