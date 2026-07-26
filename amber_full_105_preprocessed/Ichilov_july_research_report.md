# Ichilov longitudinal strain-curve research report

## Executive result

This is an exploratory, patient-level longitudinal analysis. The main early-warning test models the final functional endpoint from its baseline value, follow-up duration, and one baseline variability feature at a time; EF models also control baseline GLS and mean segment peak magnitude. Continuous GLS magnitude is primary; biplane EF is secondary because EF is missing in some studies.

The pre-specified early-warning hypothesis was not supported: neither baseline robust peak dispersion nor normalized timing dispersion independently predicted a lower final GLS magnitude or EF after multiplicity correction.

A more convincing cross-sectional signal did appear within patients: when normalized Endo timing dispersion was above a patient's usual level, GLS magnitude was lower (patient-centered r=-0.35, FDR q=0.000533). Shape incoherence showed the same direction.

Direct next-visit models likewise found no GLS early-warning signal. Some amplitude features weakly predicted EF in the opposite direction (higher variability preceding higher EF), consistent with amplitude scaling or compensation rather than the proposed deterioration mechanism.

For EF, the median same-study reanalysis difference was 0.70× the median adjacent-visit change; for Endo GLS it was 0.64×. This is a substantial measurement floor for individual-patient trajectory claims.

The duplicate Study UIDs reveal appreciable technical reanalysis variation. Moderate first-to-last changes can therefore be analytic rather than biological. Findings should be treated as hypothesis-generating until validated on independently processed studies.

## Dataset and quality control

- Source exports: 416
- Patients: 103
- True visits (unique Study UID): 400
- Technical reanalysis pairs: 16
- Curve-series rows in Parquet: 36961
- Original curve samples: 2145357
- Parse failures or length errors: 0
- True visits without biplane EF: 16
- Median intervisit interval: 96.2 days (range 16.8–970.2)

The visit identity is Study UID and the visit order uses the internal Study Date and Time—not the filename export timestamp. Reanalyses are retained in the curve Parquet, then averaged within Study UID for longitudinal analysis.

Curve-derived time-to-peak was cross-checked against the vendor table: median absolute difference 3.44 ms and correlation 0.9938. The curve global minimum and vendor peak-systolic amplitude differ when strong post-systolic shortening occurs; both versions are retained as separate sensitivity features.

## Internal repeatability finding

The provisional 95% repeatability coefficient was 2.90 percentage points for signed Endo GLS and 7.17 points for biplane EF. These estimates use only the duplicated studies and are clustered in four patients.

| metric | n_pairs | mean_absolute_difference | repeatability_coefficient_95 | icc_consistency |
| --- | --- | --- | --- | --- |
| ef_biplane | 11 | 2.895 | 7.17 | 0.771 |
| endo_curve_integrated_robust_mad | 16 | 0.4543 | 1.203 | 0.7188 |
| endo_peak_abs_robust_sd | 16 | 1.175 | 3.196 | 0.5375 |
| endo_time_to_peak_norm_circular_std | 16 | 0.01208 | 0.04072 | 0.6054 |
| gls_endo_peak_avg | 16 | 1.164 | 2.897 | 0.9488 |
| gls_mid_peak_avg | 16 | 0.915 | 2.611 | 0.9538 |

Using those analytic-repeatability thresholds as a sensitivity label, 35 of 103 patients had a GLS-magnitude decline beyond repeatability, 13 had an EF decline beyond repeatability, and 8 met both. These are not clinical event definitions.

### Patient trajectories worth manual review

These rows had the largest GLS decline beyond the provisional reanalysis threshold, with concordant EF decline sorted first. They are review candidates—not adjudicated clinical deteriorators.

| patient_id | n_visits | followup_years | baseline_gls_magnitude | final_gls_magnitude | gls_worsening_pp | baseline_ef | final_ef | ef_worsening_pp | concordant_decline_beyond_repeatability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5881605-9 | 4 | 0.5969 | 23.66 | 18.25 | 5.41 | 70.38 | 57.58 | 12.8 | True |
| 31368451-6 | 4 | 0.4162 | 20.04 | 14.75 | 5.29 | 68.68 | 53.42 | 15.26 | True |
| 723177-2 | 4 | 0.8159 | 20.67 | 16.36 | 4.31 | 70.77 | 62.96 | 7.81 | True |
| 2873723-7 | 4 | 3.006 | 26.49 | 22.39 | 4.1 | 72.52 | 60.78 | 11.74 | True |
| 33407814-4 | 4 | 0.9008 | 23.42 | 19.51 | 3.91 | 64.83 | 56.38 | 8.45 | True |
| 31015650-0 | 4 | 0.742 | 21.17 | 17.79 | 3.38 | 71.53 | 62.55 | 8.98 | True |
| 34269230-8 | 4 | 0.898 | 24.82 | 21.63 | 3.19 | 70.09 | 58.43 | 11.66 | True |
| 2484127-2 | 4 | 0.2026 | 12.52 | 9.6 | 2.92 | 41.57 | 32.37 | 9.2 | True |
| 30314808-4 | 3 | 1.12 | 24.85 | 17.53 | 7.32 | 65.28 | 62.18 | 3.1 | False |
| 3665822-7 | 2 | 0.8569 | 23.71 | 18.18 | 5.535 | 73.7 | 71.8 | 1.9 | False |

A useful reliability gate is the ratio of same-study reanalysis difference to the typical adjacent-visit change. Ratios near or above one mean apparent longitudinal movement can readily be explained by analysis variation.

| metric | technical_median_absolute_difference | longitudinal_median_absolute_change | technical_to_longitudinal_median_ratio |
| --- | --- | --- | --- |
| ef_biplane | 2.35 | 3.37 | 0.6973 |
| gls_endo_peak_avg | 1.165 | 1.83 | 0.6366 |
| endo_curve_integrated_robust_mad | 0.3786 | 0.6352 | 0.5961 |
| endo_peak_abs_robust_sd | 0.8265 | 1.46 | 0.566 |
| endo_time_to_peak_norm_circular_std | 0.006344 | 0.01275 | 0.4977 |

## Baseline variability as an early signal

Negative standardized beta means higher baseline variability predicts a worse (lower) final GLS magnitude or EF after controlling for the corresponding baseline endpoint and follow-up duration. Negative LOOCV ΔRMSE means the feature improved held-out prediction.

| endpoint | feature | n_patients | standardized_beta_feature | bootstrap_ci_low | bootstrap_ci_high | permutation_p | fdr_q | loocv_delta_rmse_extended_minus_base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_ef | endo_within_ring_peak_robust_sd_mean | 97 | 0.2309 | 0.009892 | 0.3984 | 0.012 | 0.2697 | -0.1843 |
| final_ef | endo_peak_abs_robust_sd | 97 | 0.206 | -0.01059 | 0.4113 | 0.02539 | 0.2697 | -0.09244 |
| final_ef | endo_within_view_peak_robust_sd_mean | 97 | 0.1768 | -0.009132 | 0.3512 | 0.05499 | 0.2697 | -0.06472 |
| final_ef | endo_spatial_peak_graph_roughness | 97 | 0.1773 | -0.05976 | 0.3807 | 0.05539 | 0.2697 | -0.05502 |
| final_ef | mid_peak_abs_robust_sd | 97 | 0.1719 | -0.08894 | 0.3814 | 0.05619 | 0.2697 | -0.03475 |
| final_ef | endo_time_to_peak_ms_std | 97 | 0.1744 | -0.07134 | 0.3795 | 0.07638 | 0.3055 | -0.0163 |
| final_gls_magnitude | mid_peak_abs_robust_sd | 103 | 0.1145 | -0.05771 | 0.2819 | 0.1744 | 0.8718 | 0.000598 |
| final_gls_magnitude | endo_shape_incoherence | 103 | 0.08552 | -0.1029 | 0.2665 | 0.3197 | 0.8718 | 0.01738 |
| final_gls_magnitude | endo_impaired_segment_fraction_lt15 | 103 | 0.1481 | -0.1271 | 0.4277 | 0.3231 | 0.8718 | 0.009788 |
| final_gls_magnitude | endo_within_ring_peak_robust_sd_mean | 103 | 0.07605 | -0.1046 | 0.2196 | 0.3729 | 0.8718 | 0.01155 |
| final_gls_magnitude | endo_spatial_peak_graph_roughness | 103 | 0.07595 | -0.1014 | 0.2308 | 0.3847 | 0.8718 | 0.01115 |
| final_gls_magnitude | endo_shape_dispersion_rms | 103 | 0.06518 | -0.1018 | 0.2245 | 0.4695 | 0.8718 | 0.01446 |

The two pre-specified primary features were Endo robust peak-amplitude dispersion (`1.4826 × MAD`) and circular dispersion of normalized time-to-peak. Their four endpoint tests additionally have Holm-adjusted p-values in the full CSV.

### Technical-replicate sensitivity

The primary models were repeated using the mean, first export, and latest export for duplicated Study UIDs. Stable direction across these rows is more credible than a result that depends on export choice.

| replicate_strategy | endpoint | feature | n_patients | standardized_beta_feature | bootstrap_ci_low | bootstrap_ci_high | permutation_p | loocv_delta_rmse_extended_minus_base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replicate_mean | final_gls_magnitude | endo_peak_abs_robust_sd | 103 | 0.0273 | -0.1442 | 0.1929 | 0.7369 | 0.02449 |
| replicate_mean | final_gls_magnitude | endo_time_to_peak_norm_circular_std | 103 | -0.03557 | -0.205 | 0.1444 | 0.7003 | 0.02705 |
| replicate_mean | final_ef | endo_peak_abs_robust_sd | 97 | 0.206 | -0.01059 | 0.4113 | 0.02539 | -0.09244 |
| replicate_mean | final_ef | endo_time_to_peak_norm_circular_std | 97 | 0.04879 | -0.1409 | 0.2541 | 0.6213 | 0.05836 |
| first_export | final_gls_magnitude | endo_peak_abs_robust_sd | 103 | 0.03834 | -0.1264 | 0.1996 | 0.6473 | 0.02251 |
| first_export | final_gls_magnitude | endo_time_to_peak_norm_circular_std | 103 | -0.02437 | -0.2004 | 0.1485 | 0.7932 | 0.02614 |
| first_export | final_ef | endo_peak_abs_robust_sd | 96 | 0.2051 | -0.01426 | 0.4117 | 0.025 | -0.09583 |
| first_export | final_ef | endo_time_to_peak_norm_circular_std | 96 | 0.04051 | -0.1535 | 0.2381 | 0.6861 | 0.05994 |
| latest_export | final_gls_magnitude | endo_peak_abs_robust_sd | 103 | 0.01031 | -0.1577 | 0.1772 | 0.9004 | 0.02544 |
| latest_export | final_gls_magnitude | endo_time_to_peak_norm_circular_std | 103 | -0.03762 | -0.2144 | 0.1399 | 0.6713 | 0.02732 |
| latest_export | final_ef | endo_peak_abs_robust_sd | 96 | 0.2032 | -0.007444 | 0.4237 | 0.0228 | -0.08376 |
| latest_export | final_ef | endo_time_to_peak_norm_circular_std | 96 | 0.05769 | -0.132 | 0.2576 | 0.5537 | 0.05553 |

### Direct next-visit test

This complementary model predicts the next visit from the current endpoint, interval length, and current variability; EF models also control current GLS magnitude and mean segment peak magnitude. Inference uses patient-cluster sign flips/bootstrap, and prediction is leave-one-patient-out. Negative beta means more current variability precedes a worse (lower) next endpoint; negative ΔRMSE means better held-out prediction.

| endpoint | feature | n_transitions | n_patients | standardized_beta_feature | patient_bootstrap_ci_low | patient_bootstrap_ci_high | cluster_signflip_p | fdr_q | lopo_delta_rmse_extended_minus_base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ef_biplane | endo_vendor_peak_systolic_abs_robust_sd | 276 | 101 | 0.1103 | 0.01374 | 0.1996 | 0.03019 | 0.1131 | -0.03035 |
| ef_biplane | endo_curve_integrated_robust_mad | 276 | 101 | 0.08632 | 0.001001 | 0.1681 | 0.06919 | 0.1131 | -0.01402 |
| ef_biplane | endo_vendor_time_to_peak_norm_circular_std | 276 | 101 | 0.08864 | -0.006465 | 0.179 | 0.07898 | 0.1131 | -0.009175 |
| ef_biplane | endo_shape_incoherence | 276 | 101 | 0.1066 | -0.003698 | 0.2236 | 0.08138 | 0.1131 | -0.006729 |
| ef_biplane | endo_time_to_peak_norm_circular_std | 276 | 101 | 0.08293 | -0.01301 | 0.1768 | 0.08858 | 0.1131 | -0.005229 |
| gls_endo_magnitude | endo_vendor_peak_systolic_abs_robust_sd | 297 | 103 | 0.08353 | -0.002702 | 0.1648 | 0.06719 | 0.4703 | -0.007537 |
| gls_endo_magnitude | endo_curve_integrated_robust_mad | 297 | 103 | 0.05185 | -0.03297 | 0.1277 | 0.235 | 0.489 | 0.0009097 |
| gls_endo_magnitude | endo_peak_abs_robust_sd | 297 | 103 | 0.05033 | -0.03852 | 0.1279 | 0.2434 | 0.489 | 0.0007891 |
| gls_endo_magnitude | endo_time_to_peak_norm_circular_std | 297 | 103 | 0.04743 | -0.0495 | 0.1385 | 0.3303 | 0.489 | 0.003968 |
| gls_endo_magnitude | endo_shape_incoherence | 297 | 103 | 0.04364 | -0.04466 | 0.1378 | 0.3631 | 0.489 | 0.005719 |

## Descriptive first-to-last change screen

Positive rho means higher baseline variability accompanied greater later functional worsening. This is easier to read than ANCOVA but is more susceptible to regression to the mean.

| outcome | feature | n_patients | spearman_rho | bootstrap_ci_low | bootstrap_ci_high | permutation_p | fdr_q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ef_worsening_pp | endo_impaired_segment_fraction_lt15 | 97 | -0.256 | -0.4625 | -0.03723 | 0.0132 | 0.3167 |
| ef_worsening_pp | endo_within_view_peak_robust_sd_mean | 97 | -0.2157 | -0.4016 | -0.00855 | 0.03339 | 0.4007 |
| ef_worsening_pp | endo_peak_abs_cv | 97 | -0.1849 | -0.3809 | 0.03255 | 0.07139 | 0.5039 |
| ef_worsening_pp | endo_peak_abs_robust_sd | 97 | -0.1742 | -0.3629 | 0.03463 | 0.08818 | 0.5039 |
| ef_worsening_pp | endo_within_ring_peak_robust_sd_mean | 97 | -0.1579 | -0.3523 | 0.05671 | 0.1212 | 0.5039 |
| gls_worsening_pp | endo_impaired_segment_fraction_lt15 | 103 | -0.4682 | -0.6081 | -0.2981 | 0.0002 | 0.004799 |
| gls_worsening_pp | endo_peak_abs_cv | 103 | -0.2296 | -0.4039 | -0.04518 | 0.0204 | 0.1792 |
| gls_worsening_pp | endo_apical_basal_peak_gradient | 103 | 0.2221 | 0.02373 | 0.4019 | 0.0224 | 0.1792 |
| gls_worsening_pp | endo_shape_dispersion_rms | 103 | -0.2027 | -0.398 | -0.01371 | 0.04539 | 0.2352 |
| gls_worsening_pp | endo_shape_incoherence | 103 | -0.1954 | -0.3831 | -0.00602 | 0.04899 | 0.2352 |

## Early two-visit evolution and spatial persistence

This secondary screen asks whether change during the first two visits—or persistence of the same abnormal segment pattern—precedes deterioration after visit 2.

| outcome | predictor | n_patients | spearman_rho | bootstrap_ci_low | bootstrap_ci_high | permutation_p | fdr_q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| post_early_ef_worsening_pp | early_change_endo_shape_incoherence | 92 | -0.3585 | -0.5295 | -0.1699 | 0.0009998 | 0.01836 |
| post_early_ef_worsening_pp | early_change_endo_time_to_peak_ms_std | 92 | -0.3243 | -0.5086 | -0.1346 | 0.0016 | 0.01836 |
| post_early_ef_worsening_pp | early_change_endo_spatial_timing_graph_roughness | 92 | -0.3185 | -0.4939 | -0.1379 | 0.0024 | 0.01836 |
| post_early_ef_worsening_pp | early_change_endo_time_to_peak_norm_circular_std | 92 | -0.3075 | -0.471 | -0.1311 | 0.003399 | 0.01836 |
| post_early_ef_worsening_pp | early_change_endo_vendor_time_to_peak_norm_circular_std | 92 | -0.299 | -0.4761 | -0.1033 | 0.003399 | 0.01836 |
| post_early_gls_worsening_pp | early_change_endo_impaired_segment_fraction_lt15 | 101 | -0.2966 | -0.4651 | -0.114 | 0.002599 | 0.07019 |
| post_early_gls_worsening_pp | early_change_endo_time_to_peak_es_ratio_std | 101 | -0.2699 | -0.4428 | -0.072 | 0.007399 | 0.09988 |
| post_early_gls_worsening_pp | early_change_endo_shape_dispersion_rms | 101 | -0.2393 | -0.4124 | -0.06523 | 0.0172 | 0.1242 |
| post_early_gls_worsening_pp | early_change_endo_time_to_peak_norm_std | 101 | -0.239 | -0.411 | -0.05829 | 0.0184 | 0.1242 |
| post_early_gls_worsening_pp | early_change_endo_time_to_peak_norm_circular_std | 101 | -0.2219 | -0.3981 | -0.0403 | 0.02619 | 0.1274 |

## Within-patient contemporaneous association

Negative correlation means that visits with above-usual variability for that patient also had below-usual GLS magnitude or EF. This controls stable between-patient differences but is not a prospective test.

| outcome | feature | n_visits | n_patients | patient_centered_pearson_r | cluster_bootstrap_ci_low | cluster_bootstrap_ci_high | cluster_signflip_p | fdr_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ef_biplane | endo_shape_incoherence | 384 | 102 | -0.2226 | -0.3244 | -0.1096 | 0.0007998 | 0.003199 |
| ef_biplane | mid_time_to_peak_norm_circular_std | 384 | 102 | -0.2449 | -0.3653 | -0.1089 | 0.0007998 | 0.003199 |
| ef_biplane | endo_time_to_peak_norm_circular_std | 384 | 102 | -0.2041 | -0.3157 | -0.07965 | 0.0016 | 0.004266 |
| ef_biplane | endo_spatial_peak_graph_roughness | 384 | 102 | 0.03878 | -0.0709 | 0.1409 | 0.4767 | 0.6609 |
| ef_biplane | endo_peak_abs_robust_sd | 384 | 102 | -0.03098 | -0.1271 | 0.06893 | 0.5595 | 0.6609 |
| gls_endo_magnitude | endo_time_to_peak_norm_circular_std | 400 | 103 | -0.3456 | -0.4534 | -0.2338 | 0.0002 | 0.0005332 |
| gls_endo_magnitude | endo_shape_incoherence | 400 | 103 | -0.356 | -0.472 | -0.2176 | 0.0002 | 0.0005332 |
| gls_endo_magnitude | mid_time_to_peak_norm_circular_std | 400 | 103 | -0.3654 | -0.4714 | -0.2629 | 0.0002 | 0.0005332 |
| gls_endo_magnitude | endo_spatial_peak_graph_roughness | 400 | 103 | 0.2435 | 0.126 | 0.3572 | 0.0005999 | 0.0012 |
| gls_endo_magnitude | endo_curve_integrated_robust_mad | 400 | 103 | 0.1797 | 0.05323 | 0.3037 | 0.006199 | 0.009918 |

## Interpretation limits

- There are only 33 patients, with 3–5 visits each; multiple correlated features were screened, so isolated low p-values are not confirmation.
- Duplicate reanalyses are available for only 11 visits in four patients. Repeatability estimates are provisional and may not generalize.
- No diagnoses, treatments, loading conditions, image-quality scores, or clinical events were supplied; EF/GLS change is a surrogate endpoint, not proof of clinical deterioration.
- GLS is negative in the source. The analysis uses positive magnitude so lower values are worse; the signed source values are preserved.
- Curve derivatives are sampling-sensitive. Peaks use original time; shape/roughness comparisons use normalized resampling.
- PCA coordinates are descriptive and were not used as a multivariable patient classifier.

## Files to use

- `Ichilov_july_dataset.parquet`: raw curve-series dataset, all analysis instances.
- `Ichilov_july_visits.parquet`: one row per true visit after replicate averaging.
- `patient_longitudinal_summary.csv`: patient-level endpoints and first/final features.
- `baseline_ancova_results.csv`: main early-warning screen.
- `technical_repeatability.csv`: internal reanalysis QC.
- `spatial_persistence_pairs.csv`: anatomical-pattern stability between visits.