# Feature importance of the two best Round-4 ensembles

Importance was estimated strictly from patient-held-out predictions. Individual inputs were shuffled in the held-out fold; the reported decrease is performance lost after shuffling. Model-level contributions use both leave-one-model-out ablation and exact three-player Shapley values. Confidence intervals are patient-cluster bootstraps.

## Model-level contribution

| ensemble | component | full_auc | full_ap | leave_one_out_auc_drop | leave_one_out_ap_drop | shapley_auc | shapley_ap |
|---|---|---|---|---|---|---|---|
| CNN + MOMENT + RDST | current_cnn | 0.706 | 0.362 | 0.013 | 0.013 | 0.074 | 0.060 |
| CNN + MOMENT + RDST | moment_small_frozen_curves_scalars | 0.706 | 0.362 | 0.011 | -0.009 | 0.071 | 0.047 |
| CNN + MOMENT + RDST | rdst_shapelet_curves_scalars | 0.706 | 0.362 | 0.009 | 0.009 | 0.061 | 0.049 |
| CNN + MOMENT + Catch22 | current_cnn | 0.698 | 0.364 | 0.014 | 0.023 | 0.074 | 0.058 |
| CNN + MOMENT + Catch22 | moment_small_frozen_curves_scalars | 0.698 | 0.364 | 0.019 | 0.019 | 0.074 | 0.053 |
| CNN + MOMENT + Catch22 | catch22_xgb_curves_scalars | 0.698 | 0.364 | 0.000 | 0.011 | 0.049 | 0.047 |

## CNN + MOMENT + RDST: most valuable inputs

| display_item | importance_scope | auc_drop | auc_ci_low | auc_ci_high | ap_drop | ap_ci_low | ap_ci_high |
|---|---|---|---|---|---|---|---|
| moment::embedding_pool::segment_max | one_component | 0.039 | -0.000 | 0.074 | 0.058 | 0.011 | 0.124 |
| rdst::shapelet_output::best_match_location | one_component | 0.030 | 0.001 | 0.058 | 0.030 | -0.011 | 0.078 |
| moment::curve_channel::change_endo_minus_mid | one_component | 0.020 | -0.021 | 0.058 | 0.035 | -0.013 | 0.096 |
| moment::curve_all | one_component | 0.018 | -0.012 | 0.048 | 0.040 | -0.011 | 0.106 |
| rdst::shapelet_all | one_component | 0.018 | -0.017 | 0.051 | 0.029 | -0.031 | 0.091 |
| cnn::scalar_all | one_component | 0.013 | -0.001 | 0.028 | 0.019 | 0.003 | 0.040 |
| moment::curve_channel::current_endo | one_component | 0.008 | -0.008 | 0.024 | 0.021 | -0.013 | 0.073 |
| rdst::scalar_all | one_component | 0.006 | -0.006 | 0.018 | 0.001 | -0.025 | 0.034 |
| joint_scalar::first_mid_gls | joint_across_components | 0.006 | -0.002 | 0.012 | 0.013 | 0.002 | 0.031 |
| joint_scalar::cur_var__mid_peak_abs_robust_sd | joint_across_components | 0.005 | -0.003 | 0.013 | -0.004 | -0.027 | 0.022 |
| moment::curve_channel::change_endo | one_component | 0.005 | -0.024 | 0.036 | -0.002 | -0.047 | 0.067 |
| joint_scalar::cur_var__endo_vendor_peak_systolic_abs_robust_sd | joint_across_components | 0.005 | -0.002 | 0.012 | -0.002 | -0.021 | 0.022 |
| joint_scalar::first_endo_gls | joint_across_components | 0.005 | -0.002 | 0.013 | 0.011 | 0.001 | 0.031 |
| joint_scalar::cur_var__mid_vendor_peak_systolic_abs_robust_sd | joint_across_components | 0.005 | -0.003 | 0.012 | -0.004 | -0.022 | 0.020 |
| moment::scalar::first_mid_gls | one_component | 0.004 | -0.003 | 0.010 | 0.009 | 0.001 | 0.024 |

## CNN + MOMENT + Catch22: most valuable inputs

| display_item | importance_scope | auc_drop | auc_ci_low | auc_ci_high | ap_drop | ap_ci_low | ap_ci_high |
|---|---|---|---|---|---|---|---|
| moment::embedding_pool::segment_max | one_component | 0.042 | 0.002 | 0.082 | 0.057 | 0.014 | 0.100 |
| moment::curve_channel::change_endo_minus_mid | one_component | 0.022 | -0.022 | 0.069 | 0.033 | -0.022 | 0.083 |
| moment::curve_all | one_component | 0.021 | -0.010 | 0.052 | 0.043 | -0.004 | 0.089 |
| cnn::scalar_all | one_component | 0.015 | 0.001 | 0.029 | 0.022 | -0.005 | 0.042 |
| catch22::feature::c1_min_DN_OutlierInclude_p_001_mdrmd | one_component | 0.015 | 0.002 | 0.027 | 0.026 | 0.005 | 0.049 |
| catch22::segment_aggregation::min | one_component | 0.013 | -0.004 | 0.030 | 0.021 | -0.008 | 0.057 |
| moment::scalar_all | one_component | 0.013 | -0.018 | 0.046 | 0.043 | -0.002 | 0.087 |
| catch22::catch22_descriptor::DN_OutlierInclude_p_001_mdrmd | one_component | 0.012 | -0.004 | 0.028 | 0.030 | 0.003 | 0.066 |
| moment::curve_channel::current_endo | one_component | 0.011 | -0.004 | 0.026 | 0.033 | 0.005 | 0.061 |
| moment::curve_channel::change_endo | one_component | 0.009 | -0.021 | 0.042 | 0.004 | -0.058 | 0.059 |
| moment::curve_channel::current_mid | one_component | 0.009 | -0.006 | 0.024 | 0.025 | -0.002 | 0.043 |
| catch22::curve_all | one_component | 0.008 | -0.020 | 0.035 | 0.018 | -0.045 | 0.062 |
| moment::scalar::first_mid_gls | one_component | 0.007 | -0.001 | 0.015 | 0.020 | 0.002 | 0.035 |
| moment::scalar::first_endo_gls | one_component | 0.007 | -0.000 | 0.013 | 0.021 | 0.003 | 0.036 |
| catch22::catch22_descriptor::CO_f1ecac | one_component | 0.007 | -0.002 | 0.015 | 0.020 | -0.000 | 0.035 |

## Interpretation limits

- Permutation importance measures reliance of this fitted model, not causality.
- Correlated inputs share or mask importance; a low individual value does not prove that the physiological variable is irrelevant.
- RDST shapelets are whole-heart multivariate motifs. Their segment/channel attribution is coefficient-weighted shapelet energy, a descriptive native attribution rather than a held-out permutation result.
- Deep representations do not expose named coefficients. CNN importance is grouped raw-input permutation; MOMENT importance is permutation of frozen embedding groups before fold-specific PCA.