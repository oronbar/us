# Strain Segment Variability Analysis

## Scope
- XML files discovered: 140
- XML files parsed: 132
- XML files parsed but excluded by view filter: 8
- XML files staged locally because of OneDrive read errors: 0
- XML parse failures: 0
- Segment curves extracted: 1584
- Visit/layer rows: 256
- Latent method: torch_mlp_autoencoder

Only `Strain-Endo` and `Strain-Myo` sheets were used. Within those sheets, only the `Longitudinal Strain` block was parsed; transverse strain blocks were ignored.

Visit-level variability is computed within `patient + visit + view + layer`. This means `2-chamber_endo`, `2-chamber_myo`, `4-chamber_endo`, and `4-chamber_myo` are kept separate.

## Segment Features
- Classic feature: `time_to_peak_ms` and `time_to_peak_norm`, computed at the most negative longitudinal strain value.
- Additional engineered features: `peak_abs`, normalized negative-strain area (`strain_burden`), maximum contraction/relaxation slopes, recovery fraction, RMS strain, and curve roughness.
- Visit-level variability was computed after averaging duplicate segment labels within the same patient/visit/view/layer.

## Classic Variability Summary
| hospital | view | layer | metric | patients_tested | positive_slopes | negative_slopes | zero_slopes | fraction_positive | binomial_p_positive_gt_half | median_slope_per_visit | mean_slope_per_visit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ichilov | 2-chamber | endo | contraction_rate_per_s_std | 3 | 0 | 3 | 0 | 0 | 1 | -6.38 | -6.572 |
| Ichilov | 2-chamber | endo | peak_abs_std | 3 | 0 | 3 | 0 | 0 | 1 | -0.259 | -0.4856 |
| Ichilov | 2-chamber | endo | strain_burden_std | 3 | 2 | 1 | 0 | 0.6667 | 0.5 | 0.1641 | -0.06096 |
| Ichilov | 2-chamber | endo | time_to_peak_norm_std | 3 | 2 | 1 | 0 | 0.6667 | 0.5 | 0.000118 | -0.003433 |
| Ichilov | 2-chamber | myo | contraction_rate_per_s_std | 3 | 0 | 3 | 0 | 0 | 1 | -3.37 | -4.896 |
| Ichilov | 2-chamber | myo | peak_abs_std | 3 | 0 | 3 | 0 | 0 | 1 | -0.4476 | -0.496 |
| Ichilov | 2-chamber | myo | strain_burden_std | 3 | 1 | 2 | 0 | 0.3333 | 0.875 | -0.1722 | -0.265 |
| Ichilov | 2-chamber | myo | time_to_peak_norm_std | 3 | 0 | 3 | 0 | 0 | 1 | -0.0135 | -0.01602 |
| Ichilov | 4-chamber | endo | contraction_rate_per_s_std | 3 | 3 | 0 | 0 | 1 | 0.125 | 1.848 | 3.989 |
| Ichilov | 4-chamber | endo | peak_abs_std | 3 | 2 | 1 | 0 | 0.6667 | 0.5 | 0.2591 | 0.3075 |
| Ichilov | 4-chamber | endo | strain_burden_std | 3 | 3 | 0 | 0 | 1 | 0.125 | 0.2363 | 0.3065 |
| Ichilov | 4-chamber | endo | time_to_peak_norm_std | 3 | 0 | 3 | 0 | 0 | 1 | -0.01996 | -0.0165 |
| Ichilov | 4-chamber | myo | contraction_rate_per_s_std | 3 | 3 | 0 | 0 | 1 | 0.125 | 5.195 | 4.78 |
| Ichilov | 4-chamber | myo | peak_abs_std | 3 | 3 | 0 | 0 | 1 | 0.125 | 0.5051 | 0.6841 |
| Ichilov | 4-chamber | myo | strain_burden_std | 3 | 3 | 0 | 0 | 1 | 0.125 | 0.3909 | 0.3491 |
| Ichilov | 4-chamber | myo | time_to_peak_norm_std | 3 | 1 | 2 | 0 | 0.3333 | 0.875 | -0.06036 | -0.04812 |
| SZMC | 2-chamber | endo | contraction_rate_per_s_std | 10 | 5 | 5 | 0 | 0.5 | 0.623 | 0.3916 | 1.283 |
| SZMC | 2-chamber | endo | peak_abs_std | 10 | 6 | 4 | 0 | 0.6 | 0.377 | 0.3311 | 0.2807 |
| SZMC | 2-chamber | endo | strain_burden_std | 10 | 7 | 3 | 0 | 0.7 | 0.1719 | 0.3675 | 0.1376 |
| SZMC | 2-chamber | endo | time_to_peak_norm_std | 10 | 2 | 8 | 0 | 0.2 | 0.9893 | -0.01131 | -0.01706 |
| SZMC | 2-chamber | myo | contraction_rate_per_s_std | 10 | 7 | 3 | 0 | 0.7 | 0.1719 | 2.626 | 3.226 |
| SZMC | 2-chamber | myo | peak_abs_std | 10 | 6 | 4 | 0 | 0.6 | 0.377 | 0.4426 | 0.4137 |
| SZMC | 2-chamber | myo | strain_burden_std | 10 | 7 | 3 | 0 | 0.7 | 0.1719 | 0.2614 | 0.1921 |
| SZMC | 2-chamber | myo | time_to_peak_norm_std | 10 | 4 | 6 | 0 | 0.4 | 0.8281 | -0.009802 | 0.009175 |
| SZMC | 4-chamber | endo | contraction_rate_per_s_std | 12 | 9 | 3 | 0 | 0.75 | 0.073 | 6.606 | 6.669 |
| SZMC | 4-chamber | endo | peak_abs_std | 12 | 7 | 5 | 0 | 0.5833 | 0.3872 | 0.5666 | 0.6782 |
| SZMC | 4-chamber | endo | strain_burden_std | 12 | 7 | 5 | 0 | 0.5833 | 0.3872 | 0.198 | 0.365 |
| SZMC | 4-chamber | endo | time_to_peak_norm_std | 12 | 3 | 9 | 0 | 0.25 | 0.9807 | -0.05887 | -0.04423 |
| SZMC | 4-chamber | myo | contraction_rate_per_s_std | 12 | 6 | 6 | 0 | 0.5 | 0.6128 | -0.2973 | 1.754 |
| SZMC | 4-chamber | myo | peak_abs_std | 12 | 10 | 2 | 0 | 0.8333 | 0.01929 | 0.3464 | 0.5669 |
| SZMC | 4-chamber | myo | strain_burden_std | 12 | 8 | 4 | 0 | 0.6667 | 0.1938 | 0.1923 | 0.2567 |
| SZMC | 4-chamber | myo | time_to_peak_norm_std | 12 | 4 | 8 | 0 | 0.3333 | 0.927 | -0.02956 | -0.04625 |

## Latent Variability Summary
| hospital | view | layer | metric | patients_tested | positive_slopes | negative_slopes | zero_slopes | fraction_positive | binomial_p_positive_gt_half | median_slope_per_visit | mean_slope_per_visit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ichilov | 2-chamber | endo | latent_centroid_norm | 3 | 1 | 2 | 0 | 0.3333 | 0.875 | -2.371 | -1.075 |
| Ichilov | 2-chamber | endo | latent_pairwise_mean | 3 | 1 | 2 | 0 | 0.3333 | 0.875 | -0.176 | -0.5962 |
| Ichilov | 2-chamber | myo | latent_centroid_norm | 3 | 1 | 2 | 0 | 0.3333 | 0.875 | -1.185 | -0.8286 |
| Ichilov | 2-chamber | myo | latent_pairwise_mean | 3 | 0 | 3 | 0 | 0 | 1 | -0.6557 | -1.014 |
| Ichilov | 4-chamber | endo | latent_centroid_norm | 3 | 2 | 1 | 0 | 0.6667 | 0.5 | 0.4748 | -0.3482 |
| Ichilov | 4-chamber | endo | latent_pairwise_mean | 3 | 2 | 1 | 0 | 0.6667 | 0.5 | 1.132 | 0.6186 |
| Ichilov | 4-chamber | myo | latent_centroid_norm | 3 | 1 | 2 | 0 | 0.3333 | 0.875 | -0.2027 | -0.6418 |
| Ichilov | 4-chamber | myo | latent_pairwise_mean | 3 | 3 | 0 | 0 | 1 | 0.125 | 0.2348 | 0.3918 |
| SZMC | 2-chamber | endo | latent_centroid_norm | 10 | 4 | 6 | 0 | 0.4 | 0.8281 | -0.8119 | 0.2294 |
| SZMC | 2-chamber | endo | latent_pairwise_mean | 10 | 4 | 6 | 0 | 0.4 | 0.8281 | -0.4285 | -0.3956 |
| SZMC | 2-chamber | myo | latent_centroid_norm | 10 | 4 | 6 | 0 | 0.4 | 0.8281 | -0.2607 | 0.1387 |
| SZMC | 2-chamber | myo | latent_pairwise_mean | 10 | 5 | 5 | 0 | 0.5 | 0.623 | 0.1674 | -0.1622 |
| SZMC | 4-chamber | endo | latent_centroid_norm | 12 | 4 | 8 | 0 | 0.3333 | 0.927 | -0.458 | -1.272 |
| SZMC | 4-chamber | endo | latent_pairwise_mean | 12 | 5 | 7 | 0 | 0.4167 | 0.8062 | -0.6939 | 0.5167 |
| SZMC | 4-chamber | myo | latent_centroid_norm | 12 | 5 | 7 | 0 | 0.4167 | 0.8062 | -0.7796 | -1.311 |
| SZMC | 4-chamber | myo | latent_pairwise_mean | 12 | 6 | 6 | 0 | 0.5 | 0.6128 | -0.06899 | 0.3102 |

## Interpretation Notes
- A positive slope means segment variability increased over later visits for that patient, view, and layer.
- `peak_abs_std` tests amplitude heterogeneity between segments.
- `time_to_peak_norm_std` tests temporal dyssynchrony between segments.
- `strain_burden_std` tests heterogeneity in the integrated negative strain load.
- `latent_pairwise_mean` tests shape-level heterogeneity in autoencoder latent space.
- Treat p-values as exploratory: the sample is small and there is no adjustment for repeated metrics/layers.