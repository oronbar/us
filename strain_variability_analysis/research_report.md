# Strain Segment Variability Analysis

## Scope
- XML files discovered: 140
- XML files parsed: 140
- XML files staged locally because of OneDrive read errors: 25
- XML parse failures: 0
- Segment curves extracted: 1680
- Visit/layer rows: 136
- Latent method: torch_mlp_autoencoder

Only `Strain-Endo` and `Strain-Myo` sheets were used. Within those sheets, only the `Longitudinal Strain` block was parsed; transverse strain blocks were ignored.

## Segment Features
- Classic feature: `time_to_peak_ms` and `time_to_peak_norm`, computed at the most negative longitudinal strain value.
- Additional engineered features: `peak_abs`, normalized negative-strain area (`strain_burden`), maximum contraction/relaxation slopes, recovery fraction, RMS strain, and curve roughness.
- Visit-level variability was computed after averaging duplicate segment labels within the same patient/visit/layer.

## Classic Variability Summary
| hospital | layer | metric | patients_tested | positive_slopes | negative_slopes | zero_slopes | fraction_positive | binomial_p_positive_gt_half | median_slope_per_visit | mean_slope_per_visit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ichilov | endo | contraction_rate_per_s_std | 3 | 0 | 3 | 0 | 0 | 1 | -3.492 | -2.54 |
| Ichilov | endo | peak_abs_std | 3 | 1 | 2 | 0 | 0.3333 | 0.875 | -0.3068 | -0.3201 |
| Ichilov | endo | strain_burden_std | 3 | 1 | 2 | 0 | 0.3333 | 0.875 | -0.0253 | -0.02706 |
| Ichilov | endo | time_to_peak_norm_std | 3 | 1 | 2 | 0 | 0.3333 | 0.875 | -0.007137 | -0.001084 |
| Ichilov | myo | contraction_rate_per_s_std | 3 | 1 | 2 | 0 | 0.3333 | 0.875 | -1.294 | -1.07 |
| Ichilov | myo | peak_abs_std | 3 | 0 | 3 | 0 | 0 | 1 | -0.06024 | -0.07791 |
| Ichilov | myo | strain_burden_std | 3 | 1 | 2 | 0 | 0.3333 | 0.875 | -0.08396 | -0.03807 |
| Ichilov | myo | time_to_peak_norm_std | 3 | 1 | 2 | 0 | 0.3333 | 0.875 | -0.01257 | -0.004588 |
| SZMC | endo | contraction_rate_per_s_std | 12 | 9 | 3 | 0 | 0.75 | 0.073 | 5.384 | 3.016 |
| SZMC | endo | peak_abs_std | 12 | 9 | 3 | 0 | 0.75 | 0.073 | 0.4393 | 0.434 |
| SZMC | endo | strain_burden_std | 12 | 9 | 3 | 0 | 0.75 | 0.073 | 0.34 | 0.1657 |
| SZMC | endo | time_to_peak_norm_std | 12 | 3 | 9 | 0 | 0.25 | 0.9807 | -0.01309 | -0.02464 |
| SZMC | myo | contraction_rate_per_s_std | 12 | 9 | 3 | 0 | 0.75 | 0.073 | 1.561 | 1.846 |
| SZMC | myo | peak_abs_std | 12 | 8 | 4 | 0 | 0.6667 | 0.1938 | 0.4492 | 0.3738 |
| SZMC | myo | strain_burden_std | 12 | 8 | 4 | 0 | 0.6667 | 0.1938 | 0.2052 | 0.1367 |
| SZMC | myo | time_to_peak_norm_std | 12 | 3 | 9 | 0 | 0.25 | 0.9807 | -0.01122 | -0.02888 |

## Latent Variability Summary
| hospital | layer | metric | patients_tested | positive_slopes | negative_slopes | zero_slopes | fraction_positive | binomial_p_positive_gt_half | median_slope_per_visit | mean_slope_per_visit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ichilov | endo | latent_centroid_norm | 3 | 1 | 2 | 0 | 0.3333 | 0.875 | -1.99 | -0.8261 |
| Ichilov | endo | latent_pairwise_mean | 3 | 2 | 1 | 0 | 0.6667 | 0.5 | 0.2672 | 0.1162 |
| Ichilov | myo | latent_centroid_norm | 3 | 1 | 2 | 0 | 0.3333 | 0.875 | -0.9947 | -0.6618 |
| Ichilov | myo | latent_pairwise_mean | 3 | 1 | 2 | 0 | 0.3333 | 0.875 | -0.09437 | -0.1528 |
| SZMC | endo | latent_centroid_norm | 12 | 5 | 7 | 0 | 0.4167 | 0.8062 | -2.015 | -0.7277 |
| SZMC | endo | latent_pairwise_mean | 12 | 7 | 5 | 0 | 0.5833 | 0.3872 | 0.2914 | -0.008699 |
| SZMC | myo | latent_centroid_norm | 12 | 5 | 7 | 0 | 0.4167 | 0.8062 | -0.9699 | -0.5511 |
| SZMC | myo | latent_pairwise_mean | 12 | 7 | 5 | 0 | 0.5833 | 0.3872 | 0.4457 | -0.06609 |

## Interpretation Notes
- A positive slope means segment variability increased over later visits for that patient and layer.
- `peak_abs_std` tests amplitude heterogeneity between segments.
- `time_to_peak_norm_std` tests temporal dyssynchrony between segments.
- `strain_burden_std` tests heterogeneity in the integrated negative strain load.
- `latent_pairwise_mean` tests shape-level heterogeneity in autoencoder latent space.
- Treat p-values as exploratory: the sample is small and there is no adjustment for repeated metrics/layers.