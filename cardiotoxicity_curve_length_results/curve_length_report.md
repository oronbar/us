# Native strain-curve sample-length analysis

## Cohort used by the CNN

This analysis includes only the longitudinal Endo and Mid segment curves used to
construct the CNN tensor: 14,976 curves from
416 reports, 400 true visits,
and 416 analysis instances. Each report contributes 36
curves (18 segments × 2 layers), and all 36 curves within a report have exactly the
same native sample count.

## Native sample count

| reports | curves | mean_points | sd_points | min_points | p05_points | p25_points | median_points | p75_points | p90_points | p95_points | p99_points | max_points |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 416.000 | 14976.000 | 58.115 | 19.474 | 18.000 | 30.000 | 46.000 | 57.000 | 67.000 | 81.000 | 87.000 | 133.000 | 156.000 |

Only 7 of 416 reports have exactly 65
samples. The median is 57 and the middle 50% range is 46–67.

## Candidate fixed lengths

Reconstruction error was calculated by resampling the original curve to each target
length and interpolating it back to the original native time points. Strain errors are
percentage points.

| target_length | upsampled_% | downsampled_% | input_%_of_96 | median_reconstruction_rmse_pp | p95_reconstruction_rmse_pp | median_peak_error_pp | p95_peak_error_pp | median_ttp_error_ms | p95_ttp_error_ms |
|---|---|---|---|---|---|---|---|---|---|
| 48.000 | 26.683 | 70.433 | 50.000 | 0.090 | 0.183 | 0.037 | 0.219 | 4.645 | 12.686 |
| 64.000 | 67.067 | 30.769 | 66.667 | 0.065 | 0.135 | 0.026 | 0.156 | 3.434 | 10.026 |
| 72.000 | 82.212 | 17.067 | 75.000 | 0.059 | 0.122 | 0.024 | 0.147 | 3.402 | 8.909 |
| 80.000 | 88.702 | 10.337 | 83.333 | 0.053 | 0.109 | 0.022 | 0.132 | 3.178 | 8.241 |
| 96.000 | 96.635 | 3.125 | 100.000 | 0.045 | 0.091 | 0.019 | 0.109 | 2.639 | 6.656 |

## Recommendation

- **96 was a conservative engineering choice, not a data-derived optimum.** It
  upsamples 96.6% of reports and therefore does not create new temporal information.
- **64 points is the best first alternative to test.** It is close to the native
  distribution, reduces curve-branch input and convolutional work by 33%, and its
  interpolation error is small relative to the observed strain-measurement noise:
  median RMSE 0.065 percentage points and 95th percentile RMSE 0.135.
- **72 points is a conservative compromise.** It covers 82.9% of reports without
  downsampling and uses 75% of the 96-point input size.
- Reconstruction error naturally decreases on denser interpolation grids, even above
  the native sample count. This means the piecewise-linear approximation is more
  accurate; it does **not** mean that 96 or 128 points contain new physiological
  information that was absent from the original report.
- Do not choose 65 merely because one inspected report had 65 points; only seven
  reports have that exact length. A fixed normalized grid need not match a particular
  report exactly.
- The final choice should be made with a controlled CNN ablation at 48, 64, 72, and
  96 points using identical patient folds. When changing length, the temporal kernel
  widths should also be considered because a 7-sample kernel covers 7.3% of a
  96-point cycle but 10.9% of a 64-point cycle.
