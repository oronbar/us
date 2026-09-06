# Round 4: dedicated time-series classifiers

## Locked protocol

Predict whether the immediately following visit is the first visit with at least
15% relative Mid-GLS deterioration from first-visit baseline. The evaluation has
238 transitions, 49 events, and 103 patients. All results use the same three
repeated five-fold patient-held-out splits and 2,000 patient-cluster bootstraps.
Randomized classifiers are averages of three fixed seeds.

## Results

| Model | AUC | AUC CI low | AUC CI high | AP | AP CI low | AP CI high |
|---|---|---|---|---|---|---|
| Equal CNN + MOMENT + catch22 xgb curves scalars | 0.698 | 0.622 | 0.769 | 0.364 | 0.264 | 0.493 |
| Equal CNN + MOMENT + drcif curves scalar blend | 0.699 | 0.616 | 0.775 | 0.363 | 0.260 | 0.502 |
| Equal CNN + MOMENT + rdst shapelet curves scalars | 0.706 | 0.630 | 0.779 | 0.362 | 0.262 | 0.498 |
| Equal CNN + MOMENT + TimeMIL | 0.703 | 0.624 | 0.780 | 0.355 | 0.252 | 0.498 |
| Equal CNN + MOMENT | 0.698 | 0.620 | 0.773 | 0.353 | 0.256 | 0.498 |
| Equal CNN + MOMENT + convtran small curves scalars | 0.691 | 0.608 | 0.765 | 0.346 | 0.250 | 0.483 |
| Equal CNN + MOMENT + inception segment curves scalars | 0.688 | 0.608 | 0.764 | 0.342 | 0.251 | 0.484 |
| Current CNN | 0.683 | 0.597 | 0.759 | 0.339 | 0.248 | 0.465 |
| Equal CNN + MOMENT + inception whole curves scalars | 0.679 | 0.597 | 0.757 | 0.339 | 0.241 | 0.469 |
| TimeMIL attention + scalars | 0.678 | 0.598 | 0.763 | 0.337 | 0.240 | 0.468 |
| MOMENT-small + scalars | 0.678 | 0.596 | 0.758 | 0.335 | 0.241 | 0.475 |
| DrCIF + fixed scalar blend | 0.683 | 0.600 | 0.763 | 0.330 | 0.240 | 0.466 |
| Catch22-XGBoost: curves + scalars | 0.646 | 0.563 | 0.724 | 0.329 | 0.238 | 0.464 |
| RDST shapelets: curves + scalars | 0.661 | 0.582 | 0.738 | 0.321 | 0.235 | 0.458 |
| RDST shapelets: curves | 0.645 | 0.563 | 0.717 | 0.294 | 0.216 | 0.416 |
| Catch22-XGBoost: curves | 0.644 | 0.567 | 0.718 | 0.292 | 0.216 | 0.412 |
| Clinical ridge | 0.631 | 0.532 | 0.730 | 0.289 | 0.206 | 0.440 |
| Scalar-only MLP | 0.669 | 0.584 | 0.748 | 0.286 | 0.217 | 0.407 |
| DrCIF: curves | 0.610 | 0.524 | 0.692 | 0.285 | 0.202 | 0.420 |
| InceptionTime segment-structured: curves | 0.607 | 0.513 | 0.696 | 0.267 | 0.197 | 0.387 |
| InceptionTime segment-structured: curves + scalars | 0.603 | 0.517 | 0.686 | 0.251 | 0.190 | 0.366 |
| ConvTran-small: curves | 0.515 | 0.418 | 0.612 | 0.214 | 0.152 | 0.316 |
| InceptionTime whole-heart: curves + scalars | 0.489 | 0.402 | 0.574 | 0.195 | 0.146 | 0.281 |
| ConvTran-small: curves + scalars | 0.480 | 0.391 | 0.572 | 0.194 | 0.143 | 0.283 |
| InceptionTime whole-heart: curves | 0.481 | 0.390 | 0.573 | 0.192 | 0.144 | 0.277 |

Random guessing has AUC 0.500 and expected AP 0.206. The best new classifier was
`catch22_xgb_curves_scalars`. The best prespecified Mantis-free ensemble was `ensemble_cnn_moment_catch22_xgb_curves_scalars`.

## Paired changes from current CNN

| Model | delta AUC | delta AUC CI low | delta AUC CI high | delta AP | delta AP CI low | delta AP CI high |
|---|---|---|---|---|---|---|
| InceptionTime whole-heart: curves | -0.201 | -0.302 | -0.093 | -0.147 | -0.244 | -0.059 |
| InceptionTime whole-heart: curves + scalars | -0.194 | -0.287 | -0.102 | -0.145 | -0.240 | -0.051 |
| InceptionTime segment-structured: curves | -0.076 | -0.175 | 0.026 | -0.072 | -0.166 | 0.035 |
| InceptionTime segment-structured: curves + scalars | -0.079 | -0.169 | 0.008 | -0.089 | -0.176 | 0.009 |
| ConvTran-small: curves | -0.167 | -0.265 | -0.063 | -0.126 | -0.217 | -0.035 |
| ConvTran-small: curves + scalars | -0.202 | -0.307 | -0.096 | -0.145 | -0.241 | -0.053 |
| DrCIF: curves | -0.073 | -0.174 | 0.030 | -0.054 | -0.160 | 0.069 |
| RDST shapelets: curves | -0.038 | -0.143 | 0.062 | -0.045 | -0.153 | 0.076 |
| RDST shapelets: curves + scalars | -0.021 | -0.119 | 0.079 | -0.018 | -0.121 | 0.112 |
| Catch22-XGBoost: curves | -0.039 | -0.145 | 0.067 | -0.047 | -0.154 | 0.070 |
| Catch22-XGBoost: curves + scalars | -0.037 | -0.131 | 0.056 | -0.011 | -0.121 | 0.127 |
| DrCIF + fixed scalar blend | 0.001 | -0.057 | 0.060 | -0.009 | -0.089 | 0.078 |
| Equal CNN + MOMENT | 0.015 | -0.054 | 0.083 | 0.014 | -0.078 | 0.128 |
| Equal CNN + MOMENT + TimeMIL | 0.020 | -0.042 | 0.082 | 0.016 | -0.072 | 0.107 |
| Equal CNN + MOMENT + inception whole curves scalars | -0.004 | -0.074 | 0.061 | -0.000 | -0.095 | 0.102 |
| Equal CNN + MOMENT + inception segment curves scalars | 0.006 | -0.066 | 0.068 | 0.003 | -0.086 | 0.102 |
| Equal CNN + MOMENT + convtran small curves scalars | 0.009 | -0.062 | 0.077 | 0.007 | -0.086 | 0.114 |
| Equal CNN + MOMENT + drcif curves scalar blend | 0.017 | -0.051 | 0.086 | 0.023 | -0.069 | 0.126 |
| Equal CNN + MOMENT + rdst shapelet curves scalars | 0.024 | -0.053 | 0.100 | 0.022 | -0.085 | 0.156 |
| Equal CNN + MOMENT + catch22 xgb curves scalars | 0.015 | -0.055 | 0.086 | 0.025 | -0.069 | 0.131 |

## Key comparisons

Positive values favor the first named candidate in each comparison.

| Comparison | delta AUC | delta AUC CI low | delta AUC CI high | delta AP | delta AP CI low | delta AP CI high |
|---|---|---|---|---|---|---|
| Best Mantis-free ensemble vs clinical ridge | 0.067 | -0.021 | 0.151 | 0.075 | -0.053 | 0.177 |
| Add Catch22 to CNN + MOMENT | 0.000 | -0.026 | 0.029 | 0.011 | -0.046 | 0.056 |
| Best Mantis-free vs Round 3 Mantis ensemble | -0.022 | -0.076 | 0.028 | -0.002 | -0.077 | 0.074 |
| Shapelet ensemble vs Catch22 ensemble | 0.008 | -0.030 | 0.047 | -0.002 | -0.056 | 0.072 |

## Seed stability

| Model | AUC mean [range] | AP mean [range] |
|---|---|---|
| Catch22-XGBoost: curves | 0.644 [0.640-0.646] | 0.297 [0.292-0.306] |
| Catch22-XGBoost: curves + scalars | 0.647 [0.645-0.649] | 0.327 [0.319-0.335] |
| ConvTran-small: curves | 0.498 [0.492-0.507] | 0.216 [0.209-0.219] |
| ConvTran-small: curves + scalars | 0.500 [0.479-0.537] | 0.206 [0.196-0.219] |
| DrCIF: curves | 0.603 [0.582-0.631] | 0.284 [0.266-0.299] |
| InceptionTime segment-structured: curves | 0.594 [0.576-0.608] | 0.269 [0.239-0.285] |
| InceptionTime segment-structured: curves + scalars | 0.601 [0.582-0.615] | 0.256 [0.233-0.278] |
| InceptionTime whole-heart: curves | 0.487 [0.440-0.541] | 0.197 [0.177-0.218] |
| InceptionTime whole-heart: curves + scalars | 0.495 [0.482-0.518] | 0.205 [0.193-0.228] |
| RDST shapelets: curves | 0.637 [0.626-0.649] | 0.292 [0.278-0.306] |
| RDST shapelets: curves + scalars | 0.655 [0.641-0.668] | 0.313 [0.299-0.330] |

## Model definitions

- InceptionTime whole-heart processes all 108 segment-channel curves jointly with
  six multiscale inception blocks using temporal kernels 9, 19, and 39.
- Segment-structured InceptionTime applies the same six-channel encoder to every
  segment, then pools segment mean, standard deviation, and maximum.
- ConvTran-small uses a 108-to-64 temporal convolution followed by two four-head
  transformer layers and a classification token.
- DrCIF averages three 60-tree multivariate diverse-representation interval
  forests (180 trees total). Its
  scalar result is a fixed equal probability blend with the scalar-only MLP.
- RDST uses 1,200 multivariate dilated shapelets and balanced logistic regression.
- Catch22-XGBoost summarizes 22 curve characteristics across segments using mean,
  standard deviation, minimum, maximum, and median, then fits regularized shallow
  boosted trees.
