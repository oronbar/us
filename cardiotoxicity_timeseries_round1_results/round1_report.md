# Round 1: alternatives to the current strain-curve CNN

## Task and protocol

Predict whether the immediately following visit will be the first visit with at
least 15% relative Mid-GLS deterioration from the first-visit baseline. The
evaluation contains 238 eligible transitions, 49 events (20.6%), and 103
patients. All models use the exact same three repeated five-fold patient-held-out
splits as the retained CNN. Confidence intervals are 2,000 patient-cluster
bootstrap samples.

## Results

| Model | AUC | AUC CI low | AUC CI high | AP | AP CI low | AP CI high |
|---|---|---|---|---|---|---|
| Current CNN | 0.683 | 0.599 | 0.762 | 0.339 | 0.241 | 0.463 |
| Random Mantis: curves + scalars | 0.702 | 0.623 | 0.781 | 0.337 | 0.251 | 0.474 |
| MOMENT-small frozen: curves + scalars | 0.678 | 0.597 | 0.758 | 0.335 | 0.249 | 0.466 |
| Random Mantis: curves | 0.669 | 0.584 | 0.751 | 0.321 | 0.230 | 0.453 |
| MOMENT-small frozen: curves | 0.664 | 0.584 | 0.740 | 0.312 | 0.231 | 0.442 |
| Clinical ridge | 0.631 | 0.535 | 0.724 | 0.289 | 0.208 | 0.426 |
| MultiROCKET segment: curves + scalars | 0.598 | 0.521 | 0.674 | 0.281 | 0.207 | 0.392 |
| MultiROCKET segment: curves | 0.597 | 0.515 | 0.684 | 0.281 | 0.203 | 0.393 |
| MantisV2 frozen: curves + scalars | 0.634 | 0.549 | 0.717 | 0.269 | 0.200 | 0.398 |
| MantisV2 frozen: curves | 0.621 | 0.535 | 0.703 | 0.269 | 0.201 | 0.381 |
| MultiROCKET whole: curves | 0.523 | 0.428 | 0.614 | 0.261 | 0.176 | 0.381 |
| MultiROCKET whole: curves + scalars | 0.527 | 0.431 | 0.619 | 0.258 | 0.176 | 0.383 |
| MantisV2 adapter: curves + scalars | 0.617 | 0.528 | 0.705 | 0.254 | 0.188 | 0.364 |
| MantisV2 adapter: curves | 0.550 | 0.462 | 0.640 | 0.230 | 0.169 | 0.333 |

Random guessing has AUC 0.500 and expected AP 0.206. The highest AP was
**Current CNN**, with AUC 0.683 and AP 0.339.

## Paired change from the current CNN

| Model | delta AUC | delta AUC CI low | delta AUC CI high | delta AP | delta AP CI low | delta AP CI high |
|---|---|---|---|---|---|---|
| Random Mantis: curves + scalars | 0.020 | -0.062 | 0.096 | -0.003 | -0.106 | 0.112 |
| MOMENT-small frozen: curves + scalars | -0.004 | -0.085 | 0.079 | -0.005 | -0.102 | 0.125 |
| Random Mantis: curves | -0.014 | -0.119 | 0.079 | -0.018 | -0.130 | 0.109 |
| MOMENT-small frozen: curves | -0.019 | -0.119 | 0.081 | -0.028 | -0.132 | 0.095 |
| MultiROCKET segment: curves + scalars | -0.085 | -0.190 | 0.024 | -0.058 | -0.162 | 0.051 |
| MultiROCKET segment: curves | -0.086 | -0.187 | 0.023 | -0.059 | -0.162 | 0.049 |
| MantisV2 frozen: curves + scalars | -0.049 | -0.128 | 0.028 | -0.070 | -0.161 | 0.030 |
| MantisV2 frozen: curves | -0.062 | -0.163 | 0.033 | -0.070 | -0.163 | 0.034 |
| MultiROCKET whole: curves | -0.159 | -0.280 | -0.034 | -0.078 | -0.207 | 0.054 |
| MultiROCKET whole: curves + scalars | -0.156 | -0.270 | -0.040 | -0.081 | -0.208 | 0.043 |
| MantisV2 adapter: curves + scalars | -0.065 | -0.145 | 0.013 | -0.086 | -0.165 | -0.004 |
| MantisV2 adapter: curves | -0.132 | -0.242 | -0.025 | -0.110 | -0.215 | -0.005 |

## What each model received

- MultiROCKET whole-heart: all 18 x 6 = 108 channels across 96 normalized time
  samples. It produced 49,728 convolution-response features.
- MultiROCKET segment-wise: each six-channel segment was transformed separately;
  feature mean, standard deviation, and maximum were pooled across 18 segments.
- MantisV2 and MOMENT-small: curves were interpolated from 96 to 512 samples only
  for pretrained-checkpoint compatibility. Every segment was encoded separately,
  while all six channel embeddings were retained. Segment mean, standard
  deviation, and maximum were used by the frozen probes.
- Frozen probes: fold-specific 32-component PCA followed by balanced logistic
  regression. PCA and scalar preprocessing were fit on training patients only.
- Mantis adapter: frozen pretrained embeddings, a learned six-channel weighting,
  256-to-32 embedding adapter, learned segment attention, and the same multitask
  labels and early-stopping scheme as the current CNN.

## Mantis adapter channel weights

| model | Channel | Mean weight | SD |
|---|---|---|---|
| mantis_v2_adapter_curves | change_endo | 0.172 | 0.004 |
| mantis_v2_adapter_curves | change_endo_minus_mid | 0.163 | 0.006 |
| mantis_v2_adapter_curves | change_mid | 0.171 | 0.004 |
| mantis_v2_adapter_curves | current_endo | 0.168 | 0.004 |
| mantis_v2_adapter_curves | current_endo_minus_mid | 0.160 | 0.004 |
| mantis_v2_adapter_curves | current_mid | 0.166 | 0.004 |
| mantis_v2_adapter_curves_scalars | change_endo | 0.171 | 0.003 |
| mantis_v2_adapter_curves_scalars | change_endo_minus_mid | 0.164 | 0.004 |
| mantis_v2_adapter_curves_scalars | change_mid | 0.169 | 0.002 |
| mantis_v2_adapter_curves_scalars | current_endo | 0.167 | 0.003 |
| mantis_v2_adapter_curves_scalars | current_endo_minus_mid | 0.161 | 0.004 |
| mantis_v2_adapter_curves_scalars | current_mid | 0.167 | 0.003 |

The weights remained close to the uniform value of 1/6 = 0.167. Therefore,
the adapter did not learn a stable dominant channel; the small preference for
the Endo and Mid change channels should not be interpreted as a strong finding.

## Controlled ablations

Positive values favor the named change.

| Ablation | delta AUC | delta AUC CI low | delta AUC CI high | delta AP | delta AP CI low | delta AP CI high |
|---|---|---|---|---|---|---|
| Mantis pretraining, curves | -0.048 | -0.138 | 0.052 | -0.052 | -0.145 | 0.036 |
| Mantis pretraining, curves + scalars | -0.069 | -0.136 | 0.004 | -0.067 | -0.155 | 0.008 |
| Add scalars to frozen MantisV2 | 0.013 | -0.038 | 0.064 | 0.000 | -0.054 | 0.059 |
| Add scalars to frozen MOMENT-small | 0.014 | -0.044 | 0.072 | 0.023 | -0.049 | 0.109 |
| Add scalars to random Mantis | 0.034 | -0.026 | 0.091 | 0.016 | -0.065 | 0.102 |
| Add scalars to Mantis adapter | 0.067 | 0.011 | 0.124 | 0.024 | -0.032 | 0.079 |
| Add scalars to whole-heart MultiROCKET | 0.003 | -0.002 | 0.008 | -0.003 | -0.017 | 0.003 |
| Add scalars to segment MultiROCKET | 0.001 | -0.001 | 0.003 | 0.001 | -0.000 | 0.003 |
| Segment vs whole MultiROCKET, curves | 0.074 | -0.026 | 0.178 | 0.019 | -0.088 | 0.127 |
| Segment vs whole MultiROCKET, curves + scalars | 0.071 | -0.020 | 0.167 | 0.023 | -0.080 | 0.128 |
| Mantis adapter vs frozen probe, curves | -0.071 | -0.155 | 0.016 | -0.039 | -0.117 | 0.034 |
| Mantis adapter vs frozen probe, curves + scalars | -0.016 | -0.094 | 0.056 | -0.016 | -0.082 | 0.046 |

## Interpretation guardrails

- A point estimate above the CNN is not sufficient if the paired confidence
  interval includes zero.
- Pretrained-versus-random Mantis separates transfer-learning value from the
  random-feature effect of the architecture.
- Interpolation to 512 samples adds no physiological information; it only adapts
  the data to the pretraining resolution.
- AP is the primary practical ranking measure because only 20.6% of eligible
  transitions are events.
