# Cycle-proportional CNN kernel ablation

## Controlled question

Can shorter temporal inputs improve when convolution kernels are reduced to preserve
approximately the same fraction of the normalized cardiac cycle as the 96-point
7+5-kernel reference?

The effective receptive fields were 11/96 = 11.5% for the reference, 9/72 = 12.5%
for scaled 72, and 7/64 = 10.9% for scaled 64. Odd kernels were used to preserve
symmetric centering. All patient folds, seeds, inputs, scalar features, multitask
labels, optimizer settings, and early stopping rules were unchanged. Training used
NVIDIA GeForce RTX 4060 Ti.

## Results

| model | n | events | AUC | AUC CI low | AUC CI high | AP | AP CI low | AP CI high |
|---|---|---|---|---|---|---|---|---|
| 64 fixed 7+5 | 238.000 | 49.000 | 0.683 | 0.611 | 0.761 | 0.332 | 0.245 | 0.466 |
| 64 scaled 5+3 | 238.000 | 49.000 | 0.671 | 0.589 | 0.756 | 0.294 | 0.221 | 0.429 |
| 72 fixed 7+5 | 238.000 | 49.000 | 0.684 | 0.605 | 0.764 | 0.333 | 0.241 | 0.461 |
| 72 scaled 5+5 | 238.000 | 49.000 | 0.682 | 0.600 | 0.766 | 0.334 | 0.243 | 0.480 |
| 96 reference 7+5 | 238.000 | 49.000 | 0.683 | 0.602 | 0.758 | 0.339 | 0.248 | 0.458 |

Random-ranking AUC is 0.500 and random-ranking AP is the event prevalence,
0.206.

## Paired differences

| comparison | delta AUC | delta AUC CI low | delta AUC CI high | delta AP | delta AP CI low | delta AP CI high |
|---|---|---|---|---|---|---|
| scaled64_vs_fixed64 | -0.012 | -0.055 | 0.028 | -0.039 | -0.090 | 0.027 |
| scaled72_vs_fixed72 | -0.002 | -0.037 | 0.036 | 0.001 | -0.033 | 0.060 |
| scaled64_vs_t96 | -0.012 | -0.055 | 0.029 | -0.046 | -0.094 | 0.024 |
| scaled72_vs_t96 | -0.001 | -0.037 | 0.034 | -0.005 | -0.046 | 0.053 |

A positive difference favors the scaled-kernel candidate. Intervals are paired
patient-cluster bootstrap confidence intervals.

## Interpretation

- At 72 samples, scaling from 7+5 to 5+5 was neutral: delta AUC -0.002 and
  delta AP +0.001, with both confidence intervals crossing zero.
- At 64 samples, scaling from 7+5 to 5+3 was unfavorable in point estimates:
  delta AUC -0.012 and delta AP -0.039. The intervals still cross zero, so the
  cohort cannot establish a definitive loss.
- Neither scaled model improved on the 96-point reference. The 96-point model
  retained the highest AP.
- Narrower kernels also reduce parameter count, especially for 64 samples, so its
  point-estimate loss may reflect both receptive-field and capacity changes.
- Recommendation: retain 96 with 7+5 kernels for performance-first use. If a shorter
  input is required, retain the fixed 7+5 kernels rather than scaling them down.
