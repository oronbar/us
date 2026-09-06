# Round 3: focused pooling and ensemble experiments

## Locked protocol

The target is whether the immediately following visit is the first visit with at
least 15% relative Mid-GLS deterioration from first-visit baseline. Evaluation
uses 238 transitions, 49 events, 103 patients, the same three repeated five-fold
patient-held-out splits, and 2,000 patient-cluster bootstrap samples.

## Results

| Model | AUC | AUC CI low | AUC CI high | AP | AP CI low | AP CI high |
|---|---|---|---|---|---|---|
| Equal ensemble: cnn + mantis + timemil | 0.720 | 0.645 | 0.796 | 0.366 | 0.275 | 0.516 |
| Equal ensemble: cnn + mantis + moment + timemil | 0.715 | 0.635 | 0.791 | 0.366 | 0.264 | 0.505 |
| Equal ensemble: cnn + moment | 0.698 | 0.619 | 0.769 | 0.353 | 0.266 | 0.494 |
| Equal ensemble: cnn + mantis + moment | 0.710 | 0.632 | 0.780 | 0.353 | 0.256 | 0.488 |
| Cross-fit ensemble: cnn + mantis | 0.718 | 0.638 | 0.796 | 0.351 | 0.258 | 0.485 |
| Equal ensemble: cnn + mantis | 0.717 | 0.641 | 0.789 | 0.350 | 0.259 | 0.486 |
| Equal ensemble: cnn + timemil | 0.685 | 0.604 | 0.760 | 0.348 | 0.251 | 0.467 |
| Cross-fit ensemble: cnn + moment | 0.697 | 0.618 | 0.775 | 0.344 | 0.250 | 0.486 |
| Current CNN | 0.683 | 0.606 | 0.758 | 0.339 | 0.244 | 0.466 |
| Random Mantis + scalars | 0.702 | 0.620 | 0.777 | 0.337 | 0.253 | 0.470 |
| TimeMIL attention: curves + scalars | 0.678 | 0.601 | 0.756 | 0.337 | 0.238 | 0.460 |
| MOMENT-small + scalars | 0.678 | 0.596 | 0.755 | 0.335 | 0.245 | 0.471 |
| Cross-fit ensemble: cnn + mantis + timemil | 0.708 | 0.627 | 0.784 | 0.334 | 0.240 | 0.459 |
| Cross-fit ensemble: cnn + timemil | 0.678 | 0.597 | 0.761 | 0.332 | 0.234 | 0.457 |
| Cross-fit ensemble: cnn + mantis + moment | 0.711 | 0.635 | 0.788 | 0.330 | 0.246 | 0.469 |
| Cross-fit ensemble: cnn + hierarchical | 0.684 | 0.607 | 0.759 | 0.330 | 0.247 | 0.452 |
| Equal ensemble: cnn + hierarchical | 0.684 | 0.610 | 0.758 | 0.326 | 0.234 | 0.447 |
| Cross-fit ensemble: cnn + mantis + moment + timemil | 0.709 | 0.628 | 0.785 | 0.321 | 0.238 | 0.456 |
| TimeMIL hierarchical: curves + scalars | 0.675 | 0.594 | 0.752 | 0.291 | 0.215 | 0.420 |
| clinical_ridge | 0.631 | 0.537 | 0.724 | 0.289 | 0.212 | 0.442 |
| Scalar-only MLP | 0.669 | 0.589 | 0.750 | 0.286 | 0.217 | 0.417 |
| TimeMIL attention: curves | 0.604 | 0.516 | 0.692 | 0.277 | 0.199 | 0.408 |
| TimeMIL uniform: curves + scalars | 0.668 | 0.589 | 0.747 | 0.276 | 0.207 | 0.391 |
| TimeMIL uniform: curves | 0.618 | 0.534 | 0.703 | 0.270 | 0.196 | 0.397 |
| TimeMIL hierarchical: curves | 0.568 | 0.479 | 0.649 | 0.235 | 0.173 | 0.330 |

Random guessing has AUC 0.500 and expected AP 0.206.

## Paired changes from current CNN

| Model | delta AUC | delta AUC CI low | delta AUC CI high | delta AP | delta AP CI low | delta AP CI high |
|---|---|---|---|---|---|---|
| TimeMIL uniform: curves | -0.065 | -0.156 | 0.028 | -0.069 | -0.159 | 0.040 |
| TimeMIL uniform: curves + scalars | -0.015 | -0.065 | 0.037 | -0.063 | -0.130 | 0.007 |
| TimeMIL hierarchical: curves | -0.115 | -0.216 | -0.014 | -0.104 | -0.201 | -0.010 |
| TimeMIL hierarchical: curves + scalars | -0.008 | -0.069 | 0.053 | -0.048 | -0.112 | 0.028 |
| Equal ensemble: cnn + mantis | 0.035 | -0.033 | 0.099 | 0.010 | -0.088 | 0.120 |
| Equal ensemble: cnn + moment | 0.015 | -0.053 | 0.081 | 0.014 | -0.076 | 0.128 |
| Equal ensemble: cnn + timemil | 0.003 | -0.023 | 0.027 | 0.008 | -0.020 | 0.036 |
| Equal ensemble: cnn + hierarchical | 0.001 | -0.035 | 0.033 | -0.014 | -0.054 | 0.025 |
| Equal ensemble: cnn + mantis + moment | 0.028 | -0.043 | 0.099 | 0.013 | -0.083 | 0.129 |
| Equal ensemble: cnn + mantis + timemil | 0.038 | -0.021 | 0.100 | 0.026 | -0.058 | 0.131 |
| Equal ensemble: cnn + mantis + moment + timemil | 0.032 | -0.032 | 0.097 | 0.026 | -0.069 | 0.144 |
| Cross-fit ensemble: cnn + mantis | 0.035 | -0.020 | 0.091 | 0.012 | -0.067 | 0.100 |
| Cross-fit ensemble: cnn + moment | 0.014 | -0.051 | 0.078 | 0.005 | -0.082 | 0.111 |
| Cross-fit ensemble: cnn + timemil | -0.004 | -0.028 | 0.018 | -0.007 | -0.039 | 0.023 |
| Cross-fit ensemble: cnn + hierarchical | 0.001 | -0.006 | 0.009 | -0.010 | -0.026 | 0.003 |
| Cross-fit ensemble: cnn + mantis + moment | 0.028 | -0.037 | 0.095 | -0.010 | -0.094 | 0.092 |
| Cross-fit ensemble: cnn + mantis + timemil | 0.025 | -0.031 | 0.082 | -0.006 | -0.076 | 0.074 |
| Cross-fit ensemble: cnn + mantis + moment + timemil | 0.026 | -0.037 | 0.086 | -0.018 | -0.098 | 0.081 |

## Pooling ablations

Positive values favor the first named method.

| Ablation | delta AUC | delta AUC CI low | delta AUC CI high | delta AP | delta AP CI low | delta AP CI high |
|---|---|---|---|---|---|---|
| Attention vs uniform, curves | -0.014 | -0.078 | 0.049 | 0.006 | -0.069 | 0.087 |
| Attention vs uniform, curves + scalars | 0.011 | -0.022 | 0.046 | 0.061 | 0.002 | 0.124 |
| Hierarchical vs flat attention, curves | -0.036 | -0.113 | 0.040 | -0.041 | -0.135 | 0.023 |
| Hierarchical vs flat attention, curves + scalars | -0.004 | -0.054 | 0.042 | -0.046 | -0.113 | 0.022 |
| Hierarchical vs uniform, curves | -0.050 | -0.125 | 0.023 | -0.035 | -0.114 | 0.019 |
| Hierarchical vs uniform, curves + scalars | 0.007 | -0.045 | 0.055 | 0.015 | -0.036 | 0.078 |
| Add scalars to hierarchical TimeMIL | 0.107 | 0.031 | 0.187 | 0.056 | -0.001 | 0.139 |

## Architecture definitions

- Uniform TimeMIL retains the same 216 patch tokens and global transformer as
  Round 2, but replaces learned MIL attention with an unweighted mean.
- Hierarchical TimeMIL first applies temporal self-attention to the 12 patches
  within each segment, pools each segment, then applies a second self-attention
  layer and learned pooling across the 18 segments.
- Every neural variant uses the same scalar branch, multitask loss, class
  weighting, inner patient validation, and early stopping as the retained CNN.

## Cross-fitted ensemble weights

The convex blend weights were selected separately inside every outer split using
only training-patient labels and already patient-out-of-fold base predictions.
They were then applied to the held-out patients. Equal-weight ensembles are fully
prespecified controls.

| Ensemble | Model | Mean weight | SD |
|---|---|---|---|
| cnn + hierarchical | Current CNN | 0.947 | 0.092 |
| cnn + hierarchical | TimeMIL hierarchical: curves + scalars | 0.053 | 0.092 |
| cnn + mantis | Current CNN | 0.720 | 0.115 |
| cnn + mantis | Random Mantis + scalars | 0.280 | 0.115 |
| cnn + mantis + moment | Current CNN | 0.640 | 0.176 |
| cnn + mantis + moment | Random Mantis + scalars | 0.267 | 0.150 |
| cnn + mantis + moment | MOMENT-small + scalars | 0.093 | 0.144 |
| cnn + mantis + moment + timemil | Current CNN | 0.433 | 0.250 |
| cnn + mantis + moment + timemil | Random Mantis + scalars | 0.200 | 0.136 |
| cnn + mantis + moment + timemil | MOMENT-small + scalars | 0.067 | 0.163 |
| cnn + mantis + moment + timemil | TimeMIL attention: curves + scalars | 0.300 | 0.245 |
| cnn + mantis + timemil | Current CNN | 0.473 | 0.243 |
| cnn + mantis + timemil | Random Mantis + scalars | 0.227 | 0.133 |
| cnn + mantis + timemil | TimeMIL attention: curves + scalars | 0.300 | 0.242 |
| cnn + moment | Current CNN | 0.627 | 0.234 |
| cnn + moment | MOMENT-small + scalars | 0.373 | 0.234 |
| cnn + timemil | Current CNN | 0.607 | 0.260 |
| cnn + timemil | TimeMIL attention: curves + scalars | 0.393 | 0.260 |

## Ensemble ablations

Positive values favor the equal ensemble in each comparison.

| Comparison | delta AUC | delta AUC CI low | delta AUC CI high | delta AP | delta AP CI low | delta AP CI high |
|---|---|---|---|---|---|---|
| Best equal ensemble vs random Mantis | 0.018 | -0.010 | 0.050 | 0.029 | -0.021 | 0.076 |
| Best equal ensemble vs TimeMIL attention | 0.042 | -0.012 | 0.097 | 0.029 | -0.051 | 0.121 |
| Equal vs cross-fit, CNN + Mantis + TimeMIL | 0.012 | -0.007 | 0.031 | 0.032 | -0.001 | 0.081 |
| Equal vs cross-fit, CNN + Mantis | -0.000 | -0.020 | 0.019 | -0.001 | -0.042 | 0.054 |

## Hierarchical attention

Uniform segment mass is 1/18 = 0.0556. Attention weights are descriptive rather
than causal feature importance.

| Model | Segment | Mean segment mass |
|---|---|---|
| TimeMIL hierarchical: curves | 10.000 | 0.0607 |
| TimeMIL hierarchical: curves | 18.000 | 0.0592 |
| TimeMIL hierarchical: curves | 13.000 | 0.0584 |
| TimeMIL hierarchical: curves | 3.000 | 0.0576 |
| TimeMIL hierarchical: curves | 5.000 | 0.0575 |
| TimeMIL hierarchical: curves + scalars | 18.000 | 0.0582 |
| TimeMIL hierarchical: curves + scalars | 3.000 | 0.0582 |
| TimeMIL hierarchical: curves + scalars | 11.000 | 0.0577 |
| TimeMIL hierarchical: curves + scalars | 5.000 | 0.0570 |
| TimeMIL hierarchical: curves + scalars | 6.000 | 0.0570 |
| TimeMIL uniform: curves | 1.000 | 0.0556 |
| TimeMIL uniform: curves | 2.000 | 0.0556 |
| TimeMIL uniform: curves | 3.000 | 0.0556 |
| TimeMIL uniform: curves | 4.000 | 0.0556 |
| TimeMIL uniform: curves | 5.000 | 0.0556 |
| TimeMIL uniform: curves + scalars | 1.000 | 0.0556 |
| TimeMIL uniform: curves + scalars | 2.000 | 0.0556 |
| TimeMIL uniform: curves + scalars | 3.000 | 0.0556 |
| TimeMIL uniform: curves + scalars | 4.000 | 0.0556 |
| TimeMIL uniform: curves + scalars | 5.000 | 0.0556 |
