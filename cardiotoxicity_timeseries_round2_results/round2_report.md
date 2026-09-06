# Round 2: specialized time-series models

## Task and protocol

Predict whether the immediately following visit will be the first visit with at
least 15% relative Mid-GLS deterioration from the first-visit baseline. The
locked evaluation contains 238 eligible transitions, 49 events, and 103
patients. All results use the same three repeated five-fold patient-held-out
splits as Rounds 1 and the current CNN. Confidence intervals use 2,000
patient-cluster bootstrap samples.

## Results

| Model | AUC | AUC CI low | AUC CI high | AP | AP CI low | AP CI high |
|---|---|---|---|---|---|---|
| Current CNN | 0.683 | 0.605 | 0.759 | 0.339 | 0.243 | 0.466 |
| Round 1 random-Mantis control + scalars | 0.702 | 0.621 | 0.776 | 0.337 | 0.248 | 0.463 |
| TimeMIL-lite: curves + scalars | 0.678 | 0.597 | 0.755 | 0.337 | 0.237 | 0.460 |
| Round 1 MOMENT-small + scalars | 0.678 | 0.597 | 0.761 | 0.335 | 0.247 | 0.484 |
| Random TS2Vec: curves | 0.652 | 0.562 | 0.737 | 0.319 | 0.229 | 0.458 |
| TS2Vec SSL: curves | 0.633 | 0.542 | 0.722 | 0.301 | 0.219 | 0.441 |
| Clinical ridge | 0.631 | 0.537 | 0.724 | 0.289 | 0.204 | 0.439 |
| Random TS2Vec: curves + scalars | 0.639 | 0.558 | 0.725 | 0.287 | 0.215 | 0.408 |
| Scalar-only multitask MLP | 0.669 | 0.588 | 0.753 | 0.286 | 0.213 | 0.416 |
| TimeMIL-lite: curves | 0.604 | 0.519 | 0.690 | 0.277 | 0.201 | 0.415 |
| TS2Vec SSL: curves + scalars | 0.624 | 0.535 | 0.710 | 0.267 | 0.200 | 0.388 |
| MambaSL: curves + scalars | 0.581 | 0.499 | 0.657 | 0.234 | 0.178 | 0.324 |
| MambaSL: curves | 0.452 | 0.368 | 0.537 | 0.182 | 0.137 | 0.257 |

Random guessing has AUC 0.500 and expected AP 0.206. The highest AP in the full
comparison was **Current CNN**, with AUC 0.683 and AP 0.339.

## Paired change from the current CNN

| Model | delta AUC | delta AUC CI low | delta AUC CI high | delta AP | delta AP CI low | delta AP CI high |
|---|---|---|---|---|---|---|
| TimeMIL-lite: curves + scalars | -0.004 | -0.045 | 0.038 | -0.003 | -0.048 | 0.047 |
| Random TS2Vec: curves | -0.031 | -0.120 | 0.063 | -0.020 | -0.129 | 0.103 |
| TS2Vec SSL: curves | -0.050 | -0.147 | 0.051 | -0.039 | -0.140 | 0.097 |
| Random TS2Vec: curves + scalars | -0.044 | -0.130 | 0.035 | -0.052 | -0.147 | 0.059 |
| Scalar-only multitask MLP | -0.014 | -0.056 | 0.028 | -0.053 | -0.106 | 0.011 |
| TimeMIL-lite: curves | -0.079 | -0.182 | 0.028 | -0.063 | -0.172 | 0.078 |
| TS2Vec SSL: curves + scalars | -0.058 | -0.148 | 0.032 | -0.072 | -0.170 | 0.035 |
| MambaSL: curves + scalars | -0.102 | -0.178 | -0.033 | -0.106 | -0.184 | -0.030 |
| MambaSL: curves | -0.231 | -0.339 | -0.125 | -0.158 | -0.258 | -0.065 |

## Controlled ablations

Positive values favor the named change.

| Ablation | delta AUC | delta AUC CI low | delta AUC CI high | delta AP | delta AP CI low | delta AP CI high |
|---|---|---|---|---|---|---|
| Add scalars to MambaSL | 0.128 | 0.028 | 0.226 | 0.052 | 0.000 | 0.114 |
| Add scalars to TimeMIL-lite | 0.075 | -0.017 | 0.163 | 0.060 | -0.061 | 0.162 |
| Add scalars to TS2Vec SSL | -0.008 | -0.059 | 0.041 | -0.034 | -0.112 | 0.017 |
| Add scalars to random TS2Vec | -0.013 | -0.059 | 0.033 | -0.032 | -0.089 | 0.028 |
| TS2Vec SSL vs random initialization, curves | -0.019 | -0.091 | 0.059 | -0.018 | -0.100 | 0.091 |
| TS2Vec SSL vs random initialization, curves + scalars | -0.014 | -0.070 | 0.042 | -0.020 | -0.086 | 0.039 |
| TimeMIL-lite vs MambaSL, curves | 0.152 | 0.054 | 0.240 | 0.095 | 0.025 | 0.199 |
| TimeMIL-lite vs MambaSL, curves + scalars | 0.098 | 0.028 | 0.175 | 0.103 | 0.027 | 0.186 |
| TimeMIL-lite + scalars vs scalar-only MLP | 0.009 | -0.026 | 0.047 | 0.050 | -0.005 | 0.097 |
| MambaSL + scalars vs scalar-only MLP | -0.088 | -0.159 | -0.017 | -0.053 | -0.126 | -0.001 |

## Model definitions

- MambaSL: 108 simultaneous segment-channel variables across 96 samples, a
  seven-sample input projection, one 32-dimensional selective state-space layer,
  and four-head adaptive temporal pooling. The official fused selective-scan
  extension is Linux-only, so the same recurrence was evaluated with an explicit
  pure-PyTorch scan on Windows.
- TimeMIL-lite: every eight-sample patch from every segment is an instance, giving
  18 x 12 = 216 instances per transition. It uses a 32-dimensional patch encoder,
  segment embeddings, multiscale temporal positional embeddings, one four-head
  transformer layer, 10% patch masking, and MIL attention pooling.
- TS2Vec: official hierarchical contrastive training for 200 iterations in each
  outer fold using only curves from that fold's training patients. Six-channel
  segment series are encoded into 128-dimensional embeddings and pooled across
  segments. A random-initialization control uses the identical encoder before
  contrastive training.
- All trainable supervised models use the same multitask labels, class weighting,
  internal patient validation, and early stopping as the retained CNN.
- The scalar-only MLP uses the identical 96-to-32 scalar branch and multitask
  head, but receives no curves. It isolates the incremental value of each curve
  architecture.

## TimeMIL segment attention

| Model | Segment | Mean segment mass |
|---|---|---|
| timemil_curves | 5.000 | 0.059 |
| timemil_curves | 4.000 | 0.059 |
| timemil_curves | 3.000 | 0.058 |
| timemil_curves | 6.000 | 0.058 |
| timemil_curves | 18.000 | 0.057 |
| timemil_curves_scalars | 5.000 | 0.057 |
| timemil_curves_scalars | 3.000 | 0.057 |
| timemil_curves_scalars | 4.000 | 0.057 |
| timemil_curves_scalars | 6.000 | 0.056 |
| timemil_curves_scalars | 11.000 | 0.056 |

Uniform segment mass is 1/18 = 0.0556. Attention concentration is summarized
below; normalized entropy 1.0 means completely uniform attention across all 216
instances.

| Model | Mean normalized entropy |
|---|---|
| timemil_curves | 0.987 |
| timemil_curves_scalars | 0.997 |

Attention values are descriptive, not causal feature importance. Stable segment
preferences should be interpreted only if the model itself generalizes.
