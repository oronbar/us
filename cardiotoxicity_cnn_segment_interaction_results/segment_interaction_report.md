# Segment-interaction CNN ablation

## Controlled comparison

The current 96-point six-channel CNN was compared with three focused variants:
eight-bin temporal pooling alone, four-head self-attention across the 18 segment
embeddings alone, and their combination. Patient folds, seeds, scalar features,
multitask targets, optimizer, and stopping rules were fixed. Training used
NVIDIA GeForce RTX 4060 Ti.

## Dimensions

| model | temporal representation | segment interaction | curve representation | parameters |
|---|---|---|---|---:|
| Current CNN | B x 18 x 24 | final pooling only | B x 72 | 17,491 |
| 8-bin temporal pooling only | B x 18 x 24 x 8 -> B x 18 x 24 | final pooling only | B x 72 | 22,123 |
| Segment self-attention | B x 18 x 24 | 4 x 18 x 18 maps | B x 72 | 22,363 |
| 8-bin + segment attention | B x 18 x 24 x 8 -> B x 18 x 24 | 4 x 18 x 18 maps | B x 72 | 26,995 |

## Results

| model | n | events | AUC | AUC CI low | AUC CI high | AP | AP CI low | AP CI high |
|---|---|---|---|---|---|---|---|---|
| Current CNN | 238.000 | 49.000 | 0.683 | 0.610 | 0.761 | 0.339 | 0.253 | 0.468 |
| 8-bin temporal pooling only | 238.000 | 49.000 | 0.691 | 0.606 | 0.774 | 0.333 | 0.243 | 0.494 |
| Segment self-attention | 238.000 | 49.000 | 0.677 | 0.599 | 0.757 | 0.310 | 0.222 | 0.433 |
| 8-bin + segment self-attention | 238.000 | 49.000 | 0.702 | 0.620 | 0.778 | 0.324 | 0.243 | 0.457 |

Random-ranking AUC is 0.500. Random-ranking AP equals the event prevalence,
0.206.

## Paired differences

| comparison | delta AUC | delta AUC CI low | delta AUC CI high | delta AP | delta AP CI low | delta AP CI high |
|---|---|---|---|---|---|---|
| timing8_only_vs_current | 0.008 | -0.052 | 0.062 | -0.007 | -0.066 | 0.091 |
| segment_attention_vs_current | -0.005 | -0.076 | 0.057 | -0.029 | -0.112 | 0.051 |
| timing8_attention_vs_current | 0.019 | -0.046 | 0.081 | -0.015 | -0.085 | 0.069 |
| timing8_attention_vs_timing8_only | 0.011 | -0.036 | 0.062 | -0.008 | -0.094 | 0.053 |
| timing8_vs_segment_attention | 0.024 | -0.031 | 0.081 | 0.014 | -0.042 | 0.083 |

A positive difference favors the candidate model. Intervals are paired
patient-cluster bootstrap confidence intervals.

## Interpretation

- Segment self-attention alone did not help: AUC decreased by 0.005 and AP by
  0.029 versus the current CNN.
- Preserving eight temporal bins alone increased AUC by 0.008 while AP decreased
  by 0.007.
- Combining eight bins with segment attention produced the highest AUC, 0.702
  versus 0.683, but AP was lower, 0.324 versus 0.339.
- Every paired confidence interval crossed zero. The observed changes are therefore
  hypotheses for a larger cohort, not established improvements.
- For the imbalanced early-alert task, retain the current CNN as the performance
  default because it has the highest AP. Keep the eight-bin combined model as an
  AUC-oriented research candidate.
