# CNN temporal input-length ablation

## Controlled question

Does changing only the normalized cardiac-cycle length from 96 to 72 or 64 samples
improve prediction of a 15% relative Mid-GLS decline at the immediately following visit?

The six curve channels, 18 segments, 96 scalar features, multitask labels, three
repeated five-fold patient-held-out splits, train/validation partitions, seeds,
CNN kernels, optimizer, and early stopping were held fixed. Candidate curves were
interpolated directly from native samples and timestamps onto the same normalized
cycle grid. Training used NVIDIA GeForce RTX 4060 Ti.

## Results

| length | n | events | AUC | AUC CI low | AUC CI high | AP | AP CI low | AP CI high |
|---|---|---|---|---|---|---|---|---|
| 64.000 | 238.000 | 49.000 | 0.683 | 0.611 | 0.761 | 0.332 | 0.245 | 0.466 |
| 72.000 | 238.000 | 49.000 | 0.684 | 0.605 | 0.764 | 0.333 | 0.241 | 0.461 |
| 96.000 | 238.000 | 49.000 | 0.683 | 0.602 | 0.758 | 0.339 | 0.248 | 0.458 |

Event prevalence, and therefore random-ranking AP, was 0.206. Random-ranking
AUC is 0.500.

## Paired differences versus 96 samples

| comparison | delta AUC | delta AUC CI low | delta AUC CI high | delta AP | delta AP CI low | delta AP CI high |
|---|---|---|---|---|---|---|
| t64_vs_t96 | 0.000 | -0.003 | 0.004 | -0.007 | -0.021 | 0.004 |
| t72_vs_t96 | 0.001 | -0.001 | 0.005 | -0.006 | -0.019 | 0.004 |

A positive difference favors the shorter candidate. Confidence intervals are
patient-cluster bootstraps and account for paired predictions on the same cases.

## GPU fit time

| length | total fit seconds | fit time vs 96 |
|---|---|---|
| 64.000 | 39.200 | 0.902 |
| 72.000 | 40.380 | 0.929 |
| 96.000 | 43.470 | 1.000 |

These totals cover the 15 fold fits per length, not preprocessing or bootstrap time.

## Interpretation

- Highest AUC: 72 samples (0.684).
- Highest AP: 96 samples (0.339).
- The AUC differences are negligible. Both shorter inputs reduce AP by about 0.006
  to 0.007, but their paired confidence intervals include zero.
- For performance-first use, retain 96 because it has the highest AP. If reducing
  input size is operationally important, 64 is a defensible compressed setting with
  no meaningful observed AUC loss, although a small AP loss cannot be excluded.
- This primary ablation deliberately keeps the 7- and 5-sample convolution kernels
  unchanged. It therefore tests the deployed architecture's input length, including
  the associated change in each kernel's fraction of the cardiac cycle.
