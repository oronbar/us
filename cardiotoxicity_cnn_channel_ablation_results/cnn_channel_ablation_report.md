# Attention-CNN curve-channel ablation

## Question

Does restricting the curve branch to the Endo–Mid gap improve prediction of a
15% relative Mid-GLS decline at the immediately following visit?

The experiment kept the 96 scalar features, attention pooling, multitask labels,
three repeated five-fold patient-held-out splits, validation procedure, seeds,
optimizer, and stopping rule fixed. Only the curve channels changed.

## Results

| model | n | events | AUC | AUC CI low | AUC CI high | AP | AP CI low | AP CI high |
|---|---|---|---|---|---|---|---|---|
| Clinical ridge | 238.000 | 49.000 | 0.631 | 0.535 | 0.724 | 0.289 | 0.206 | 0.433 |
| Combined Extra Trees | 238.000 | 49.000 | 0.672 | 0.591 | 0.752 | 0.313 | 0.226 | 0.445 |
| Attention CNN: full 6 channels | 238.000 | 49.000 | 0.683 | 0.602 | 0.758 | 0.339 | 0.248 | 0.458 |
| Attention CNN: Endo-Mid gap only | 238.000 | 49.000 | 0.669 | 0.582 | 0.753 | 0.308 | 0.225 | 0.456 |
| Attention CNN: separate layers only | 238.000 | 49.000 | 0.682 | 0.601 | 0.767 | 0.313 | 0.230 | 0.459 |
| Attention CNN: gap + normalized shape gap | 238.000 | 49.000 | 0.675 | 0.595 | 0.752 | 0.299 | 0.224 | 0.432 |

## Paired differences from the full six-channel attention CNN

| candidate | ΔAUC | ΔAUC CI low | ΔAUC CI high | ΔAP | ΔAP CI low | ΔAP CI high |
|---|---|---|---|---|---|---|
| Attention CNN: Endo-Mid gap only | -0.014 | -0.063 | 0.035 | -0.031 | -0.096 | 0.063 |
| Attention CNN: separate layers only | -0.000 | -0.037 | 0.036 | -0.026 | -0.071 | 0.037 |
| Attention CNN: gap + normalized shape gap | -0.008 | -0.048 | 0.034 | -0.041 | -0.090 | 0.021 |

The highest channel-ablation AUC was **Attention CNN: full 6 channels**: AUC 0.683,
AP 0.339. Confidence intervals are patient-cluster bootstraps.

## Conclusion

- Restricting the curve branch to the two Endo–Mid gap channels did not improve the
  model. Its point estimates were lower by 0.014 AUC and 0.031 AP.
- Removing the explicit gap channels while retaining separate Endo and Mid curves
  preserved AUC almost exactly, but reduced AP by 0.026. The CNN can learn a
  subtraction internally, although the explicit gap channels may help event ranking.
- Adding normalized shape-gap channels did not recover the loss and had the lowest
  AP of the CNN variants.
- None of the paired intervals excluded zero, so the dataset cannot establish a
  definitive difference. The full six-channel input remains the preferred
  representation because it had the best AUC/AP point estimates and retains all
  layer-specific information.

## Interpretation rules

- A positive paired difference favors the reduced channel representation.
- An interval crossing zero means the change is not stable in this cohort.
- The gap-only experiment changes only the CNN curve branch. The parallel scalar
  branch still supplies baseline/current GLS, EF, trajectory, and variability
  features.
- `gap + normalized shape gap` adds the current and longitudinal change in
  `Endo/max|Endo| - Mid/max|Mid|`, matching the strongest engineered shape-gap
  features more closely than the raw difference alone.

## Reproduction check

The newly trained full-six reference differed from the previously saved attention
CNN by -0.000216 AUC and
0.006981 AP.
