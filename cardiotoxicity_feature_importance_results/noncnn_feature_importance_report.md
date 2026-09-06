# Clear report: feature importance in the non-CNN cardiotoxicity models

## Executive summary

The clearest engineered signal came from the **relationship between the Endocardial
and Mid-wall strain curves**, particularly how that relationship varied across
segments and changed between visits. Inter-segment variability contributed less and
was less stable.

- Strongest engineered feature: **change in the phase pattern of segmental Endo–Mid
  gap variability** (`d_tm_sd_gap_dct04`).
- Strongest Extra Trees feature: **overall normalized Endo–Mid curve-shape
  separation** (`cur_tm_mean_shape_gap_dct01`).
- Baseline GLS remained important, but baseline Mid-GLS is mathematically connected
  to the relative-decline outcome and must be interpreted cautiously.
- Most individual variability features had confidence intervals crossing zero.

## Prediction task and dataset

At every eligible visit, the model uses information available up to that visit to
predict whether the **immediately following visit** will show at least a **15% relative
decline in Mid-GLS from the first-visit baseline**:

`relative decline = 1 - (next-visit Mid-GLS magnitude / first-visit Mid-GLS magnitude)`

An event is recorded when `relative decline >= 0.15`.

- 103 patients
- 238 eligible current-to-next-visit predictions
- 49 deterioration events (20.6%)
- 189 non-events (79.4%)
- All visits from the same patient remain in the same held-out fold.

## Models and performance

| model | AUC | AP |
|---|---|---|
| Clinical ridge | 0.631 | 0.289 |
| Clinical + Endo–Mid | 0.644 | 0.309 |
| Clinical + variability | 0.650 | 0.288 |
| Combined Extra Trees | 0.672 | 0.313 |

Random guessing has AUC 0.500. Its expected AP equals the event rate, approximately
0.206.

## How predictive contribution was measured

- The four existing non-CNN models were reconstructed using the original three
  repeated five-fold patient splits and the same preprocessing, regularization, and
  random seeds.
- The strongest candidates were screened using standardized coefficient magnitude for
  the logistic models and tree importance for Extra Trees.
- Final ranking used **held-out permutation importance**: one feature was randomly
  shuffled, the model predicted again, and the decrease in AUC/AP was measured.
- A large positive decrease means the model genuinely depended on that feature in
  held-out patients. A value near zero means the feature was redundant, unused, or too
  noisy. A negative value means shuffling improved performance.
- Confidence intervals resample patients while keeping all transitions belonging to a
  sampled patient together.

## How to read the feature names

| Name component | Meaning |
|---|---|
| `first_` | Value from the patient's first visit, used as the baseline |
| `cur_` | Value at the current visit from which the next visit is predicted |
| `d_` | Current value minus the value at the immediately previous visit |
| `tm_` | Transmural feature comparing matched Endocardial and Mid-wall curves |
| `var__` | Variability between myocardial segments |
| `gap` | Endocardial value minus Mid-wall value |
| `shape_gap` | Endo–Mid difference after each curve is normalized by its own amplitude |
| `DCT01`, `DCT04`, etc. | Coefficients summarizing the complete time-dependent curve pattern |

### What DCT means here

DCT stands for **Discrete Cosine Transform**. It converts a 96-point curve into a
small number of coefficients:

- `DCT01`: the broad overall level or offset of the curve.
- `DCT02`: a slow trend across the cardiac cycle.
- `DCT04`: a more detailed phase-dependent pattern.
- `DCT07`: a still finer temporal pattern.

A DCT coefficient is not a value at one time point. It summarizes a pattern spread
across the cardiac cycle. Higher-numbered coefficients are progressively less
intuitive physiologically and should be interpreted as curve-shape descriptors.

## Most important feature in each model

### 1. Clinical ridge: baseline Endocardial GLS magnitude

**Technical name:** `first_endo_gls`

The source Endocardial GLS is negative. The model uses its positive magnitude:

`first_endo_gls = absolute value of first-visit gls_endo_peak_avg`

For example, a baseline Endo-GLS of `-21.4%` becomes `21.4%`. This value is then
carried forward for every prediction belonging to that patient.

The positive standardized coefficient means that, conditional on the other clinical
trajectory variables, a larger baseline Endo-GLS magnitude produced a higher predicted
risk. This may reflect physiological reserve, correlation with baseline Mid-GLS, or
regression to the mean; it should not be interpreted as evidence that better baseline
function causes cardiotoxicity.

**Importance:** AUC decrease **0.018** (patient-bootstrap 95% CI 0.003 to 0.035); AP decrease **0.016**; CI excludes zero: **yes**.

### 2. Clinical + Endo–Mid sparse model: change in segmental Endo–Mid gap variability

**Technical name:** `d_tm_sd_gap_dct04`

This feature is calculated in six steps:

1. Pair the Endocardial and Mid-wall longitudinal-strain curves from the same segment.
2. At each cardiac-cycle point, calculate the raw layer gap for every segment:
   `segment gap_s(t) = Endo_s(t) - Mid_s(t)`.
3. At each time point, calculate the standard deviation of that gap across the matched
   segments: `SD_gap(t) = SD across segments of segment gap_s(t)`.
4. `SD_gap(t)` is now a complete curve describing *when* segments disagree in their
   Endo–Mid separation.
5. Apply the DCT and retain coefficient 4 (`DCT04`, mathematical index `k=3`).
6. Subtract the previous visit's coefficient from the current visit's coefficient:
   `current DCT04 - previous DCT04`.

In compact form:

`paired Endo/Mid curves -> segment gap curves -> SD across segments -> DCT04 -> current minus previous`

It does **not** simply mean that overall variability increased. It means that a
particular phase-dependent pattern of Endo–Mid heterogeneity changed between visits.
The sparse logistic model selected it in every outer fold, with a positive median
standardized coefficient: an increase in this component raised predicted risk.

**Importance:** AUC decrease **0.037** (patient-bootstrap 95% CI 0.001 to 0.066); AP decrease **0.043**; CI excludes zero: **yes**.

### 3. Clinical + variability sparse model: baseline Mid-wall GLS magnitude

**Technical name:** `first_mid_gls`

The calculation is:

`first_mid_gls = absolute value of first-visit gls_mid_peak_avg`

For example, `-18.2%` becomes `18.2%`. This was the strongest feature in the model
that also contained segment-variability features.

Important caution: the outcome is itself defined relative to this value:

`event if 1 - (next Mid-GLS / first Mid-GLS) >= 0.15`

Therefore, baseline Mid-GLS is mathematically coupled to the label. This is not future
data leakage, because baseline is known at prediction time, but it can amplify
baseline measurement error and regression-to-the-mean effects. Its confidence
interval also crossed zero.

**Importance:** AUC decrease **0.030** (patient-bootstrap 95% CI -0.023 to 0.089); AP decrease **0.011**; CI excludes zero: **no**.

#### Strongest actual variability feature

The strongest variability-specific feature was **robust between-segment dispersion of
current Mid-wall peak strain** (`cur_var__mid_peak_abs_robust_sd`). For segment peak
magnitudes `p_s`, it is calculated as:

`1.4826 x median(|p_s - median(p)|)`

This robust standard deviation is less affected by one badly tracked segment than an
ordinary standard deviation. In this fitted sparse model, its coefficient was
negative, but its AP contribution was slightly negative and its confidence interval
crossed zero. It is therefore not a stable biological finding.

**Importance:** AUC decrease **0.010** (patient-bootstrap 95% CI -0.014 to 0.036); AP decrease **-0.001**; CI excludes zero: **no**.

### 4. Combined Extra Trees: overall normalized Endo–Mid curve-shape separation

**Technical name:** `cur_tm_mean_shape_gap_dct01`

This feature intentionally removes most amplitude information:

1. For every matched segment, divide the Endocardial curve by its own maximum absolute
   amplitude.
2. Do the same for the Mid-wall curve. Curves with maximum magnitude below 3% are
   treated as invalid.
3. Calculate the normalized Endo–Mid shape gap at each time point:
   `normalized Endo_s(t) - normalized Mid_s(t)`.
4. Average this gap across segments to obtain one mean shape-gap curve.
5. Apply the DCT and retain `DCT01`. This first coefficient is proportional to the
   average level of the shape-gap curve across the cardiac cycle.

It represents broad, systematic separation between Endocardial and Mid-wall curve
shapes, largely independent of their absolute strain amplitudes. Extra Trees can use
different thresholds and interactions, so this feature has no single global
"higher means higher risk" direction.

**Importance:** AUC decrease **0.017** (patient-bootstrap 95% CI 0.008 to 0.026); AP decrease **0.020**; CI excludes zero: **yes**.

## Features with stable positive held-out importance

Only features whose patient-bootstrap AUC interval was entirely above zero are shown.

| Plain-language feature | Technical name | Model | Family | AUC decrease | AUC 95% CI | AP decrease |
|---|---|---|---|---|---|---|
| Change in the phase pattern of segmental Endo–Mid gap variability | d_tm_sd_gap_dct04 | Clinical + Endo–Mid | Endo–Mid | 0.037 | 0.001 to 0.066 | 0.043 |
| Baseline Endocardial GLS magnitude | first_endo_gls | Clinical ridge | clinical | 0.018 | 0.003 to 0.035 | 0.016 |
| Overall normalized Endo–Mid curve-shape separation | cur_tm_mean_shape_gap_dct01 | Combined Extra Trees | Endo–Mid | 0.017 | 0.008 to 0.026 | 0.020 |
| Finer phase pattern of normalized Endo–Mid shape separation | cur_tm_mean_shape_gap_dct07 | Combined Extra Trees | Endo–Mid | 0.009 | 0.002 to 0.015 | 0.009 |
| Endocardial between-segment time-to-peak dispersion | cur_var__endo_vendor_time_to_peak_norm_circular_std | Combined Extra Trees | variability | 0.008 | 0.003 to 0.014 | 0.007 |
| Change in the mean vendor Endo–Mid peak-systolic gap | d_tm_vendor_peak_gap_mean | Combined Extra Trees | Endo–Mid | 0.005 | 0.001 to 0.010 | 0.001 |

## Top five features within each model

The technical name is retained so every row can be traced back to the source table.

### Clinical ridge

| Plain-language feature | Technical name | Family | AUC decrease | AUC 95% CI | AP decrease | Stable | Model direction |
|---|---|---|---|---|---|---|---|
| Baseline Endocardial GLS magnitude | first_endo_gls | clinical | 0.018 | 0.003 to 0.035 | 0.016 | Yes | Higher value -> higher predicted risk |
| Baseline Mid-wall GLS magnitude | first_mid_gls | clinical | 0.015 | -0.003 to 0.033 | 0.010 | No | Higher value -> higher predicted risk |
| Current relative Mid-GLS decline from baseline | current_mid_decline_from_first | clinical | 0.013 | -0.007 to 0.031 | -0.007 | No | Higher value -> higher predicted risk |
| Current relative Endo-GLS decline from baseline | current_endo_decline_from_first | clinical | 0.007 | -0.007 to 0.019 | -0.005 | No | Higher value -> higher predicted risk |
| Most recent relative EF change | last_ef_relative_change | clinical | 0.006 | -0.006 to 0.023 | 0.008 | No | Higher value -> higher predicted risk |
### Clinical + Endo–Mid

| Plain-language feature | Technical name | Family | AUC decrease | AUC 95% CI | AP decrease | Stable | Model direction |
|---|---|---|---|---|---|---|---|
| Change in the phase pattern of segmental Endo–Mid gap variability | d_tm_sd_gap_dct04 | Endo–Mid | 0.037 | 0.001 to 0.066 | 0.043 | Yes | Higher value -> higher predicted risk |
| Variation across segments in Endo/Mid curve similarity | cur_tm_segment_curve_correlation_std | Endo–Mid | 0.017 | -0.007 to 0.040 | 0.023 | No | Higher value -> lower predicted risk |
| Baseline Mid-wall GLS magnitude | first_mid_gls | clinical | 0.012 | -0.009 to 0.038 | -0.003 | No | Higher value -> higher predicted risk |
| Change in the mean Endo–Mid peak-strain gap | d_tm_peak_gap_mean | Endo–Mid | 0.008 | -0.008 to 0.030 | 0.008 | No | Higher value -> lower predicted risk |
| Current finer phase pattern of segmental Endo–Mid gap variability | cur_tm_sd_gap_dct06 | Endo–Mid | 0.008 | -0.006 to 0.021 | 0.015 | No | Higher value -> lower predicted risk |
### Clinical + variability

| Plain-language feature | Technical name | Family | AUC decrease | AUC 95% CI | AP decrease | Stable | Model direction |
|---|---|---|---|---|---|---|---|
| Baseline Mid-wall GLS magnitude | first_mid_gls | clinical | 0.030 | -0.023 to 0.089 | 0.011 | No | Higher value -> higher predicted risk |
| Robust between-segment dispersion of Mid-wall peak strain | cur_var__mid_peak_abs_robust_sd | variability | 0.010 | -0.014 to 0.036 | -0.001 | No | Higher value -> lower predicted risk |
| Baseline Endocardial GLS magnitude | first_endo_gls | clinical | 0.010 | -0.002 to 0.025 | 0.005 | No | Higher value -> higher predicted risk |
| Robust between-segment dispersion of vendor Endo peak-systolic strain | cur_var__endo_vendor_peak_systolic_abs_robust_sd | variability | 0.007 | -0.029 to 0.041 | 0.015 | No | Higher value -> lower predicted risk |
| Mid-GLS decline slope per 100 days | mid_decline_slope_per_100d | clinical | 0.004 | -0.027 to 0.031 | 0.003 | No | Higher value -> higher predicted risk |
### Combined Extra Trees

| Plain-language feature | Technical name | Family | AUC decrease | AUC 95% CI | AP decrease | Stable | Model direction |
|---|---|---|---|---|---|---|---|
| Overall normalized Endo–Mid curve-shape separation | cur_tm_mean_shape_gap_dct01 | Endo–Mid | 0.017 | 0.008 to 0.026 | 0.020 | Yes | Nonlinear / interaction-dependent |
| Finer phase pattern of normalized Endo–Mid shape separation | cur_tm_mean_shape_gap_dct07 | Endo–Mid | 0.009 | 0.002 to 0.015 | 0.009 | Yes | Nonlinear / interaction-dependent |
| Endocardial between-segment time-to-peak dispersion | cur_var__endo_vendor_time_to_peak_norm_circular_std | variability | 0.008 | 0.003 to 0.014 | 0.007 | Yes | Nonlinear / interaction-dependent |
| Broad current temporal trend in the Endo–Mid curve gap | cur_tm_mean_gap_dct02 | Endo–Mid | 0.006 | -0.006 to 0.017 | 0.008 | No | Nonlinear / interaction-dependent |
| Intermediate phase pattern of normalized Endo–Mid shape separation | cur_tm_mean_shape_gap_dct04 | Endo–Mid | 0.006 | -0.001 to 0.012 | 0.012 | No | Nonlinear / interaction-dependent |

## Feature-family permutation

Here all features from one family are shuffled together. This measures how strongly a
model depends on the family as a whole, including correlated features that can replace
one another.

| model | family | AUC drop | AUC CI low | AUC CI high | AP drop |
|---|---|---|---|---|---|
| Clinical ridge | clinical | 0.103 | -0.013 | 0.230 | 0.050 |
| Clinical + Endo–Mid | clinical | 0.005 | -0.053 | 0.046 | -0.042 |
| Clinical + Endo–Mid | Endo–Mid | 0.064 | -0.012 | 0.136 | 0.069 |
| Clinical + variability | clinical | 0.040 | -0.037 | 0.119 | 0.042 |
| Clinical + variability | variability | 0.018 | -0.041 | 0.079 | 0.007 |
| Combined Extra Trees | clinical | -0.008 | -0.038 | 0.022 | -0.014 |
| Combined Extra Trees | Endo–Mid | 0.054 | -0.006 | 0.115 | 0.032 |
| Combined Extra Trees | variability | 0.016 | -0.009 | 0.047 | 0.001 |

The Endo–Mid family caused a larger AUC decrease than the variability family in the
combined Extra Trees model. This is the clearest family-level support for the Endo–Mid
hypothesis. However, the family-level confidence intervals still included zero.

## Main interpretation

1. **Endo–Mid curve relationships are the most promising engineered signal.** Both the
   sparse model and Extra Trees relied on time-dependent differences between the
   Endocardial and Mid-wall curves.
2. **Inter-segment variability is weaker.** A few amplitude and timing-dispersion
   features contributed modestly, but most were not stable and added little AP.
3. **Baseline GLS is important but partly label-coupled.** This is especially relevant
   for baseline Mid-GLS because the 15% outcome is calculated from it.
4. **DCT features are compact mathematical descriptors, not direct clinical
   measurements.** They should be validated using curve visualizations and an external
   cohort before receiving physiological labels.
5. **Permutation importance is model-specific, not causal.** Correlated features can
   substitute for one another, and Extra Trees can use nonlinear interactions.

## Reproducibility check

The reconstructed models exactly reproduced the saved out-of-fold results:

| model | n | events | roc_auc_reproduced | average_precision_reproduced | prediction_repeats_min | prediction_repeats_max |
|---|---|---|---|---|---|---|
| clinical_ridge | 238.000 | 49.000 | 0.631 | 0.289 | 3.000 | 3.000 |
| clinical_plus_transmural_sparse | 238.000 | 49.000 | 0.644 | 0.309 | 3.000 | 3.000 |
| clinical_plus_variability_sparse | 238.000 | 49.000 | 0.650 | 0.288 | 3.000 | 3.000 |
| combined_extra_trees | 238.000 | 49.000 | 0.672 | 0.313 | 3.000 | 3.000 |

## Output files

- `noncnn_feature_importance_top.csv`: detailed model-specific permutation results.
- `noncnn_feature_family_importance.csv`: family-level permutation results.
- `noncnn_feature_consensus.csv`: cross-model prioritization.
- `noncnn_feature_importance_all.csv`: model-native screening values.
- `top_feature_permutation_importance.png`: top-feature figure.
- `feature_family_permutation_importance.png`: feature-family figure.

The feature calculations are implemented in `cardiotoxicity_early_detection.py` and
`cardiotoxicity_next_visit_gpu.py`.
