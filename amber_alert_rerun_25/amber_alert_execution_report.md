# Amber deterioration alert: execution report

## Bottom line

The curve CNN did not consistently improve both forecasting error and incident-alert ranking over the clinical ridge.

The best selected-transition GLS MAE was 1.85 strain points (Persistence). The best incident-alert average precision was 0.70 (Engineered ridge).

All performance estimates are exploratory because the independent sample contains only 33 patients and few incident events.

## Frozen endpoint

Primary layer: Mid GLS. Each patient is truncated at the first >15% relative reduction from baseline; if no event occurs, the latest qualifying follow-up is used. The selected timeline must contain at least three visits and the prediction-to-outcome interval must be ≤180 days. For non-events with a late final interval, visits are dropped from the end until the latest qualifying endpoint is found.

## Label audit

| layer | n_transitions | incident_events | incident_rate_among_eligible | recovered_by_dropping_late_visits | excluded_patients |
| --- | --- | --- | --- | --- | --- |
| mid | 25 | 7 | 0.280 | 3 | 8 |
| endo | 26 | 8 | 0.308 | 3 | 7 |

## Patient-held-out selected-transition performance

| model | n_transitions | n_incident_events | mae_next_gls | mae_ci_low | mae_ci_high | average_precision | average_precision_ci_low | average_precision_ci_high | brier_score | sensitivity_at_p50 | precision_at_p50 | alerts_at_p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Persistence | 25 | 7 | 1.849 | 1.375 | 2.338 | 0.496 | 0.212 | 0.878 | 0.186 | 0.000 | 0.000 | 0 |
| Clinical ridge | 25 | 7 | 1.873 | 1.454 | 2.300 | 0.640 | 0.358 | 0.981 | 0.169 | 0.000 | 0.000 | 1 |
| Engineered ridge | 25 | 7 | 1.892 | 1.414 | 2.356 | 0.699 | 0.359 | 0.967 | 0.155 | 0.286 | 1.000 | 2 |
| Endo–Mid curve CNN | 25 | 7 | 2.237 | 1.635 | 2.934 | 0.678 | 0.337 | 0.976 | 0.154 | 0.143 | 0.500 | 2 |

At a descriptive 20% alert budget, the following table shows how many first events were captured among the highest-risk selected patients:

| model | sensitivity_top20pct | precision_top20pct | alerts_top20pct |
| --- | --- | --- | --- |
| Persistence | 0.286 | 0.400 | 5 |
| Clinical ridge | 0.571 | 0.800 | 5 |
| Engineered ridge | 0.429 | 0.600 | 5 |
| Endo–Mid curve CNN | 0.571 | 0.800 | 5 |

## Endo-target sensitivity analysis

| model | n_transitions | n_incident_events | mae_next_gls | average_precision | brier_score | sensitivity_top20pct | precision_top20pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Persistence | 26 | 8 | 2.165 | 0.722 | 0.180 | 0.500 | 0.667 |
| Clinical ridge | 26 | 8 | 2.313 | 0.751 | 0.181 | 0.375 | 0.500 |
| Engineered ridge | 26 | 8 | 2.455 | 0.676 | 0.197 | 0.375 | 0.500 |
| Endo–Mid curve CNN | 26 | 8 | 2.569 | 0.609 | 0.197 | 0.500 | 0.667 |

## Model implementation

The Endo–Mid model used a shared three-view 1D CNN with current and previous curve embeddings, explicit trajectory differences, and a scalar branch. Each CNN contained 18643 trainable parameters. Three seeded CNN fits were averaged within each fold to reduce small-sample training instability. A Student-t head predicted next GLS and uncertainty; a masked auxiliary head predicted EF.

## Interpretation rule

A complex curve model should be retained only if it improves out-of-patient next-GLS error and alert ranking over the clinical trajectory model. Threshold-at-0.5 sensitivity and precision are shown descriptively; the operating probability cutoff must be chosen in a larger development cohort.