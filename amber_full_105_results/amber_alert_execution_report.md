# Amber deterioration alert: execution report

## Bottom line

The curve CNN did not consistently improve both forecasting error and incident-alert ranking over the clinical ridge.

The best selected-transition GLS MAE was 1.88 strain points (Clinical ridge). The best incident-alert average precision was 0.48 (Engineered ridge).

All performance estimates are exploratory because the full parsed cohort contains 103 patients, with 71 eligible Mid-GLS endpoints and 15 incident events.

## Frozen endpoint

Primary layer: Mid GLS. Each patient is truncated at the first >15% relative reduction from baseline; if no event occurs, the latest qualifying follow-up is used. The selected timeline must contain at least three visits and the prediction-to-outcome interval must be ≤180 days. For non-events with a late final interval, visits are dropped from the end until the latest qualifying endpoint is found.

## Label audit

| layer | n_transitions | incident_events | incident_rate_among_eligible | recovered_by_dropping_late_visits | excluded_patients |
| --- | --- | --- | --- | --- | --- |
| mid | 71 | 15 | 0.211 | 7 | 32 |
| endo | 75 | 17 | 0.227 | 8 | 28 |

## Patient-held-out selected-transition performance

| model | n_transitions | n_incident_events | mae_next_gls | mae_ci_low | mae_ci_high | average_precision | average_precision_ci_low | average_precision_ci_high | brier_score | sensitivity_at_p50 | precision_at_p50 | alerts_at_p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Persistence | 71 | 15 | 1.950 | 1.664 | 2.253 | 0.295 | 0.168 | 0.527 | 0.166 | 0.000 | 0.000 | 0 |
| Clinical ridge | 71 | 15 | 1.882 | 1.603 | 2.201 | 0.390 | 0.199 | 0.676 | 0.162 | 0.000 | 0.000 | 0 |
| Engineered ridge | 71 | 15 | 1.914 | 1.615 | 2.223 | 0.482 | 0.244 | 0.721 | 0.158 | 0.000 | 0.000 | 0 |
| Endo–Mid curve CNN | 71 | 15 | 1.903 | 1.559 | 2.302 | 0.320 | 0.170 | 0.565 | 0.166 | 0.000 | 0.000 | 2 |

At a descriptive 20% alert budget, the following table shows how many first events were captured among the highest-risk selected patients:

| model | sensitivity_top20pct | precision_top20pct | alerts_top20pct |
| --- | --- | --- | --- |
| Persistence | 0.267 | 0.267 | 15 |
| Clinical ridge | 0.533 | 0.533 | 15 |
| Engineered ridge | 0.467 | 0.467 | 15 |
| Endo–Mid curve CNN | 0.467 | 0.467 | 15 |

## Endo-target sensitivity analysis

| model | n_transitions | n_incident_events | mae_next_gls | average_precision | brier_score | sensitivity_top20pct | precision_top20pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Persistence | 75 | 17 | 2.219 | 0.486 | 0.157 | 0.412 | 0.467 |
| Clinical ridge | 75 | 17 | 2.065 | 0.438 | 0.163 | 0.471 | 0.533 |
| Engineered ridge | 75 | 17 | 2.107 | 0.432 | 0.166 | 0.471 | 0.533 |
| Endo–Mid curve CNN | 75 | 17 | 2.121 | 0.388 | 0.166 | 0.353 | 0.400 |

## Model implementation

The Endo–Mid model used a shared three-view 1D CNN with current and previous curve embeddings, explicit trajectory differences, and a scalar branch. Each CNN contained 18643 trainable parameters. Three seeded CNN fits were averaged within each fold to reduce small-sample training instability. A Student-t head predicted next GLS and uncertainty; a masked auxiliary head predicted EF.

## Interpretation rule

A complex curve model should be retained only if it improves out-of-patient next-GLS error and alert ranking over the clinical trajectory model. Threshold-at-0.5 sensitivity and precision are shown descriptively; the operating probability cutoff must be chosen in a larger development cohort.
