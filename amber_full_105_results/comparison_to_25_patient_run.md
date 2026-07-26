# Full-cohort comparison with the previous 25-eligible-patient run

## Cohort and endpoint

The new folder contained 416 valid AutoStrainCap exports. All parsed successfully into 400 true visits from 103 unique anonymized patient IDs. Although the expected count was 105, only 103 unique patient identifiers are present in the supplied files.

Using the unchanged primary endpoint (Mid GLS, first >15% relative decline from baseline, minimum three visits, next visit within 180 days):

| Quantity | Previous run | Full cohort | Change |
| --- | ---: | ---: | ---: |
| Parsed patients | 33 | 103 | +70 |
| All transitions | 95 | 297 | +202 |
| Transitions within 180 days | 85 | 262 | +177 |
| Eligible Mid-GLS endpoints | 25 | 71 | +46 |
| Incident Mid-GLS events | 7 | 15 | +8 |
| Event prevalence | 28.0% | 21.1% | -6.9 percentage points |

Of the 32 patients excluded from the primary Mid endpoint, 25 crossed the deterioration threshold before the third visit, three had no qualifying three-visit endpoint, three had no visit interval within 180 days, and one had a first-crossing interval longer than 180 days. Seven eligible non-event timelines were recovered by dropping late trailing visits.

## Patient-held-out Mid-GLS results

| Model | Previous MAE | Full MAE | Previous AP | Full AP | Previous Brier | Full Brier | Full top-20% sensitivity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | 1.849 | 1.950 | 0.496 | 0.295 | 0.186 | 0.166 | 0.267 |
| Clinical ridge | 1.873 | **1.882** | 0.640 | 0.390 | 0.169 | 0.162 | **0.533** |
| Engineered ridge | 1.892 | 1.914 | **0.699** | **0.482** | 0.155 | **0.158** | 0.467 |
| Endo-Mid curve CNN | 2.237 | 1.903 | 0.678 | 0.320 | 0.154 | 0.166 | 0.467 |

Average precision is prevalence-dependent. Relative to prevalence, the engineered ridge retained the strongest ranking lift in the full cohort (2.28x prevalence), followed by clinical ridge (1.85x), CNN (1.51x), and persistence (1.40x).

## Interpretation

The larger cohort materially improves the credibility of the comparison but does not reverse the earlier model-selection conclusion. The CNN's continuous MAE improved substantially (2.237 to 1.903) and is now close to both ridge models, suggesting that it benefited from more training patients. However, its amber-event ranking fell to AP 0.320 and did not beat either ridge model. The engineered ridge remains the best alert-ranking model, while clinical ridge has the best continuous Mid-GLS MAE and captures the most events at the fixed 20% alert budget.

The full-cohort results therefore favor the classical approaches. There is not yet evidence that raw Endo-Mid curve morphology adds reliable out-of-patient alert value beyond clinical trajectory and engineered curve-summary features.

All estimates remain exploratory: only 15 primary incident events are available, confidence intervals overlap, and the operating threshold should not be clinically fixed from this dataset alone.
