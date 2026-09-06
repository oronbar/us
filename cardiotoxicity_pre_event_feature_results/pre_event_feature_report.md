# Feature behavior before first 15% relative Mid-GLS deterioration

## Question and design

The event is the first visit with at least a 15% relative drop in Mid-GLS magnitude from the first visit. For every event patient, the visit immediately before that event is the positive pre-event observation. Feature ranking never uses the event visit itself.

Each feature was evaluated alone using ROC AUC. AUCs were calculated separately within current visit order and averaged, so ordinary differences between visit 1, visit 2, and visit 3 do not create a false signal. Confidence intervals use patient-level bootstrap resampling.

## Cohort

- Eligible transitions: 238
- First-deterioration events: 49 in 49 patients
- Non-event eligible transitions: 189
- Event visit distribution: V2: 26, V3: 12, V4: 11
- Later events with a genuine V(t-2) to V(t-1) comparison: 23

**Key limitation:** events at visit 2 have only baseline as the one-visit-early observation. They can test baseline risk markers, but they cannot show an evolving within-patient warning signal.

## Strongest one-visit-early features: all 49 events

| rank | feature_name | family | Direction | AUC (95% CI) | Cases |
| --- | --- | --- | --- | --- | --- |
| 3 | Baseline Endocardial GLS magnitude | Clinical/trajectory | higher | 0.661 (0.573-0.741) | 49 |
| 7 | Between-segment dispersion of vendor Endocardial peak-systolic strain | Segment variability | lower | 0.642 (0.549-0.718) | 49 |
| 8 | Between-segment dispersion of Mid-wall peak strain | Segment variability | lower | 0.641 (0.550-0.721) | 49 |
| 9 | Current finer phase pattern of segmental layer-gap variability | Endo-Mid | lower | 0.626 (0.536-0.712) | 49 |
| 10 | Between-segment dispersion of vendor Mid-wall peak-systolic strain | Segment variability | lower | 0.618 (0.535-0.698) | 49 |
| 11 | Fraction of Endocardial segments with strain magnitude below 15 | Segment variability | lower | 0.614 (0.537-0.702) | 49 |
| 12 | Between-segment dispersion of Endocardial peak strain | Segment variability | lower | 0.610 (0.521-0.705) | 49 |
| 13 | Intermediate phase pattern of normalized layer-shape separation | Endo-Mid | lower | 0.608 (0.527-0.690) | 49 |
| 14 | Fraction of Mid-wall segments with strain magnitude below 15 | Segment variability | lower | 0.606 (0.519-0.694) | 49 |
| 15 | Current Endocardial GLS magnitude | Clinical/trajectory | higher | 0.604 (0.515-0.694) | 49 |

Outcome-coupled Mid-GLS features were flagged in the CSV because they are mathematically related to the label definition. The table above prioritizes the remaining features for biological interpretation.

## Sensitivity analysis: 23 events at visit 3 or 4

| rank | feature_name | family | Direction | AUC (95% CI) | Cases |
| --- | --- | --- | --- | --- | --- |
| 1 | Current endo decline from roll3 | Clinical/trajectory | higher | 0.725 (0.540-0.874) | 11 |
| 3 | Mean within-ring dispersion of Mid-wall peak strain | Segment variability | lower | 0.697 (0.584-0.802) | 23 |
| 4 | Change in phase pattern of segmental layer-gap variability | Endo-Mid | higher | 0.693 (0.591-0.799) | 23 |
| 5 | Current relative Endocardial GLS decline from baseline | Clinical/trajectory | higher | 0.691 (0.576-0.800) | 23 |
| 6 | Between-segment dispersion of Endocardial peak strain | Segment variability | lower | 0.691 (0.564-0.803) | 23 |
| 7 | Between-segment dispersion of Mid-wall peak strain | Segment variability | lower | 0.690 (0.572-0.800) | 23 |
| 8 | Mean within-view dispersion of Mid-wall peak strain | Segment variability | lower | 0.686 (0.568-0.795) | 23 |
| 9 | Current phase pattern of segmental layer-gap variability (DCT 4) | Endo-Mid | higher | 0.679 (0.567-0.785) | 23 |
| 11 | Current finer phase pattern of segmental layer-gap variability | Endo-Mid | lower | 0.675 (0.551-0.799) | 23 |
| 12 | Across-segment variation in Endocardial/Mid peak-strain ratio | Endo-Mid | lower | 0.675 (0.565-0.771) | 23 |

This smaller analysis is the better test of an evolving warning signal, but its confidence intervals are wider.

## Largest within-patient changes from two visits before to one visit before

| rank | feature_name | family | patients | Median change | Absolute change / IQR | FDR q |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Current Endocardial-Mid: Vendor time-to-peak gap median | Endo-Mid | 23 | 0 | 1.99 | 0.901 |
| 2 | Finer temporal pattern of normalized layer-shape separation | Endo-Mid | 23 | -0.006643 | 1.05 | 0.681 |
| 3 | Current Endocardial-Mid: Fraction time-to-peak discordant gt 5pct cycle | Endo-Mid | 23 | 0 | 1.00 | 0.871 |
| 4 | Current: Endocardial post systolic fraction | Segment variability | 23 | 0 | 1.00 | 0.901 |
| 5 | Current Endocardial-Mid: Time-to-peak gap p90 p10 | Endo-Mid | 23 | 0.001115 | 0.99 | 0.901 |
| 6 | Between-segment dispersion of Mid-wall peak strain | Segment variability | 23 | -1.016 | 0.92 | 0.502 |
| 7 | Current Endocardial-Mid: Peak ratio robust dispersion | Endo-Mid | 23 | -0.002164 | 0.82 | 0.926 |
| 8 | Current Endocardial-Mid: Sd gap dct coefficient 01 | Endo-Mid | 23 | -0.4 | 0.81 | 0.896 |
| 9 | Current Endocardial-Mid: Mean gap dct coefficient 08 | Endo-Mid | 23 | 0.1276 | 0.80 | 0.901 |
| 10 | Current Endocardial-Mid: Segment curve correlation iqr | Endo-Mid | 23 | 0.0004459 | 0.79 | 0.864 |

This within-patient ranking measures amount of change, not predictive accuracy. A feature can change strongly yet still be noisy across patients.
None of the paired changes survived false-discovery-rate correction (minimum q=0.467). Therefore, the apparent trajectories are descriptive, not confirmed group-wide trends.

## Interpretation

A feature is a plausible one-visit-early marker only when it has useful AUC, a reasonably stable bootstrap interval, and a coherent trajectory before the event. These results are exploratory because many features were screened on the same cohort; the strongest candidates should be tested in patient-held-out modeling or an external cohort.

Trajectory figure features: Mean within-ring dispersion of Mid-wall peak strain; Change in phase pattern of segmental layer-gap variability; Between-segment dispersion of Endocardial peak strain; Between-segment dispersion of Mid-wall peak strain; Mean within-view dispersion of Mid-wall peak strain; Current phase pattern of segmental layer-gap variability (DCT 4).