# Audience Measurement Calibration

A compact, runnable reference implementation of the survey-statistics pipeline
behind digital **audience measurement**: reconcile a large but biased
**site-centric census** (tags / server logs / device identifiers) with an
independent **Establishment Survey** to produce reliable **Reach**,
**Real Users**, **demographic profiles**, and **cross-device / cross-platform**
estimates.

Everything runs on synthetic data generated from a known ground truth, so every
output can be checked against a number we actually control.

```
python main.py
```

Runs in ~3 seconds, no external services, no API keys.

---

## What it does

```
  Establishment Survey                    Site-centric census
  (probability sample)                    (device identifiers)
  - population marginals                  - biased: young / urban skew
    age x gender x region                 - multi-device over-count
  - devices-per-person behaviour          - one row per device/platform
          |                                        |
          |                                        v
          |                            1. Cross-device / platform DEDUP
          |                               deterministic (person key) OR
          |                               probabilistic (devices/person
          |                               from the survey)
          |                                        |
          |                                        v
          +----------- targets --------> 2. RAKING / calibration (IPF)
                                            weight persons so weighted
                                            marginals == survey marginals
                                                     |
                                                     v
                                         3. AUDIENCE INDICATORS
                                            Reach, Real Users,
                                            demographic profile,
                                            bootstrap SE
                                                     |
                                                     v
                                         4. CALIBRATION QA
                                            target vs calibrated,
                                            per cell (must be ~0%)
```

## Methods implemented

| Step | Module | Method |
|------|--------|--------|
| Deduplication | `dedup.py` | Deterministic collapse on a person/household key; probabilistic deflation by devices-per-person estimated from the survey |
| Calibration | `calibration.py` | Raking (iterative proportional fitting) and single-dimension post-stratification, with a convergence check and per-cell diagnostics |
| Estimation | `estimators.py` | Reach (% of population), Real Users (weighted unique persons), weighted demographic profiles, bootstrap standard error |
| Data | `data_sim.py` | Establishment Survey + biased multi-device census from a known ground truth |

## Sample output

```
Site-centric census  : 186,652 device identifiers (80,000 true persons)
Raw identifier count (naive 'users')      : 186,652
Devices per person (from survey)          : 2.232
Probabilistic unique persons (raw/dpp)    : 83,619
Deterministic unique persons (dedup key)  : 80,000
Real Users (calibrated, deduplicated)     : 80,000
Reach                                     : 2.35%  (bootstrap SE ~0.00 pp)

Bias correction (age composition):
       raw_census_%  calibrated_%  population_%
18-24         20.01         11.15         11.15
25-34         26.98         19.75         19.75
...
65+            8.02         19.27         19.27      <- census under-counts 65+,
                                                       raking restores it
```

The raw census over-represents 18-34 and under-represents 65+; after raking to
the survey marginals the calibrated composition matches the population, and the
calibration QA reports a maximum per-cell error of 0.0000%.

## Why this design

- **Marginals, not the joint.** An Establishment Survey delivers reliable
  *marginal* distributions (age, gender, region, device). Raking is the standard
  estimator that hits every marginal without needing the unknown joint
  distribution.
- **Devices are not people.** Site-centric data counts identifiers. Reach and
  Real Users must be person-level, so deduplication comes *before* calibration.
- **Auditable.** Every wave reruns identically and the QA table proves the
  calibrated totals equal the survey targets.

## Install

```
pip install -r requirements.txt   # numpy, pandas
python main.py
```

## Adapting to real data

Replace `data_sim.py` with two loaders:

1. `load_survey()` returning respondent rows with `age_group, gender, region,
   devices_per_person, design_weight`.
2. `load_census()` returning identifier rows with `person_id` (from panel/login
   linkage) or without it (then use the probabilistic path), plus the same
   demographic columns and a `platform` column.

The calibration variables and targets are declared in one place (`survey_margins`
in `main.py`); add region x age interactions or device/platform marginals there.

---

Author: **Dr. Sandeep Grover** - epidemiologist / biostatistician. Survey
methodology, population weighting and calibration, sampling and stratification.
