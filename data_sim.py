"""
Synthetic data generator: an Establishment Survey and a site-centric census.

This stands in for the two real inputs of an audience-measurement project:

  1. Establishment Survey - an independent probability sample of the population,
     carrying the true population marginals (age, gender, region) and the
     device-multiplicity behaviour (devices per person). This is the source of
     truth the census is calibrated to.

  2. Site-centric census - identifier-level rows (one per device/platform) with
     self-declared or modelled demographics that are deliberately BIASED
     (younger, more urban, over-counted because of multi-device use). This is
     what raw tag/server measurement gives you before calibration.

Both are generated from a known ground truth so the pipeline's accuracy can be
checked against a number we actually know.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
GENDERS = ["F", "M"]
REGIONS = ["North", "Central", "South", "Coast"]
PLATFORMS = ["web_desktop", "web_mobile", "app_ios", "app_android"]

# Ground-truth population (persons) by age group - the number Reach is a share of.
POP_TOTAL = 3_400_000
POP_AGE_SHARE = np.array([0.11, 0.19, 0.18, 0.17, 0.16, 0.19])
POP_GENDER_SHARE = np.array([0.51, 0.49])
POP_REGION_SHARE = np.array([0.24, 0.31, 0.28, 0.17])


def make_establishment_survey(n: int = 6000, seed: int = 7) -> pd.DataFrame:
    """Probability sample reproducing the population marginals + device behaviour."""
    rng = np.random.default_rng(seed)
    age = rng.choice(AGE_GROUPS, size=n, p=POP_AGE_SHARE)
    gender = rng.choice(GENDERS, size=n, p=POP_GENDER_SHARE)
    region = rng.choice(REGIONS, size=n, p=POP_REGION_SHARE)

    # Devices/platforms per person falls with age (older = fewer devices).
    age_idx = {g: i for i, g in enumerate(AGE_GROUPS)}
    base = np.array([2.9, 2.7, 2.4, 2.1, 1.8, 1.5])
    lam = base[[age_idx[a] for a in age]]
    devices = np.clip(rng.poisson(lam) , 1, 4)

    # Design weight so the sample expands to POP_TOTAL.
    design_weight = np.full(n, POP_TOTAL / n)
    return pd.DataFrame(
        {
            "person_id": np.arange(n),
            "age_group": age,
            "gender": gender,
            "region": region,
            "devices_per_person": devices,
            "design_weight": design_weight,
        }
    )


def make_site_centric_census(n_persons: int = 80_000, seed: int = 11) -> pd.DataFrame:
    """
    Identifier-level census with multi-device inflation and a young/urban skew.

    Returns one row per device-identifier, tagged with the true person_id (so
    deterministic dedup is possible in this demo) and biased demographics.
    """
    rng = np.random.default_rng(seed)

    # BIASED audience: over-represents 18-34 and Central/Coast vs the population.
    biased_age_share = np.array([0.20, 0.27, 0.19, 0.15, 0.11, 0.08])
    biased_region_share = np.array([0.18, 0.34, 0.24, 0.24])

    age = rng.choice(AGE_GROUPS, size=n_persons, p=biased_age_share)
    gender = rng.choice(GENDERS, size=n_persons, p=[0.47, 0.53])
    region = rng.choice(REGIONS, size=n_persons, p=biased_region_share)

    age_idx = {g: i for i, g in enumerate(AGE_GROUPS)}
    base = np.array([2.9, 2.7, 2.4, 2.1, 1.8, 1.5])
    lam = base[[age_idx[a] for a in age]]
    devices = np.clip(rng.poisson(lam), 1, 4)

    rows = []
    for pid in range(n_persons):
        d = devices[pid]
        chosen = rng.choice(PLATFORMS, size=d, replace=False)
        for plat in chosen:
            rows.append((pid, age[pid], gender[pid], region[pid], plat))
    census = pd.DataFrame(rows, columns=["person_id", "age_group", "gender", "region", "platform"])
    return census
