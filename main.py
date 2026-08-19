"""
Audience measurement calibration - end-to-end demo.

Pipeline:
  1. Load the Establishment Survey (population marginals + device behaviour)
     and the site-centric census (biased, multi-device identifier counts).
  2. Deduplicate the census to persons (cross-device / cross-platform).
  3. Rake the person-level census to the survey marginals (age, gender, region).
  4. Produce Reach, Real Users, and demographic profiles, with a bootstrap SE.
  5. Check calibrated marginals against survey targets.

Run:  python main.py
Everything is synthetic and generated from a known ground truth, so the output
can be compared against numbers we actually control.

Author: Dr. Sandeep Grover
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_sim import make_establishment_survey, make_site_centric_census, POP_TOTAL
from calibration import rake, calibration_diagnostics
from dedup import deterministic_dedup, estimate_devices_per_person, probabilistic_dedup_total
from estimators import reach, demographic_profile, real_users_by_cell, bootstrap_reach_se

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def survey_margins(survey: pd.DataFrame) -> dict:
    """Population marginals (targets) from the Establishment Survey design weights."""
    margins = {}
    for var in ["age_group", "gender", "region"]:
        margins[var] = survey.groupby(var)["design_weight"].sum().to_dict()
    return margins


def main() -> None:
    print("=" * 72)
    print("AUDIENCE MEASUREMENT CALIBRATION  |  Establishment Survey + site-centric")
    print("=" * 72)

    # --- 1. Inputs -------------------------------------------------------
    survey = make_establishment_survey()
    census = make_site_centric_census()
    print(f"\nEstablishment Survey : {len(survey):,} respondents -> expands to {POP_TOTAL:,} persons")
    print(f"Site-centric census  : {len(census):,} device identifiers "
          f"({census['person_id'].nunique():,} true persons)")

    raw_ids = len(census)
    print(f"\nRaw identifier count (naive 'users')      : {raw_ids:,}")

    # --- 2. Cross-device / cross-platform dedup --------------------------
    # (a) probabilistic: deflate by devices/person from the survey (no key needed)
    dpp = estimate_devices_per_person(survey, "devices_per_person", weight="design_weight")
    prob_persons = probabilistic_dedup_total(raw_ids, dpp)
    print(f"Devices per person (from survey)          : {dpp:.3f}")
    print(f"Probabilistic unique persons (raw/dpp)    : {prob_persons:,.0f}")

    # (b) deterministic: collapse on the person key (panel/login linkage)
    persons = deterministic_dedup(census, person_key="person_id")
    print(f"Deterministic unique persons (dedup key)  : {len(persons):,}")

    # --- 3. Calibrate the person-level census to survey marginals --------
    # Give every deduplicated person an equal starting weight that scales the
    # census sample up to the census's own reach, then rake to survey marginals.
    persons = persons.copy()
    # Starting weight: assume the census persons are a sample of the reachable
    # audience; scale so the deterministic person count is preserved pre-raking.
    persons["w0"] = 1.0
    margins = survey_margins(survey)

    # Reach target: what fraction of the population the census implies, corrected
    # to the survey's device behaviour. We rake the census demographics to the
    # POPULATION SHAPE but hold the audience SIZE at the deduplicated estimate.
    audience_size = len(persons)
    # Rescale survey margins to the audience size (calibrate composition, keep size).
    audience_margins = {
        var: {lvl: share * audience_size for lvl, share in
              (pd.Series(tgt) / sum(tgt.values())).to_dict().items()}
        for var, tgt in margins.items()
    }
    persons["cal_weight"] = rake(persons, audience_margins, base_weight="w0")

    # --- 4. Audience indicators -----------------------------------------
    real_users = float(persons["cal_weight"].sum())
    r = reach(real_users, POP_TOTAL)
    se = bootstrap_reach_se(persons, "cal_weight", POP_TOTAL, n_boot=300)

    print("\n" + "-" * 72)
    print("AUDIENCE INDICATORS")
    print("-" * 72)
    print(f"Real Users (calibrated, deduplicated)     : {real_users:,.0f}")
    print(f"Reach                                     : {r:.2f}%  (bootstrap SE {se:.2f} pp)")

    print("\nAudience profile by age (calibrated %):")
    print(demographic_profile(persons, "cal_weight", "age_group").to_string(index=False))

    print("\nAudience profile by region (calibrated %):")
    print(demographic_profile(persons, "cal_weight", "region").to_string(index=False))

    # --- 5. Calibration QA ----------------------------------------------
    diag = calibration_diagnostics(persons, "cal_weight", audience_margins)
    max_err = diag["abs_pct_error"].max()
    print("\n" + "-" * 72)
    print(f"CALIBRATION QA  |  max abs marginal error after raking: {max_err:.4f}%")
    print("-" * 72)
    print(diag.to_string(index=False))

    # --- 6. Before/after bias illustration ------------------------------
    raw_age = (census.drop_duplicates("person_id")
               .groupby("age_group").size())
    raw_age_pct = (100 * raw_age / raw_age.sum()).round(2)
    cal_age = demographic_profile(persons, "cal_weight", "age_group").set_index("age_group")["audience_pct"]
    pop_age = (pd.Series(margins["age_group"]) / sum(margins["age_group"].values()) * 100).round(2)
    compare = pd.DataFrame({
        "raw_census_%": raw_age_pct,
        "calibrated_%": cal_age,
        "population_%": pop_age,
    })
    print("\nBias correction (age composition):")
    print(compare.to_string())

    print("\nDone. All figures derive from the synthetic ground truth in data_sim.py.")


if __name__ == "__main__":
    main()
