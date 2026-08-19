"""
Audience indicators derived from calibrated, deduplicated data.

  - Reach            : share of the population that used the property (%)
  - Real Users       : estimated count of unique persons (deduplicated)
  - Demographic      : weighted composition of the audience by age/gender/region
  - Variance         : linearized / bootstrap standard errors for the estimates
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reach(real_users: float, population_total: float) -> float:
    """Reach as a percentage of the population."""
    if population_total <= 0:
        raise ValueError("population_total must be positive")
    return 100.0 * real_users / population_total


def demographic_profile(person_df: pd.DataFrame, weight: str, by: str) -> pd.DataFrame:
    """Weighted audience composition by a demographic variable (shares sum to 100)."""
    grp = person_df.groupby(by)[weight].sum()
    prof = (100 * grp / grp.sum()).round(2).rename("audience_pct").reset_index()
    return prof.sort_values(by).reset_index(drop=True)


def real_users_by_cell(person_df: pd.DataFrame, weight: str, by: list[str]) -> pd.DataFrame:
    """Estimated unique persons per demographic cell (weighted)."""
    out = person_df.groupby(by)[weight].sum().round(0).rename("real_users").reset_index()
    return out


def bootstrap_reach_se(
    person_df: pd.DataFrame,
    weight: str,
    population_total: float,
    n_boot: int = 500,
    seed: int = 42,
) -> float:
    """
    Bootstrap standard error for Reach.

    Resamples persons with replacement, recomputes the weighted person total and
    the implied Reach, and returns the standard deviation across replicates.
    """
    rng = np.random.default_rng(seed)
    w = person_df[weight].to_numpy()
    n = len(w)
    reaches = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Resampled weighted person total -> implied Reach for this replicate.
        reaches[b] = 100.0 * w[idx].sum() / population_total
    return float(np.std(reaches, ddof=1))
