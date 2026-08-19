"""
Population weighting and calibration for audience measurement.

Anchors site-centric census counts to Establishment Survey marginals using
raking (iterative proportional fitting, IPF) and post-stratification. These
are the standard survey-statistics estimators used to reconcile a large but
biased census (server/tag data) with an independent probability-based survey.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def post_stratify(df: pd.DataFrame, strata_cols: list[str], targets: dict) -> pd.Series:
    """
    Single-dimension post-stratification weights.

    Each stratum's weight scales the sample so its weighted total matches the
    known population total for that stratum.

    targets: {stratum_value: population_total}
    Returns a weight per row.
    """
    if len(strata_cols) != 1:
        raise ValueError("post_stratify handles a single stratum column; use rake() for joint calibration")
    col = strata_cols[0]
    counts = df.groupby(col).size()
    w = pd.Series(1.0, index=df.index)
    for level, pop_total in targets.items():
        sample_n = counts.get(level, 0)
        if sample_n == 0:
            continue
        w.loc[df[col] == level] = pop_total / sample_n
    return w


def rake(
    df: pd.DataFrame,
    margins: dict[str, dict],
    base_weight: str | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> pd.Series:
    """
    Raking / iterative proportional fitting.

    Calibrates row weights so that, for every calibration variable, the weighted
    marginal totals equal the Establishment Survey population marginals. This is
    the workhorse when the joint population distribution is unknown but the
    marginals (age, gender, region, device, ...) are known from the survey.

    margins: {var_name: {level: population_total, ...}, ...}
    base_weight: optional column of starting weights (e.g. design weights).
    Returns the calibrated weight per row.
    """
    if base_weight is None:
        w = pd.Series(1.0, index=df.index)
    else:
        w = df[base_weight].astype(float).copy()

    # Validate that every margin sums to the same population total.
    pop_totals = {var: sum(levels.values()) for var, levels in margins.items()}
    ref = round(next(iter(pop_totals.values())), 3)
    for var, tot in pop_totals.items():
        if round(tot, 3) != ref:
            raise ValueError(
                f"Margin '{var}' sums to {tot:.1f} but expected {ref:.1f}; "
                "all calibration marginals must describe the same population."
            )

    for iteration in range(max_iter):
        max_rel_change = 0.0
        for var, targets in margins.items():
            weighted = w.groupby(df[var]).sum()
            for level, target in targets.items():
                current = weighted.get(level, 0.0)
                if current <= 0:
                    continue
                factor = target / current
                mask = df[var] == level
                w.loc[mask] *= factor
                max_rel_change = max(max_rel_change, abs(factor - 1.0))
        if max_rel_change < tol:
            break
    return w


def calibration_diagnostics(df: pd.DataFrame, weight: str, margins: dict) -> pd.DataFrame:
    """Return, per calibration cell, the survey target vs the calibrated weighted total."""
    rows = []
    for var, targets in margins.items():
        weighted = df.groupby(var)[weight].sum()
        for level, target in targets.items():
            got = float(weighted.get(level, 0.0))
            rows.append(
                {
                    "variable": var,
                    "level": level,
                    "survey_target": target,
                    "calibrated_total": round(got, 1),
                    "abs_pct_error": round(100 * abs(got - target) / target, 4) if target else np.nan,
                }
            )
    return pd.DataFrame(rows)
