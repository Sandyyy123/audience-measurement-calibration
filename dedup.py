"""
Cross-device / cross-platform deduplication.

Site-centric measurement counts identifiers (cookies, app instances, device
IDs), not people. One person shows up on phone + laptop + tablet, inflating raw
"users". Deduplication collapses identifiers to persons so Reach and Real Users
reflect unique individuals, not devices.

Two strategies are provided:
  1. deterministic_dedup  - collapse on a known person/household key (panel or
     login-based linkage).
  2. probabilistic_devices_per_person - when no key exists, correct the census
     count using a device-multiplicity distribution estimated from the
     Establishment Survey (how many devices/platforms a person uses).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _modal_per_group(df: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    """
    Vectorized modal value per group for each column (no per-group Python lambda).

    For each column we count (key, value) pairs, sort by frequency, and keep the
    most frequent value per key. This is the robust choice when a person's
    devices carry slightly different modelled demographics; it runs in a few
    pandas/C passes rather than one Python call per group.
    """
    out = pd.DataFrame({key: df[key].unique()}).set_index(key)
    for c in cols:
        counts = df.groupby([key, c], observed=True).size().reset_index(name="_n")
        top = counts.sort_values([key, "_n"]).drop_duplicates(key, keep="last")
        out[c] = top.set_index(key)[c]
    return out


def deterministic_dedup(df: pd.DataFrame, person_key: str, weight: str | None = None) -> pd.DataFrame:
    """
    Collapse device-level rows to one row per person.

    Keeps the modal demographics per person and averages the weight so the
    population total is preserved. Returns a person-level frame.
    """
    cols = [c for c in df.columns if c not in (person_key, weight)]
    person = _modal_per_group(df, person_key, cols)
    if weight is not None:
        person[weight] = df.groupby(person_key)[weight].mean()
    return person.reset_index()


def estimate_devices_per_person(survey: pd.DataFrame, device_count_col: str, weight: str | None = None) -> float:
    """
    Mean number of devices/platforms per person, from the Establishment Survey.

    This is the deflation factor applied to a raw census count that cannot be
    person-linked: unique_persons ~= raw_identifiers / mean_devices_per_person.
    """
    if weight is None:
        return float(survey[device_count_col].mean())
    w = survey[weight]
    return float((survey[device_count_col] * w).sum() / w.sum())


def probabilistic_dedup_total(raw_identifier_count: float, devices_per_person: float) -> float:
    """Deflate a raw identifier count to an estimated unique-person count."""
    if devices_per_person <= 0:
        raise ValueError("devices_per_person must be positive")
    return raw_identifier_count / devices_per_person
