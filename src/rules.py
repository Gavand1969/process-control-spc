"""
rules.py
========
Western Electric / Nelson Rules for Out-of-Control Detection

Implements the four classic Western Electric rules and four additional run-pattern
checks. The rule numbers used by this toolkit are listed below.

Rule definitions (applied to X-bar chart subgroup means):
  Rule 1: 1 point beyond ±3σ (Zone A)
  Rule 2: 2 of 3 consecutive points beyond ±2σ (same side)
  Rule 3: 4 of 5 consecutive points beyond ±1σ (same side)
  Rule 4: 8 consecutive points on the same side of the centerline
  Rule 5: 6 consecutive points trending up or down (run of 6)
  Rule 6: 15 consecutive points within ±1σ (hugging centerline — stratification)
  Rule 7: 14 consecutive points alternating up-down
  Rule 8: 8 consecutive points beyond ±1σ on both sides (mixture pattern)

Usage:
    from src.rules import detect_violations
    violations = detect_violations(xbar_values, center, ucl, lcl)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict


def _zones(values: np.ndarray, center: float, sigma: float) -> np.ndarray:
    """
    Assign each value to a signed zone integer:
      +3 / -3  beyond 3σ   (Zone A outside)
      +2 / -2  between 2σ and 3σ  (Zone A)
      +1 / -1  between 1σ and 2σ  (Zone B)
       0       within 1σ          (Zone C)
    """
    zones = np.zeros(len(values), dtype=int)
    for i, v in enumerate(values):
        deviation = v - center
        if abs(deviation) > 3 * sigma:
            zones[i] = 3 if deviation > 0 else -3
        elif abs(deviation) > 2 * sigma:
            zones[i] = 2 if deviation > 0 else -2
        elif abs(deviation) > sigma:
            zones[i] = 1 if deviation > 0 else -1
        else:
            zones[i] = 0
    return zones


def rule1(values: np.ndarray, center: float, sigma: float) -> List[int]:
    """1 point beyond ±3σ."""
    return [i for i, v in enumerate(values) if abs(v - center) > 3 * sigma]


def rule2(values: np.ndarray, center: float, sigma: float) -> List[int]:
    """2 of 3 consecutive points beyond 2σ on the same side."""
    violations = []
    n = len(values)
    for i in range(2, n):
        window = values[i - 2 : i + 1]
        above = sum(1 for v in window if v > center + 2 * sigma)
        below = sum(1 for v in window if v < center - 2 * sigma)
        if above >= 2 or below >= 2:
            violations.append(i)
    return violations


def rule3(values: np.ndarray, center: float, sigma: float) -> List[int]:
    """4 of 5 consecutive points beyond 1σ on the same side."""
    violations = []
    n = len(values)
    for i in range(4, n):
        window = values[i - 4 : i + 1]
        above = sum(1 for v in window if v > center + sigma)
        below = sum(1 for v in window if v < center - sigma)
        if above >= 4 or below >= 4:
            violations.append(i)
    return violations


def rule4(values: np.ndarray, center: float, sigma: float) -> List[int]:
    """8 consecutive points on the same side of the centerline."""
    violations = []
    n = len(values)
    for i in range(7, n):
        window = values[i - 7 : i + 1]
        above = all(v > center for v in window)
        below = all(v < center for v in window)
        if above or below:
            violations.append(i)
    return violations


def rule5(values: np.ndarray, center: float, sigma: float) -> List[int]:
    """6 consecutive points steadily increasing or decreasing."""
    violations = []
    n = len(values)
    for i in range(5, n):
        window = values[i - 5 : i + 1]
        diffs = np.diff(window)
        if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
            violations.append(i)
    return violations


def rule6(values: np.ndarray, center: float, sigma: float) -> List[int]:
    """15 consecutive points within ±1σ (hugging / stratification)."""
    violations = []
    n = len(values)
    for i in range(14, n):
        window = values[i - 14 : i + 1]
        if all(abs(v - center) < sigma for v in window):
            violations.append(i)
    return violations


def rule7(values: np.ndarray, center: float, sigma: float) -> List[int]:
    """14 consecutive points alternating up and down."""
    violations = []
    n = len(values)
    for i in range(13, n):
        window = values[i - 13 : i + 1]
        diffs = np.diff(window)
        directions = np.sign(diffs)
        alternating = np.all(directions[1:] * directions[:-1] < 0)
        if alternating:
            violations.append(i)
    return violations


def rule8(values: np.ndarray, center: float, sigma: float) -> List[int]:
    """8 consecutive points outside ±1σ with points on both sides."""
    violations = []
    n = len(values)
    for i in range(7, n):
        window = values[i - 7 : i + 1]
        if all(abs(v - center) > sigma for v in window) and min(window) < center < max(window):
            violations.append(i)
    return violations


# ─── Master detector ──────────────────────────────────────────────────────────

RULE_NAMES = {
    1: "Rule 1: 1pt beyond 3σ",
    2: "Rule 2: 2/3 beyond 2σ",
    3: "Rule 3: 4/5 beyond 1σ",
    4: "Rule 4: 8 pts one side",
    5: "Rule 5: 6 pt trend",
    6: "Rule 6: 15 pts in 1σ",
    7: "Rule 7: 14 alternating",
    8: "Rule 8: 8 pts both sides >1σ",
}

_RULE_FUNCTIONS = {
    1: rule1,
    2: rule2,
    3: rule3,
    4: rule4,
    5: rule5,
    6: rule6,
    7: rule7,
    8: rule8,
}


def detect_violations(
    values: np.ndarray | list,
    center: float,
    ucl: float,
    lcl: float,
    rules: tuple = (1, 2, 3, 4, 5),
) -> Dict[int, List[int]]:
    """
    Apply the specified Western Electric rules.

    Parameters
    ----------
    values : array-like
        Subgroup statistics (X-bar values or individuals).
    center : float
        Process center line (grand mean).
    ucl : float
        Upper Control Limit (center + 3σ).
    lcl : float
        Lower Control Limit (center - 3σ).
    rules : tuple of int
        Which rules to apply (1-8). Defaults to the 5 most common.

    Returns
    -------
    dict mapping rule number → list of violating indices.
    """
    x = np.asarray(values, dtype=float)
    sigma = (ucl - center) / 3.0

    result: Dict[int, List[int]] = {}
    for r in rules:
        fn = _RULE_FUNCTIONS.get(r)
        if fn is None:
            raise ValueError(f"Unknown rule number: {r}. Must be 1-8.")
        result[r] = fn(x, center, sigma)

    return result


def violations_to_dataframe(
    violations: Dict[int, List[int]],
    index_values: list | None = None,
) -> pd.DataFrame:
    """
    Convert violations dict to a tidy DataFrame.

    Parameters
    ----------
    violations : dict from detect_violations()
    index_values : optional list mapping positional index → label (e.g. subgroup id)

    Returns
    -------
    pd.DataFrame with columns: rule, rule_name, index, label (if provided)
    """
    rows = []
    for rule_num, indices in violations.items():
        for idx in indices:
            label = index_values[idx] if index_values is not None else idx
            rows.append(
                {
                    "rule": rule_num,
                    "rule_name": RULE_NAMES[rule_num],
                    "subgroup_index": idx,
                    "label": label,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["rule", "rule_name", "subgroup_index", "label"])
    return pd.DataFrame(rows).sort_values("subgroup_index").reset_index(drop=True)


if __name__ == "__main__":
    # Quick smoke-test
    rng = np.random.default_rng(42)
    test_vals = rng.normal(1000, 3.5, 40)
    # Inject a few anomalies
    test_vals[15] = 1013.0   # Rule 1 — beyond 3σ
    test_vals[25:35] = test_vals[25:35] + 4.0  # above center

    center = 1000.0
    ucl = 1000 + 3 * 3.5
    lcl = 1000 - 3 * 3.5

    viols = detect_violations(test_vals, center, ucl, lcl)
    df = violations_to_dataframe(viols)
    print(df)
