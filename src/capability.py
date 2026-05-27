"""
capability.py
=============
Process Capability Analysis

Implements the standard Six Sigma / AIAG indices:

    Cp   = (USL - LSL) / (6 * sigma_within)
    Cpk  = min((USL - mean) / (3 * sigma_within),
               (mean  - LSL) / (3 * sigma_within))
    Pp   = (USL - LSL) / (6 * sigma_overall)
    Ppk  = min((USL - mean) / (3 * sigma_overall),
               (mean  - LSL) / (3 * sigma_overall))

Where:
  sigma_within  = Rbar / d2   (estimated from average range, short-term)
  sigma_overall = std(all observations, ddof=1)  (long-term)

SPC constants (n=5):
  d2 = 2.326   (from Montgomery, Introduction to Statistical Quality Control)

Reference: Montgomery, D.C. (2020). Introduction to Statistical Quality Control,
           8th ed., Table VI.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import NamedTuple


# ─── SPC constants table (Montgomery Table VI) ─────────────────────────────────
# Keyed by subgroup size n.  Only the most common sizes are listed.
_D2 = {
    2: 1.128,
    3: 1.693,
    4: 2.059,
    5: 2.326,
    6: 2.534,
    7: 2.704,
    8: 2.847,
    9: 2.970,
    10: 3.078,
}


def d2_constant(n: int) -> float:
    """Return the d2 unbiasing constant for subgroup size n."""
    if n not in _D2:
        raise ValueError(f"d2 constant not tabulated for n={n}. Supported: {sorted(_D2)}")
    return _D2[n]


class CapabilityResult(NamedTuple):
    """Container for process capability indices."""
    mean: float
    sigma_within: float     # short-term (from Rbar/d2)
    sigma_overall: float    # long-term (sample std dev)
    usl: float
    lsl: float
    cp: float
    cpk: float
    pp: float
    ppk: float
    cpu: float              # upper one-sided index
    cpl: float              # lower one-sided index
    sigma_level: float      # process sigma level (Cpk * 3)


def compute_capability(
    data: np.ndarray | list,
    usl: float,
    lsl: float,
    subgroup_size: int = 5,
) -> CapabilityResult:
    """
    Compute Cp, Cpk, Pp, Ppk for a 1-D array of individual measurements.

    Parameters
    ----------
    data : array-like
        Individual measurements (all observations pooled).
    usl : float
        Upper Specification Limit.
    lsl : float
        Lower Specification Limit.
    subgroup_size : int
        Subgroup size n used during production (needed for d2 lookup).

    Returns
    -------
    CapabilityResult  (named tuple with all indices)
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 1:
        raise ValueError("data must be 1-D array of individual measurements.")

    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    # ── Descriptive statistics ──────────────────────────────────────────────
    mean = float(np.mean(x))
    sigma_overall = float(np.std(x, ddof=1))

    # ── Short-term sigma from subgroup ranges ───────────────────────────────
    # Reshape data into subgroups, compute range per subgroup, then Rbar/d2
    d2 = d2_constant(subgroup_size)

    remainder = n % subgroup_size
    if remainder != 0:
        # Trim to complete subgroups only
        x_trimmed = x[: n - remainder]
    else:
        x_trimmed = x

    subgroups = x_trimmed.reshape(-1, subgroup_size)
    ranges = np.ptp(subgroups, axis=1)  # range = max - min per subgroup
    rbar = float(np.mean(ranges))
    sigma_within = rbar / d2

    # ── Capability indices ──────────────────────────────────────────────────
    tolerance = usl - lsl

    cp  = tolerance / (6.0 * sigma_within)
    cpu = (usl - mean) / (3.0 * sigma_within)
    cpl = (mean - lsl) / (3.0 * sigma_within)
    cpk = min(cpu, cpl)

    pp  = tolerance / (6.0 * sigma_overall)
    ppu = (usl - mean) / (3.0 * sigma_overall)
    ppl = (mean - lsl) / (3.0 * sigma_overall)
    ppk = min(ppu, ppl)

    sigma_level = cpk * 3.0

    return CapabilityResult(
        mean=round(mean, 4),
        sigma_within=round(sigma_within, 4),
        sigma_overall=round(sigma_overall, 4),
        usl=usl,
        lsl=lsl,
        cp=round(cp, 4),
        cpk=round(cpk, 4),
        pp=round(pp, 4),
        ppk=round(ppk, 4),
        cpu=round(cpu, 4),
        cpl=round(cpl, 4),
        sigma_level=round(sigma_level, 4),
    )


def capability_by_line(
    df: pd.DataFrame,
    usl: float,
    lsl: float,
    subgroup_size: int = 5,
) -> pd.DataFrame:
    """
    Compute capability indices for each machine line.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: machine_line, weight_mg
    usl, lsl : float
        Specification limits.

    Returns
    -------
    pd.DataFrame indexed by machine_line with one row per line.
    """
    rows = []
    for line, grp in df.groupby("machine_line"):
        cap = compute_capability(grp["weight_mg"].values, usl, lsl, subgroup_size)
        rows.append({"machine_line": line, **cap._asdict()})
    return pd.DataFrame(rows).set_index("machine_line")


if __name__ == "__main__":
    # Quick smoke-test with known values
    rng = np.random.default_rng(0)
    data = rng.normal(1000, 3.5, 600)
    result = compute_capability(data, usl=1015, lsl=985, subgroup_size=5)
    print("Capability indices (smoke test):")
    for k, v in result._asdict().items():
        print(f"  {k:20s}: {v}")
