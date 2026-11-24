"""
spc_charts.py
=============
X-bar and R Control Chart Calculations and Plotting

SPC constants for n=5 (from Montgomery, Introduction to SQC, 8th ed., Table VI):
  A2 = 0.577    (X-bar chart control limit factor from Rbar)
  D3 = 0        (R chart LCL factor — LCL=0 for n≤6)
  D4 = 2.114    (R chart UCL factor)
  d2 = 2.326    (expected value of the relative range)

Control limit formulas:
  X-bar chart:
    Center line = grand mean (X̄̄)
    UCL = X̄̄ + A2 × R̄
    LCL = X̄̄ − A2 × R̄
  R chart:
    Center line = R̄ (average range)
    UCL = D4 × R̄
    LCL = D3 × R̄   (= 0 for n ≤ 6)

Usage:
    from src.spc_charts import build_control_chart, plot_xbar_r
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional
from src.rules import detect_violations, RULE_NAMES


# ─── SPC constants (Montgomery Table VI) ──────────────────────────────────────
SPC_CONSTANTS = {
    # n: (A2, D3, D4, d2)
    2:  (1.880, 0,     3.267, 1.128),
    3:  (1.023, 0,     2.574, 1.693),
    4:  (0.729, 0,     2.282, 2.059),
    5:  (0.577, 0,     2.114, 2.326),
    6:  (0.483, 0,     2.004, 2.534),
    7:  (0.419, 0.076, 1.924, 2.704),
    8:  (0.373, 0.136, 1.864, 2.847),
    9:  (0.337, 0.184, 1.816, 2.970),
    10: (0.308, 0.223, 1.777, 3.078),
}


def get_constants(n: int) -> dict:
    """Return A2, D3, D4, d2 for subgroup size n."""
    if n not in SPC_CONSTANTS:
        raise ValueError(f"Constants not tabulated for n={n}. Supported: {sorted(SPC_CONSTANTS)}")
    a2, d3, d4, d2 = SPC_CONSTANTS[n]
    return {"A2": a2, "D3": d3, "D4": d4, "d2": d2}


class ControlChartData:
    """
    Holds subgroup statistics and control limits for an X-bar & R chart pair.
    """

    def __init__(
        self,
        subgroup_labels: list,
        xbar: np.ndarray,
        ranges: np.ndarray,
        xbar_cl: float,
        xbar_ucl: float,
        xbar_lcl: float,
        r_cl: float,
        r_ucl: float,
        r_lcl: float,
        sigma_xbar: float,
        n: int,
    ):
        self.subgroup_labels = subgroup_labels
        self.xbar = xbar
        self.ranges = ranges
        self.xbar_cl = xbar_cl
        self.xbar_ucl = xbar_ucl
        self.xbar_lcl = xbar_lcl
        self.r_cl = r_cl
        self.r_ucl = r_ucl
        self.r_lcl = r_lcl
        self.sigma_xbar = sigma_xbar   # σ of the X-bar chart (= Rbar/d2/sqrt(n))
        self.n = n


def build_control_chart(
    df: pd.DataFrame,
    value_col: str = "weight_mg",
    group_col: str = "subgroup_id",
    n: int = 5,
) -> ControlChartData:
    """
    Compute subgroup means, ranges, and control limits.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain value_col and group_col.
    value_col : str
        Column of individual measurements.
    group_col : str
        Column identifying subgroup membership. Supply subgroups in observation order.
    n : int
        Subgroup size.

    Returns
    -------
    ControlChartData
    """
    consts = get_constants(n)
    A2, D3, D4, d2 = consts["A2"], consts["D3"], consts["D4"], consts["d2"]

    # Rule windows must follow production order, even when labels are strings.
    grouped = df.groupby(group_col, sort=False)[value_col]
    if not grouped.count().eq(n).all():
        raise ValueError(f"Each subgroup must contain {n} measurements for these control limits.")
    xbar = grouped.mean().values
    ranges = (grouped.max() - grouped.min()).values
    labels = grouped.mean().index.tolist()

    xbar_cl = float(np.mean(xbar))
    r_cl = float(np.mean(ranges))

    xbar_ucl = xbar_cl + A2 * r_cl
    xbar_lcl = xbar_cl - A2 * r_cl

    r_ucl = D4 * r_cl
    r_lcl = D3 * r_cl

    sigma_xbar = r_cl / (d2 * np.sqrt(n))

    return ControlChartData(
        subgroup_labels=labels,
        xbar=xbar,
        ranges=ranges,
        xbar_cl=xbar_cl,
        xbar_ucl=xbar_ucl,
        xbar_lcl=xbar_lcl,
        r_cl=r_cl,
        r_ucl=r_ucl,
        r_lcl=r_lcl,
        sigma_xbar=sigma_xbar,
        n=n,
    )


def plot_xbar_r(
    chart: ControlChartData,
    title_prefix: str = "",
    save_path: Optional[Path] = None,
    usl: Optional[float] = None,
    lsl: Optional[float] = None,
    highlight_violations: bool = True,
) -> plt.Figure:
    """
    Plot the X-bar and R charts stacked vertically.

    Parameters
    ----------
    chart : ControlChartData
    title_prefix : str
        Optional prefix (e.g. machine line name) for the plot title.
    save_path : Path, optional
        If provided, saves the figure to disk.
    usl, lsl : float, optional
        Specification limits to overlay on the X-bar chart.
    highlight_violations : bool
        Whether to mark Western Electric rule violations.

    Returns
    -------
    matplotlib Figure
    """
    x = np.arange(len(chart.xbar))
    sigma = chart.sigma_xbar

    # Detect violations (Rules 1-5)
    violations = detect_violations(
        chart.xbar, chart.xbar_cl, chart.xbar_ucl, chart.xbar_lcl
    )
    viol_indices = set()
    for idxs in violations.values():
        viol_indices.update(idxs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("#F7F6F2")
    for ax in (ax1, ax2):
        ax.set_facecolor("#F9F8F5")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, color="#D4D1CA")

    # ── X-bar chart ───────────────────────────────────────────────────────────
    ax1.plot(x, chart.xbar, color="#20808D", linewidth=1.4, zorder=3, label="X-bar")
    ax1.axhline(chart.xbar_cl, color="#28251D", linewidth=1.2, label=f"CL={chart.xbar_cl:.2f}")
    ax1.axhline(chart.xbar_ucl, color="#A13544", linewidth=1.0, linestyle="--",
                label=f"UCL={chart.xbar_ucl:.2f}")
    ax1.axhline(chart.xbar_lcl, color="#A13544", linewidth=1.0, linestyle="--",
                label=f"LCL={chart.xbar_lcl:.2f}")

    # Zone lines (1σ, 2σ)
    for mult, alpha in [(1, 0.3), (2, 0.3)]:
        ax1.axhline(chart.xbar_cl + mult * sigma, color="#7A7974", linewidth=0.6,
                    linestyle=":", alpha=alpha)
        ax1.axhline(chart.xbar_cl - mult * sigma, color="#7A7974", linewidth=0.6,
                    linestyle=":", alpha=alpha)

    # Spec limits
    if usl is not None:
        ax1.axhline(usl, color="#964219", linewidth=1.0, linestyle="-.",
                    label=f"USL={usl:.0f}")
    if lsl is not None:
        ax1.axhline(lsl, color="#964219", linewidth=1.0, linestyle="-.",
                    label=f"LSL={lsl:.0f}")

    # Violation markers
    if highlight_violations and viol_indices:
        viol_x = list(viol_indices)
        viol_y = [chart.xbar[i] for i in viol_x]
        ax1.scatter(viol_x, viol_y, color="#A13544", s=60, zorder=5,
                    label="OOC signal", marker="^")

    ax1.set_ylabel("Subgroup Mean (mg)", fontsize=11)
    ax1.set_title(
        f"{title_prefix} — X-bar Chart  (n={chart.n})",
        fontsize=13, fontweight="bold", color="#28251D", pad=8,
    )
    ax1.legend(fontsize=8, loc="upper right", ncol=3, framealpha=0.85)

    # ── R chart ───────────────────────────────────────────────────────────────
    ax2.plot(x, chart.ranges, color="#1B474D", linewidth=1.4, zorder=3, label="Range")
    ax2.axhline(chart.r_cl, color="#28251D", linewidth=1.2, label=f"R̄={chart.r_cl:.2f}")
    ax2.axhline(chart.r_ucl, color="#A13544", linewidth=1.0, linestyle="--",
                label=f"UCL={chart.r_ucl:.2f}")
    if chart.r_lcl > 0:
        ax2.axhline(chart.r_lcl, color="#A13544", linewidth=1.0, linestyle="--",
                    label=f"LCL={chart.r_lcl:.2f}")

    ax2.set_ylabel("Subgroup Range (mg)", fontsize=11)
    ax2.set_title("R Chart", fontsize=11, fontweight="bold", color="#28251D", pad=6)
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.85)

    # X-axis labels — show every 5th label to avoid crowding
    ax2.set_xlabel("Subgroup", fontsize=11)
    tick_step = max(1, len(x) // 20)
    ax2.set_xticks(x[::tick_step])
    ax2.set_xticklabels(
        [str(chart.subgroup_labels[i]) for i in x[::tick_step]],
        rotation=45, ha="right", fontsize=8,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[spc_charts] Saved X-bar & R chart → {save_path}")

    return fig


def plot_individuals_run_chart(
    df: pd.DataFrame,
    value_col: str = "weight_mg",
    usl: float = 1015.0,
    lsl: float = 985.0,
    target: float = 1000.0,
    save_path: Optional[Path] = None,
    title: str = "Individual Measurements — Run Chart",
) -> plt.Figure:
    """Plot individual measurements as a run chart with spec limits and target."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#F7F6F2")
    ax.set_facecolor("#F9F8F5")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, color="#D4D1CA")

    y = df[value_col].values
    x = np.arange(len(y))

    ax.plot(x, y, color="#20808D", linewidth=0.7, alpha=0.8, label="Individual")
    ax.axhline(usl, color="#A13544", linewidth=1.2, linestyle="--", label=f"USL={usl:.0f}")
    ax.axhline(lsl, color="#A13544", linewidth=1.2, linestyle="--", label=f"LSL={lsl:.0f}")
    ax.axhline(target, color="#28251D", linewidth=1.0, linestyle="-", label=f"Target={target:.0f}")

    # Highlight out-of-spec points
    oos_mask = (y > usl) | (y < lsl)
    if oos_mask.any():
        ax.scatter(x[oos_mask], y[oos_mask], color="#A13544", s=20, zorder=5,
                   label="Out-of-spec")

    ax.set_xlabel("Observation", fontsize=11)
    ax.set_ylabel("Weight (mg)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#28251D")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[spc_charts] Saved run chart → {save_path}")

    return fig


if __name__ == "__main__":
    consts = get_constants(5)
    print("SPC constants for n=5:")
    for k, v in consts.items():
        print(f"  {k} = {v}")
