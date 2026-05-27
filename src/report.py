"""
report.py
=========
End-to-end SPC report runner.

Generates:
  1. Synthetic factory data (if not already present)
  2. X-bar & R charts (overall + per machine line)
  3. Histogram with specification limits
  4. Process capability analysis (Cp, Cpk, Pp, Ppk)
  5. Pareto chart of defect causes
  6. Run chart
  7. One-page control-plan-style text report

Usage:
    python src/report.py
    python src/report.py --regen   # Force regenerate data even if CSV exists
"""

from __future__ import annotations
import argparse
import textwrap
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── project imports ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generate_factory_data import generate_data, TARGET, USL, LSL, SAMPLES_PER_SHIFT
from src.capability import compute_capability, capability_by_line
from src.spc_charts import build_control_chart, plot_xbar_r, plot_individuals_run_chart, get_constants
from src.rules import detect_violations, violations_to_dataframe, RULE_NAMES

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "factory_measurements.csv"
CHARTS_DIR = BASE_DIR / "reports" / "charts"
REPORT_PATH = BASE_DIR / "reports" / "control_plan_report.txt"

CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Helper: assign subgroup IDs ──────────────────────────────────────────────

def assign_subgroup_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'subgroup_id' column (day × shift × line combination)."""
    df = df.copy()
    df["subgroup_id"] = (
        df["day"].astype(str)
        + "-S"
        + df["shift"].astype(str)
        + "-"
        + df["machine_line"].str.replace(" ", "")
    )
    return df


# ─── Chart generators ─────────────────────────────────────────────────────────

def make_xbar_r_overall(df: pd.DataFrame) -> Path:
    """X-bar & R chart for all lines combined (subgroups = day+shift+line)."""
    df2 = assign_subgroup_ids(df)
    chart = build_control_chart(df2, value_col="weight_mg", group_col="subgroup_id", n=5)
    out = CHARTS_DIR / "xbar_r_overall.png"
    plot_xbar_r(chart, title_prefix="All Lines", save_path=out, usl=USL, lsl=LSL)
    plt.close("all")
    return out


def make_xbar_r_per_line(df: pd.DataFrame) -> dict:
    """X-bar & R charts per machine line."""
    paths = {}
    for line in df["machine_line"].unique():
        sub = df[df["machine_line"] == line].copy()
        sub["subgroup_id"] = sub["day"].astype(str) + "-S" + sub["shift"].astype(str)
        chart = build_control_chart(sub, value_col="weight_mg", group_col="subgroup_id", n=5)
        slug = line.replace(" ", "_").lower()
        out = CHARTS_DIR / f"xbar_r_{slug}.png"
        plot_xbar_r(chart, title_prefix=line, save_path=out, usl=USL, lsl=LSL)
        plt.close("all")
        paths[line] = out
    return paths


def make_histogram(df: pd.DataFrame) -> Path:
    """Histogram of all measurements with spec limits and normal overlay."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#F7F6F2")
    ax.set_facecolor("#F9F8F5")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    weights = df["weight_mg"].values
    ax.hist(
        weights, bins=50, color="#20808D", alpha=0.75, edgecolor="#1B474D",
        linewidth=0.5, label="Measurements", density=True, zorder=2,
    )

    # Normal overlay
    mu, sigma = weights.mean(), weights.std(ddof=1)
    x_range = np.linspace(weights.min() - 2, weights.max() + 2, 300)
    from scipy.stats import norm
    pdf = norm.pdf(x_range, mu, sigma)
    ax.plot(x_range, pdf, color="#28251D", linewidth=2, label=f"Normal (μ={mu:.1f}, σ={sigma:.2f})")

    # Spec limits
    ax.axvline(USL, color="#A13544", linewidth=2, linestyle="--", label=f"USL={USL:.0f} mg")
    ax.axvline(LSL, color="#A13544", linewidth=2, linestyle="--", label=f"LSL={LSL:.0f} mg")
    ax.axvline(TARGET, color="#437A22", linewidth=1.5, linestyle="-", label=f"Target={TARGET:.0f} mg")

    # Shade OOS regions
    ax.axvspan(USL, x_range.max(), alpha=0.12, color="#A13544", zorder=1)
    ax.axvspan(x_range.min(), LSL, alpha=0.12, color="#A13544", zorder=1)

    ax.set_xlabel("Rod Weight (mg)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Histogram with Specification Limits — All Lines", fontsize=13,
                 fontweight="bold", color="#28251D")
    ax.legend(fontsize=9, framealpha=0.85)
    plt.tight_layout()

    out = CHARTS_DIR / "histogram_spec_limits.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[report] Saved histogram → {out}")
    return out


def make_pareto(df: pd.DataFrame) -> Path:
    """Pareto chart of defect causes for out-of-spec observations."""
    oos = df[df["out_of_spec"]].copy()
    if oos.empty:
        print("[report] No out-of-spec observations — skipping Pareto.")
        return None

    counts = oos["defect_cause"].value_counts()
    cumulative_pct = (counts.cumsum() / counts.sum() * 100)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#F7F6F2")
    ax1.set_facecolor("#F9F8F5")
    ax1.spines["top"].set_visible(False)
    ax2 = ax1.twinx()

    bars = ax1.bar(range(len(counts)), counts.values, color="#20808D", alpha=0.85,
                   edgecolor="#1B474D", linewidth=0.5)
    ax2.plot(range(len(counts)), cumulative_pct.values, color="#A84B2F",
             marker="o", linewidth=2, markersize=6, label="Cumulative %")
    ax2.axhline(80, color="#964219", linewidth=1.0, linestyle="--", alpha=0.7)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Cumulative %", fontsize=11)
    ax2.yaxis.set_tick_params(labelsize=9)

    ax1.set_xticks(range(len(counts)))
    ax1.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=10)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_xlabel("Defect Cause", fontsize=11)
    ax1.set_title("Pareto Chart of Defect Causes", fontsize=13,
                  fontweight="bold", color="#28251D")

    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(val), ha="center", va="bottom", fontsize=9, color="#28251D")

    ax2.legend(loc="center right", fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    plt.tight_layout()
    out = CHARTS_DIR / "pareto_defect_causes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[report] Saved Pareto → {out}")
    return out


def make_capability_gauge(cap_overall) -> Path:
    """Visual Cpk gauge chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#F7F6F2")
    ax.set_facecolor("#F9F8F5")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    indices = ["Cp", "Cpk", "Pp", "Ppk"]
    values = [cap_overall.cp, cap_overall.cpk, cap_overall.pp, cap_overall.ppk]
    colors = []
    for v in values:
        if v >= 1.67:
            colors.append("#437A22")   # Excellent
        elif v >= 1.33:
            colors.append("#20808D")   # Good
        elif v >= 1.00:
            colors.append("#FFC553")   # Marginal
        else:
            colors.append("#A13544")   # Inadequate

    bars = ax.barh(indices, values, color=colors, alpha=0.85, edgecolor="#28251D",
                   linewidth=0.5, height=0.55)
    ax.axvline(1.0, color="#964219", linewidth=1.5, linestyle="--", label="Min target (1.00)")
    ax.axvline(1.33, color="#20808D", linewidth=1.5, linestyle="--", label="Good target (1.33)")
    ax.axvline(1.67, color="#437A22", linewidth=1.5, linestyle="--", label="Six Sigma (1.67)")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=12, fontweight="bold", color="#28251D")

    ax.set_xlim(0, max(max(values) + 0.4, 2.0))
    ax.set_xlabel("Index Value", fontsize=11)
    ax.set_title("Process Capability Indices — All Lines Combined", fontsize=13,
                 fontweight="bold", color="#28251D")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()

    out = CHARTS_DIR / "capability_gauge.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[report] Saved capability gauge → {out}")
    return out


# ─── Text report ──────────────────────────────────────────────────────────────

def write_text_report(
    df: pd.DataFrame,
    cap_overall,
    cap_by_line: pd.DataFrame,
    violations_summary: dict,
) -> Path:
    """Write a one-page control-plan-style text report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    oos_rate = df["out_of_spec"].mean()
    n_oos = df["out_of_spec"].sum()
    n_total = len(df)

    lines_text = []
    for line_name, row in cap_by_line.iterrows():
        lines_text.append(
            f"  {line_name:10s}  Cp={row['cp']:.3f}  Cpk={row['cpk']:.3f}  "
            f"Pp={row['pp']:.3f}  Ppk={row['ppk']:.3f}  "
            f"Mean={row['mean']:.2f} mg  σ_within={row['sigma_within']:.3f}"
        )

    report_text = textwrap.dedent(f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║           CONTROL PLAN REPORT — TOBACCO ROD WEIGHT (Statistical SPC)        ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    Generated : {now}
    Author    : Gavin Anderson (M.S. Data Science)
    Reference : AIAG SPC Manual, 2nd Ed. | Montgomery SQC, 8th Ed.

    ─────────────────────────── PART INFORMATION ────────────────────────────────
    Part           : Tobacco Rod
    Characteristic : Fill Weight (mg)
    Target         : {TARGET:.0f} mg
    USL            : {USL:.0f} mg
    LSL            : {LSL:.0f} mg
    Tolerance      : {USL - LSL:.0f} mg
    Subgroup size  : n = {SAMPLES_PER_SHIFT}
    SPC Constants  : A2=0.577, D3=0, D4=2.114, d2=2.326  (Montgomery Table VI)

    ─────────────────────────── DATA SUMMARY ────────────────────────────────────
    Total observations : {n_total}
    Days               : 30  (4 shifts/day × 3 lines × 5 samples)
    Machine lines      : Line A, Line B, Line C
    Overall mean       : {df['weight_mg'].mean():.3f} mg
    Overall std dev    : {df['weight_mg'].std(ddof=1):.3f} mg
    Out-of-spec count  : {n_oos}  ({oos_rate:.2%} of observations)

    ─────────────────────────── CAPABILITY INDICES (ALL LINES) ──────────────────
    Short-term σ (Rbar/d2) : {cap_overall.sigma_within:.4f} mg
    Long-term σ  (s)       : {cap_overall.sigma_overall:.4f} mg
    Process mean           : {cap_overall.mean:.4f} mg

      Cp  = {cap_overall.cp:.4f}   (tolerance / 6σ_within)
      Cpk = {cap_overall.cpk:.4f}   min((USL-mean)/3σ, (mean-LSL)/3σ) — short-term
      Pp  = {cap_overall.pp:.4f}   (tolerance / 6σ_overall)
      Ppk = {cap_overall.ppk:.4f}   min((USL-mean)/3σ, (mean-LSL)/3σ) — long-term
      Sigma level = {cap_overall.sigma_level:.2f}σ

      Interpretation:
        Cp  ≥ 1.33  → Good potential capability         : {'✓ PASS' if cap_overall.cp >= 1.33 else '✗ FAIL'}
        Cpk ≥ 1.33  → Process centered and capable      : {'✓ PASS' if cap_overall.cpk >= 1.33 else '✗ FAIL'}
        Ppk ≥ 1.33  → Long-term performance adequate     : {'✓ PASS' if cap_overall.ppk >= 1.33 else '✗ FAIL'}

    ─────────────────────────── CAPABILITY BY LINE ───────────────────────────────
{chr(10).join(lines_text)}

    ─────────────────────────── OUT-OF-CONTROL SIGNALS ─────────────────────────
    Western Electric rules applied to X-bar chart (all lines combined):
""")

    total_violations = 0
    for rule_num, indices in violations_summary.items():
        count = len(indices)
        total_violations += count
        report_text += f"    {RULE_NAMES[rule_num]:35s}: {count} signal(s)\n"

    report_text += f"\n    Total out-of-control signals detected : {total_violations}\n"

    report_text += textwrap.dedent("""
    ─────────────────────────── CORRECTIVE ACTIONS ───────────────────────────────
    Quality Event (Line B, Days 18-23):
      Root cause   : Progressive mean shift (+8 mg) with increased variability
      Likely cause : Tobacco blend variation / hopper feed rate inconsistency
      Action taken : Hopper recalibration, blend adjustment (Day 23)
      Result       : Process returned to control by Day 24

    ─────────────────────────── RECOMMENDED ACTIONS ──────────────────────────────
    1. Continue X-bar & R monitoring on 5-sample subgroups each shift
    2. Investigate Hopper Feed Rate (30% of defects — top Pareto cause)
    3. Implement SPC alert automation to trigger immediate corrective action
    4. Target Cpk ≥ 1.67 (Six Sigma level) through process centering
    5. Review D3/D4 constants if subgroup size changes

    ─────────────────────────── CHART INVENTORY ──────────────────────────────────
    reports/charts/xbar_r_overall.png        — X-bar & R (all lines)
    reports/charts/xbar_r_line_a.png         — X-bar & R (Line A)
    reports/charts/xbar_r_line_b.png         — X-bar & R (Line B) — quality event visible
    reports/charts/xbar_r_line_c.png         — X-bar & R (Line C)
    reports/charts/histogram_spec_limits.png  — Histogram with USL/LSL
    reports/charts/capability_gauge.png       — Cp/Cpk/Pp/Ppk bar chart
    reports/charts/pareto_defect_causes.png   — Pareto of defect causes
    reports/charts/run_chart.png              — Individual run chart (all observations)

    ═══════════════════════════════════════════════════════════════════════════════
    END OF REPORT
    ═══════════════════════════════════════════════════════════════════════════════
    """)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text)
    print(f"[report] Saved control plan report → {REPORT_PATH}")
    return REPORT_PATH


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(regen: bool = False):
    print("=" * 70)
    print("  SPC Process Control — End-to-End Report")
    print("=" * 70)

    # 1. Load or generate data
    if DATA_PATH.exists() and not regen:
        print(f"[report] Loading existing data: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print("[report] Generating factory data...")
        df = generate_data(save=True)

    print(f"[report] {len(df)} observations loaded. Lines: {df['machine_line'].unique().tolist()}")

    # 2. Process capability — all lines
    cap_overall = compute_capability(
        df["weight_mg"].values, usl=USL, lsl=LSL, subgroup_size=SAMPLES_PER_SHIFT
    )
    print(f"\n[report] Overall Cp={cap_overall.cp:.4f}  Cpk={cap_overall.cpk:.4f}  "
          f"Pp={cap_overall.pp:.4f}  Ppk={cap_overall.ppk:.4f}")

    # Capability by line
    cap_by_line = capability_by_line(df, usl=USL, lsl=LSL, subgroup_size=SAMPLES_PER_SHIFT)
    print("\n[report] Capability by line:")
    print(cap_by_line[["cp", "cpk", "pp", "ppk", "mean", "sigma_within"]].to_string())

    # 3. Build overall control chart and detect violations
    df2 = assign_subgroup_ids(df)
    overall_chart = build_control_chart(df2, value_col="weight_mg",
                                        group_col="subgroup_id", n=5)
    violations_summary = detect_violations(
        overall_chart.xbar, overall_chart.xbar_cl,
        overall_chart.xbar_ucl, overall_chart.xbar_lcl,
    )
    viol_df = violations_to_dataframe(violations_summary)
    if not viol_df.empty:
        print(f"\n[report] {len(viol_df)} out-of-control signals detected:")
        print(viol_df.head(20).to_string())

    # 4. Generate all charts
    print("\n[report] Generating charts...")
    make_xbar_r_overall(df)
    make_xbar_r_per_line(df)
    make_histogram(df)
    make_pareto(df)
    make_capability_gauge(cap_overall)
    plot_individuals_run_chart(
        df, usl=USL, lsl=LSL, target=TARGET,
        save_path=CHARTS_DIR / "run_chart.png",
        title="Individual Measurements — Run Chart (All Lines, All Days)",
    )
    plt.close("all")

    # 5. Write text report
    write_text_report(df, cap_overall, cap_by_line, violations_summary)

    print("\n" + "=" * 70)
    print("  REPORT COMPLETE")
    print(f"  Charts : {CHARTS_DIR}")
    print(f"  Report : {REPORT_PATH}")
    print("=" * 70)

    return cap_overall, cap_by_line


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end SPC report.")
    parser.add_argument("--regen", action="store_true",
                        help="Regenerate synthetic data even if CSV exists.")
    args = parser.parse_args()
    main(regen=args.regen)
