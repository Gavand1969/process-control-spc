"""
generate_factory_data.py
========================
Generates synthetic tobacco rod weight data simulating a PMI-style
cigarette manufacturing environment.

Setup:
  - Target weight: 1000 mg  (tobacco rod fill weight)
  - Specification limits: USL = 1015 mg, LSL = 985 mg  (±15 mg)
  - 30 days × 4 shifts/day × 5 samples/shift = 600 observations
  - 3 machine lines: Line A, Line B, Line C

Deliberate quality event:
  Line B drifts out of specification around days 18-22 (mean shifts +8 mg,
  sigma increases). A corrective action is applied after day 23, restoring
  Line B to normal from day 24 onward.

Usage:
    python src/generate_factory_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─── Process parameters ────────────────────────────────────────────────────────
TARGET = 1000.0   # mg
USL = 1015.0      # Upper Specification Limit
LSL = 985.0       # Lower Specification Limit
SIGMA_NORMAL = 3.5  # process std dev under control (gives ~Cpk 1.43)

RANDOM_SEED = 42

LINES = ["Line A", "Line B", "Line C"]
N_DAYS = 30
SHIFTS_PER_DAY = 4
SAMPLES_PER_SHIFT = 5  # subgroup size n = 5


def _shift_label(shift_idx: int) -> str:
    return ["Day", "Evening", "Night", "Maintenance"][shift_idx % 4]


def generate_data(save: bool = True) -> pd.DataFrame:
    """
    Generate synthetic factory measurement data and optionally save to CSV.

    Returns
    -------
    pd.DataFrame with columns:
        day, shift, shift_name, machine_line, sample_id,
        weight_mg, out_of_spec, defect_cause
    """
    rng = np.random.default_rng(RANDOM_SEED)
    records = []
    sample_counter = 0

    for day in range(1, N_DAYS + 1):
        for shift_idx in range(SHIFTS_PER_DAY):
            shift_name = _shift_label(shift_idx)
            for line in LINES:
                # ── Determine process parameters for this subgroup ──────────
                mu = TARGET
                sigma = SIGMA_NORMAL

                # Line B quality event: drift days 18-22, corrective action day 24+
                if line == "Line B":
                    if 18 <= day <= 22:
                        # Progressive drift: mean shifts up, variability increases
                        drift_factor = (day - 17) / 5.0  # 0.2 to 1.0
                        mu = TARGET + drift_factor * 8.0   # up to +8 mg shift
                        sigma = SIGMA_NORMAL * (1 + drift_factor * 0.6)
                    elif day == 23:
                        # Corrective action day – partially recovered, still elevated
                        mu = TARGET + 4.0
                        sigma = SIGMA_NORMAL * 1.3

                # Small random walk to add realistic day-to-day variation
                mu += rng.normal(0, 0.5)

                samples = rng.normal(mu, sigma, SAMPLES_PER_SHIFT)

                for s_idx, weight in enumerate(samples):
                    sample_counter += 1
                    weight = round(weight, 3)
                    out_of_spec = weight < LSL or weight > USL

                    # Assign a defect cause category when out-of-spec
                    defect_cause = None
                    if out_of_spec:
                        causes = [
                            "Tobacco blend variation",
                            "Hopper feed rate",
                            "Humidity change",
                            "Cutting blade wear",
                            "Operator error",
                        ]
                        # Weight the causes: blade wear & hopper most common
                        cause_weights = [0.15, 0.30, 0.20, 0.25, 0.10]
                        defect_cause = rng.choice(causes, p=cause_weights)

                    records.append(
                        {
                            "sample_id": sample_counter,
                            "day": day,
                            "shift": shift_idx + 1,
                            "shift_name": shift_name,
                            "machine_line": line,
                            "weight_mg": weight,
                            "out_of_spec": out_of_spec,
                            "defect_cause": defect_cause,
                        }
                    )

    df = pd.DataFrame(records)

    if save:
        out_path = Path(__file__).resolve().parent.parent / "data" / "factory_measurements.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[generate_factory_data] Saved {len(df)} rows → {out_path}")

    return df


if __name__ == "__main__":
    df = generate_data(save=True)
    print(df.describe())
    print(f"\nOut-of-spec rate: {df['out_of_spec'].mean():.2%}")
    print(df.groupby("machine_line")["out_of_spec"].mean().rename("OOS rate"))
