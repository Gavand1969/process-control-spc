"""
test_rules.py
=============
Unit tests for Western Electric out-of-control rules.

Tests verify:
  1. Rule 1 triggers on points beyond ±3σ and not otherwise
  2. Rule 2 triggers on 2-of-3 consecutive points beyond 2σ (same side)
  3. Rule 3 triggers on 4-of-5 consecutive points beyond 1σ (same side)
  4. Rule 4 triggers on 8 consecutive points same side of centerline
  5. Rule 5 triggers on 6 consecutive strictly monotone points
  6. detect_violations returns correct structure
  7. violations_to_dataframe returns correct columns
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from src.rules import (
    rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8,
    detect_violations, violations_to_dataframe, RULE_NAMES,
)

CENTER = 1000.0
SIGMA = 3.5
UCL = CENTER + 3 * SIGMA   # 1010.5
LCL = CENTER - 3 * SIGMA   # 989.5


def _flat(n=30) -> np.ndarray:
    """Return n points all exactly at center — no violations."""
    return np.full(n, CENTER)


# ─── Rule 1: 1 point beyond ±3σ ──────────────────────────────────────────────

class TestRule1:
    def test_no_violation_within_limits(self):
        """Points exactly at UCL and LCL are NOT violations (boundary excluded)."""
        x = _flat()
        x[10] = CENTER + 3 * SIGMA  # exactly at UCL — borderline, should not trigger
        # Rule 1 uses strict >, so exactly 3σ does NOT trigger
        result = rule1(x, CENTER, SIGMA)
        assert 10 not in result

    def test_violation_above_ucl(self):
        """A point above UCL must trigger Rule 1."""
        x = _flat()
        x[5] = CENTER + 3.01 * SIGMA  # just beyond 3σ
        result = rule1(x, CENTER, SIGMA)
        assert 5 in result

    def test_violation_below_lcl(self):
        """A point below LCL must trigger Rule 1."""
        x = _flat()
        x[15] = CENTER - 3.5 * SIGMA
        result = rule1(x, CENTER, SIGMA)
        assert 15 in result

    def test_all_in_control_no_violations(self):
        x = _flat()
        assert rule1(x, CENTER, SIGMA) == []

    def test_multiple_violations(self):
        x = _flat()
        x[3] = CENTER + 4 * SIGMA
        x[20] = CENTER - 4 * SIGMA
        result = rule1(x, CENTER, SIGMA)
        assert 3 in result
        assert 20 in result
        assert len(result) == 2


# ─── Rule 2: 2-of-3 beyond 2σ, same side ─────────────────────────────────────

class TestRule2:
    def test_triggers_above(self):
        """2 of 3 consecutive points above +2σ triggers Rule 2."""
        x = _flat(30)
        # indices 5, 6, 7: two above 2σ
        x[5] = CENTER + 2.5 * SIGMA
        x[6] = CENTER + 0.5 * SIGMA  # in zone
        x[7] = CENTER + 2.1 * SIGMA
        result = rule2(x, CENTER, SIGMA)
        assert 7 in result

    def test_triggers_below(self):
        """2 of 3 consecutive points below −2σ triggers Rule 2."""
        x = _flat(30)
        x[10] = CENTER - 2.5 * SIGMA
        x[11] = CENTER - 0.5 * SIGMA
        x[12] = CENTER - 2.2 * SIGMA
        result = rule2(x, CENTER, SIGMA)
        assert 12 in result

    def test_no_trigger_opposite_sides(self):
        """One point above 2σ, one below 2σ — different sides, no trigger."""
        x = _flat(30)
        x[10] = CENTER + 2.5 * SIGMA
        x[12] = CENTER - 2.5 * SIGMA
        # These are on opposite sides, so rule 2 should NOT trigger for the window
        result = rule2(x, CENTER, SIGMA)
        # Window [10,11,12]: one above, one below — no trigger
        assert 12 not in result

    def test_no_violation_in_control(self):
        assert rule2(_flat(), CENTER, SIGMA) == []


# ─── Rule 3: 4-of-5 beyond 1σ, same side ─────────────────────────────────────

class TestRule3:
    def test_triggers_above(self):
        """4 of 5 consecutive points above +1σ triggers Rule 3."""
        x = _flat(30)
        for i in range(5, 10):
            x[i] = CENTER + 1.5 * SIGMA  # all above 1σ
        result = rule3(x, CENTER, SIGMA)
        assert 9 in result

    def test_triggers_below(self):
        x = _flat(30)
        x[0] = CENTER - 1.2 * SIGMA
        x[1] = CENTER - 1.3 * SIGMA
        x[2] = CENTER - 1.1 * SIGMA
        x[3] = CENTER - 0.2 * SIGMA  # this one is within 1σ
        x[4] = CENTER - 1.4 * SIGMA
        # 4 of 5 below -1σ → triggers
        result = rule3(x, CENTER, SIGMA)
        assert 4 in result

    def test_no_violation_in_control(self):
        assert rule3(_flat(), CENTER, SIGMA) == []


# ─── Rule 4: 8 consecutive same side of centerline ───────────────────────────

class TestRule4:
    def test_triggers_above(self):
        """8 consecutive points above centerline triggers Rule 4."""
        x = _flat(30)
        for i in range(10, 18):
            x[i] = CENTER + 0.5 * SIGMA  # above center
        result = rule4(x, CENTER, SIGMA)
        assert 17 in result

    def test_triggers_below(self):
        x = _flat(30)
        for i in range(5, 13):
            x[i] = CENTER - 0.3  # just below center
        result = rule4(x, CENTER, SIGMA)
        assert 12 in result

    def test_7_points_no_trigger(self):
        """7 consecutive same side should NOT trigger Rule 4."""
        x = _flat(30)
        for i in range(0, 7):
            x[i] = CENTER + 0.1
        result = rule4(x, CENTER, SIGMA)
        assert result == []

    def test_no_violation_in_control(self):
        assert rule4(_flat(), CENTER, SIGMA) == []


# ─── Rule 5: 6 consecutive trending ──────────────────────────────────────────

class TestRule5:
    def test_triggers_increasing(self):
        """6 strictly increasing points triggers Rule 5."""
        x = _flat(30)
        for i, val in enumerate([999.0, 999.5, 1000.0, 1000.5, 1001.0, 1001.5]):
            x[i] = val
        result = rule5(x, CENTER, SIGMA)
        assert 5 in result

    def test_triggers_decreasing(self):
        x = _flat(30)
        for i, val in enumerate([1002.0, 1001.5, 1001.0, 1000.5, 1000.0, 999.5]):
            x[i] = val
        result = rule5(x, CENTER, SIGMA)
        assert 5 in result

    def test_5_increasing_no_trigger(self):
        """Only 5 consecutive increasing — should NOT trigger (need 6)."""
        x = _flat(30)
        for i, val in enumerate([999.0, 999.5, 1000.0, 1000.5, 1001.0]):
            x[i] = val
        result = rule5(x, CENTER, SIGMA)
        assert result == []

    def test_no_violation_in_control(self):
        assert rule5(_flat(), CENTER, SIGMA) == []


# ─── Rule 6: 15 consecutive points within ±1σ (stratification) ───────────────

class TestRule6:
    def test_triggers_on_15_hugging_centerline(self):
        """15 consecutive points within ±1σ trigger Rule 6."""
        x = _flat(30)
        for i in range(15):
            x[i] = CENTER + 0.1 * SIGMA  # well within 1σ
        result = rule6(x, CENTER, SIGMA)
        assert 14 in result

    def test_no_trigger_one_point_outside_band(self):
        """A single point outside ±1σ breaks the run — no trigger."""
        x = _flat(30)
        for i in range(15):
            x[i] = CENTER + 0.1 * SIGMA
        x[10] = CENTER + 1.5 * SIGMA  # break the run
        result = rule6(x, CENTER, SIGMA)
        assert 14 not in result

    def test_no_violation_when_points_vary(self):
        """Normal noisy data should not trigger Rule 6."""
        rng = np.random.default_rng(123)
        x = rng.normal(CENTER, SIGMA, 50)
        # Some points must fall outside 1σ in random normal data
        assert rule6(x, CENTER, SIGMA) == []


# ─── Rule 7: 14 consecutive alternating up-down ──────────────────────────────

class TestRule7:
    def test_triggers_on_perfect_alternation(self):
        """14 strictly alternating points trigger Rule 7."""
        x = _flat(30)
        # Build a strict zig-zag of length 14
        base = CENTER
        for i in range(14):
            x[i] = base + (0.3 if i % 2 == 0 else -0.3)
        result = rule7(x, CENTER, SIGMA)
        assert 13 in result

    def test_no_trigger_on_monotone(self):
        """Strictly increasing values are NOT alternating."""
        x = _flat(30)
        for i in range(14):
            x[i] = CENTER + i * 0.1
        result = rule7(x, CENTER, SIGMA)
        assert 13 not in result

    def test_no_violation_in_control(self):
        assert rule7(_flat(), CENTER, SIGMA) == []


# ─── Rule 8: 8 consecutive points beyond ±1σ (mixture) ───────────────────────

class TestRule8:
    def test_triggers_on_mixture(self):
        """8 consecutive points all beyond ±1σ (either side) trigger Rule 8."""
        x = _flat(30)
        # Alternate sides — all beyond 1σ in absolute value
        for i in range(8):
            x[i] = CENTER + (1.5 * SIGMA if i % 2 == 0 else -1.5 * SIGMA)
        result = rule8(x, CENTER, SIGMA)
        assert 7 in result

    def test_no_trigger_with_one_in_zone_c(self):
        """A single point within ±1σ breaks the mixture pattern."""
        x = _flat(30)
        for i in range(8):
            x[i] = CENTER + (1.5 * SIGMA if i % 2 == 0 else -1.5 * SIGMA)
        x[4] = CENTER + 0.2 * SIGMA  # within 1σ — breaks rule
        result = rule8(x, CENTER, SIGMA)
        assert 7 not in result

    def test_no_violation_in_control(self):
        assert rule8(_flat(), CENTER, SIGMA) == []


# ─── detect_violations integration ────────────────────────────────────────────

class TestDetectViolations:
    def test_returns_dict_with_requested_rules(self):
        x = _flat(30)
        result = detect_violations(x, CENTER, UCL, LCL, rules=(1, 2, 3, 4, 5))
        assert set(result.keys()) == {1, 2, 3, 4, 5}

    def test_all_in_control_empty_violations(self):
        """All-centered data should produce no Rule 1 violations."""
        x = _flat(30)
        result = detect_violations(x, CENTER, UCL, LCL, rules=(1,))
        assert result[1] == []

    def test_invalid_rule_raises(self):
        with pytest.raises(ValueError, match="Unknown rule"):
            detect_violations(_flat(), CENTER, UCL, LCL, rules=(9,))

    def test_rule1_flagged_correctly(self):
        x = _flat(30)
        x[7] = CENTER + 4 * SIGMA  # clear Rule 1 violation
        result = detect_violations(x, CENTER, UCL, LCL, rules=(1,))
        assert 7 in result[1]

    def test_sigma_derived_from_ucl(self):
        """detect_violations computes sigma = (UCL - center) / 3."""
        x = _flat(30)
        x[0] = 1015.0  # beyond UCL=1010.5 → Rule 1
        result = detect_violations(x, CENTER, UCL, LCL, rules=(1,))
        assert 0 in result[1]

    def test_all_eight_rules_dispatchable(self):
        """All 8 WECO rules must be selectable via detect_violations."""
        x = _flat(30)
        result = detect_violations(x, CENTER, UCL, LCL, rules=(1, 2, 3, 4, 5, 6, 7, 8))
        assert set(result.keys()) == {1, 2, 3, 4, 5, 6, 7, 8}
        # On flat data, only rule 6 (15 in ±1σ) should fire — all 30 points are exactly at center
        for r in (1, 2, 3, 5, 7, 8):
            assert result[r] == [], f"Rule {r} should not trigger on flat data"


# ─── violations_to_dataframe ──────────────────────────────────────────────────

class TestViolationsToDataframe:
    def test_columns_present(self):
        x = _flat(30)
        x[5] = CENTER + 4 * SIGMA
        viols = detect_violations(x, CENTER, UCL, LCL, rules=(1, 2))
        df = violations_to_dataframe(viols)
        assert "rule" in df.columns
        assert "rule_name" in df.columns
        assert "subgroup_index" in df.columns

    def test_rule_names_match_constants(self):
        """All rule_name values must come from the RULE_NAMES dict."""
        x = _flat(30)
        x[5] = CENTER + 4 * SIGMA
        viols = detect_violations(x, CENTER, UCL, LCL)
        df = violations_to_dataframe(viols)
        if not df.empty:
            for name in df["rule_name"]:
                assert name in RULE_NAMES.values()

    def test_empty_when_no_violations(self):
        x = _flat(30)
        viols = detect_violations(x, CENTER, UCL, LCL)
        df = violations_to_dataframe(viols)
        assert df.empty


# ─── SPC constants from spc_charts module ────────────────────────────────────
# These tests cross-verify the A2, D3, D4, d2 values directly

class TestSPCChartsConstants:
    def test_a2_n5(self):
        """A2 for n=5 must be 0.577 (Montgomery Table VI)."""
        from src.spc_charts import get_constants
        consts = get_constants(5)
        assert consts["A2"] == pytest.approx(0.577, abs=1e-6)

    def test_d3_n5_is_zero(self):
        """D3 for n=5 must be 0 (LCL of R chart is 0 for n≤6)."""
        from src.spc_charts import get_constants
        consts = get_constants(5)
        assert consts["D3"] == pytest.approx(0.0, abs=1e-9)

    def test_d4_n5(self):
        """D4 for n=5 must be 2.114 (Montgomery Table VI)."""
        from src.spc_charts import get_constants
        consts = get_constants(5)
        assert consts["D4"] == pytest.approx(2.114, abs=1e-6)

    def test_d2_n5(self):
        """d2 for n=5 must be 2.326 (Montgomery Table VI)."""
        from src.spc_charts import get_constants
        consts = get_constants(5)
        assert consts["d2"] == pytest.approx(2.326, abs=1e-6)

    def test_invalid_n_raises(self):
        from src.spc_charts import get_constants
        with pytest.raises(ValueError):
            get_constants(11)
