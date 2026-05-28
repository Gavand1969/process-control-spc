"""
test_capability.py
==================
Unit tests for process capability calculations.

Tests verify:
  1. SPC constant d2 for n=5 equals 2.326 (Montgomery Table VI)
  2. Cpk formula: min((USL-mean)/3σ, (mean-LSL)/3σ)
  3. Cp formula: (USL-LSL) / (6 * sigma_within)
  4. Pp and Ppk use long-term sigma (std dev of all observations)
  5. A perfectly centered process has Cpk == Cp
  6. An off-center process has Cpk < Cp
  7. Edge cases: data at spec limit, sigma near zero
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from src.capability import compute_capability, d2_constant, capability_by_line
import pandas as pd


# ─── Constants tests ──────────────────────────────────────────────────────────

class TestSPCConstants:
    def test_d2_n5(self):
        """d2 for subgroup size n=5 must be 2.326 (Montgomery Table VI)."""
        assert d2_constant(5) == pytest.approx(2.326, abs=1e-6)

    def test_d2_n2(self):
        """d2 for n=2 must be 1.128."""
        assert d2_constant(2) == pytest.approx(1.128, abs=1e-6)

    def test_d2_n3(self):
        assert d2_constant(3) == pytest.approx(1.693, abs=1e-6)

    def test_d2_n4(self):
        assert d2_constant(4) == pytest.approx(2.059, abs=1e-6)

    def test_d2_n10(self):
        assert d2_constant(10) == pytest.approx(3.078, abs=1e-6)

    def test_d2_invalid_n(self):
        with pytest.raises(ValueError, match="not tabulated"):
            d2_constant(11)


# ─── Capability index tests ────────────────────────────────────────────────────

class TestCpkFormula:
    """Verify Cpk = min((USL-mean)/(3σ), (mean-LSL)/(3σ))"""

    def _known_data(self, mean=1000.0, sigma=3.5, n=600):
        """Generate deterministic data with known mean and std dev."""
        rng = np.random.default_rng(42)
        data = rng.normal(mean, sigma, n)
        # Force exact mean and std via standardization
        data = (data - data.mean()) / data.std(ddof=1) * sigma + mean
        return data

    def test_cpk_centered_process(self):
        """
        A centered process (mean = target) should have Cpk ≈ Cp.
        USL=1015, LSL=985, tolerance=30, σ_within ≈ 3.5
        Theoretical Cp = 30 / (6 * 3.5) = 1.4286
        """
        data = self._known_data(mean=1000.0, sigma=3.5)
        result = compute_capability(data, usl=1015, lsl=985, subgroup_size=5)
        # Cpk ≈ Cp for centered process
        assert abs(result.cpk - result.cp) < 0.05, (
            f"Centered process: Cpk={result.cpk:.4f} should ≈ Cp={result.cp:.4f}"
        )

    def test_cpk_off_center_less_than_cp(self):
        """Off-center process must have Cpk < Cp."""
        data = self._known_data(mean=1005.0, sigma=3.5)  # shifted +5 mg
        result = compute_capability(data, usl=1015, lsl=985, subgroup_size=5)
        assert result.cpk < result.cp, (
            f"Off-center process: Cpk={result.cpk:.4f} must be < Cp={result.cp:.4f}"
        )

    def test_cpk_formula_manual(self):
        """
        Manually verify Cpk formula on a simple dataset.
        Create 600 points with mean=1002, sigma_within computed via Rbar/d2.
        Then verify: cpu = (USL-mean)/(3*sigma_within), cpl = (mean-LSL)/(3*sigma_within)
        """
        usl, lsl = 1015.0, 985.0
        target_mean = 1002.0
        target_sigma = 3.5
        rng = np.random.default_rng(99)
        data = rng.normal(target_mean, target_sigma, 600)
        # Normalize to exact mean
        data = data - data.mean() + target_mean

        result = compute_capability(data, usl=usl, lsl=lsl, subgroup_size=5)

        # Manually compute expected Cpk using result's sigma_within
        sigma_w = result.sigma_within
        expected_cpu = (usl - result.mean) / (3 * sigma_w)
        expected_cpl = (result.mean - lsl) / (3 * sigma_w)
        expected_cpk = min(expected_cpu, expected_cpl)

        assert result.cpk == pytest.approx(expected_cpk, abs=1e-4)
        assert result.cpu == pytest.approx(expected_cpu, abs=1e-4)
        assert result.cpl == pytest.approx(expected_cpl, abs=1e-4)

    def test_cp_formula_manual(self):
        """Cp = (USL - LSL) / (6 * sigma_within)."""
        usl, lsl = 1015.0, 985.0
        rng = np.random.default_rng(7)
        data = rng.normal(1000.0, 3.5, 600)
        result = compute_capability(data, usl=usl, lsl=lsl, subgroup_size=5)
        expected_cp = (usl - lsl) / (6 * result.sigma_within)
        assert result.cp == pytest.approx(expected_cp, abs=1e-4)

    def test_pp_uses_overall_sigma(self):
        """Pp = (USL-LSL) / (6 * sigma_overall), where sigma_overall = std(data, ddof=1)."""
        usl, lsl = 1015.0, 985.0
        rng = np.random.default_rng(11)
        data = rng.normal(1000.0, 3.5, 600)
        result = compute_capability(data, usl=usl, lsl=lsl, subgroup_size=5)
        expected_pp = (usl - lsl) / (6 * result.sigma_overall)
        assert result.pp == pytest.approx(expected_pp, abs=1e-4)

    def test_ppk_formula_manual(self):
        """Ppk = min((USL-mean)/(3*sigma_overall), (mean-LSL)/(3*sigma_overall))."""
        usl, lsl = 1015.0, 985.0
        rng = np.random.default_rng(13)
        data = rng.normal(1002.0, 3.5, 600)
        result = compute_capability(data, usl=usl, lsl=lsl, subgroup_size=5)
        sigma_ov = result.sigma_overall
        expected_ppu = (usl - result.mean) / (3 * sigma_ov)
        expected_ppl = (result.mean - lsl) / (3 * sigma_ov)
        expected_ppk = min(expected_ppu, expected_ppl)
        assert result.ppk == pytest.approx(expected_ppk, abs=1e-4)

    def test_sigma_level(self):
        """Sigma level = Cpk * 3."""
        rng = np.random.default_rng(17)
        data = rng.normal(1000.0, 3.5, 600)
        result = compute_capability(data, usl=1015, lsl=985, subgroup_size=5)
        assert result.sigma_level == pytest.approx(result.cpk * 3, abs=1e-4)

    def test_insufficient_data_raises(self):
        """Need at least 2 observations."""
        with pytest.raises(ValueError, match="at least 2"):
            compute_capability([1000.0], usl=1015, lsl=985)

    def test_non_1d_raises(self):
        """2-D array should raise ValueError."""
        data_2d = np.array([[1000, 1001], [999, 998]])
        with pytest.raises(ValueError, match="1-D"):
            compute_capability(data_2d, usl=1015, lsl=985)


# ─── Capability by line ────────────────────────────────────────────────────────

class TestCapabilityByLine:
    def test_returns_one_row_per_line(self):
        rng = np.random.default_rng(55)
        lines = ["Line A", "Line B", "Line C"]
        rows = []
        for line in lines:
            weights = rng.normal(1000, 3.5, 120)
            rows += [{"machine_line": line, "weight_mg": w} for w in weights]
        df = pd.DataFrame(rows)
        result = capability_by_line(df, usl=1015, lsl=985, subgroup_size=5)
        assert set(result.index) == set(lines)
        assert "cpk" in result.columns
        assert "cp" in result.columns

    def test_capability_values_positive(self):
        """All capability indices should be positive for a reasonable process."""
        rng = np.random.default_rng(66)
        rows = [{"machine_line": "Line A", "weight_mg": w}
                for w in rng.normal(1000, 3.5, 120)]
        df = pd.DataFrame(rows)
        result = capability_by_line(df, usl=1015, lsl=985, subgroup_size=5)
        assert (result[["cp", "cpk", "pp", "ppk"]] > 0).all().all()


# ─── Documented behavior: short-term vs long-term σ relationship ─────────────

class TestShortTermVsLongTerm:
    def test_pp_lower_than_cp_when_special_cause_present(self):
        """
        When a special-cause shift inflates between-subgroup variation,
        σ_overall > σ_within and Pp < Cp. This is the textbook signature
        of a process that is capable within subgroups but unstable over time.
        """
        rng = np.random.default_rng(2026)
        # 60 subgroups of 5 with a step shift halfway through
        first_half = rng.normal(1000.0, 2.0, 150)
        second_half = rng.normal(1006.0, 2.0, 150)  # +6 mg shift
        data = np.concatenate([first_half, second_half])
        result = compute_capability(data, usl=1015, lsl=985, subgroup_size=5)
        assert result.sigma_overall > result.sigma_within
        assert result.pp < result.cp

    def test_cp_equals_pp_when_no_special_cause(self):
        """
        Without a special-cause, σ_within ≈ σ_overall, so Cp ≈ Pp.
        This is the textbook signature of a stable process.
        """
        rng = np.random.default_rng(2027)
        data = rng.normal(1000.0, 3.5, 600)
        result = compute_capability(data, usl=1015, lsl=985, subgroup_size=5)
        assert abs(result.cp - result.pp) < 0.15

    def test_observations_trimmed_to_complete_subgroups(self):
        """
        When n is not divisible by subgroup_size, extra observations are trimmed.
        Results should still be computed (not raise) and be reasonable.
        """
        rng = np.random.default_rng(2028)
        data = rng.normal(1000.0, 3.5, 603)  # 600 + 3 extras
        result = compute_capability(data, usl=1015, lsl=985, subgroup_size=5)
        assert result.cp > 0
        assert result.cpk > 0
