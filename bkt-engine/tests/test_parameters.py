"""
test_parameters.py — Unit tests for IRT → BKT parameter derivation.

Tests verify:
  1. irt_p_correct() matches the 2PL formula exactly.
  2. derive_p_g_from_irt() is monotonically decreasing in P_irt.
  3. derive_p_s_from_irt() is monotonically decreasing in P_irt.
  4. derive_p_t_from_difficulty() is monotonically decreasing in mean_b.
  5. seed_p_l0_from_irt() produces values in [SEED_PRIOR_MIN, SEED_PRIOR_MAX].
  6. from_irt() constructs valid BKTSkillParameters.
  7. Invalid IRT inputs raise IRTParameterRangeError.
"""

import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bkt import (
    BKTSkillParameters,
    derive_p_g_from_irt,
    derive_p_s_from_irt,
    derive_p_t_from_difficulty,
    from_irt,
    irt_p_correct,
    seed_p_l0_from_irt,
)
from bkt.config import (
    DEFAULT_P_G,
    DEFAULT_P_L0,
    DEFAULT_P_S,
    DEFAULT_P_T,
    MASTERY_REFERENCE_DISCRIMINATION,
    P_G_CLAMP_MAX,
    P_G_CLAMP_MIN,
    P_G_IRT_BASE,
    P_G_IRT_FLOOR,
    P_S_CLAMP_MAX,
    P_S_CLAMP_MIN,
    P_S_IRT_BASE,
    P_S_IRT_FLOOR,
    P_T_CLAMP_MAX,
    P_T_CLAMP_MIN,
    SEED_PRIOR_MAX,
    SEED_PRIOR_MIN,
)
from bkt.exceptions import IRTParameterRangeError


# ── Test 1: irt_p_correct formula ─────────────────────────────────────────────


class TestIrtPCorrect:
    """2PL ICC must match the logistic formula exactly."""

    def test_theta_equals_b_gives_half(self):
        """When θ = b (equal ability and difficulty), P(correct) = 0.5."""
        assert irt_p_correct(theta=1.0, b=1.0, a=1.5) == pytest.approx(0.5, abs=1e-9)

    def test_theta_much_greater_than_b(self):
        """θ >> b → P(correct) → 1."""
        p = irt_p_correct(theta=4.0, b=-4.0, a=1.0)
        assert p > 0.99

    def test_theta_much_less_than_b(self):
        """θ << b → P(correct) → 0."""
        p = irt_p_correct(theta=-4.0, b=4.0, a=1.0)
        assert p < 0.01

    def test_higher_discrimination_steeper_curve(self):
        """Higher a → the ICC is steeper around b."""
        # At θ = b + 0.5, higher a → higher P(correct)
        p_low_a  = irt_p_correct(theta=0.5, b=0.0, a=0.5)
        p_high_a = irt_p_correct(theta=0.5, b=0.0, a=2.0)
        assert p_high_a > p_low_a

    def test_nonpositive_a_raises(self):
        with pytest.raises(IRTParameterRangeError, match="discrimination"):
            irt_p_correct(theta=0.0, b=0.0, a=0.0)

    def test_nonfinite_theta_raises(self):
        with pytest.raises(IRTParameterRangeError, match="theta"):
            irt_p_correct(theta=float("inf"), b=0.0, a=1.0)

    def test_nonfinite_b_raises(self):
        with pytest.raises(IRTParameterRangeError, match="difficulty"):
            irt_p_correct(theta=0.0, b=float("nan"), a=1.0)

    def test_output_in_unit_interval(self):
        """P(correct) must be in (0, 1) for all finite inputs."""
        test_cases = [(-4, -4, 0.5), (0, 0, 1.0), (4, 4, 2.0), (-2, 3, 0.8)]
        for theta, b, a in test_cases:
            p = irt_p_correct(theta, b, a)
            assert 0 < p < 1, f"P({theta},{b},{a}) = {p} outside (0,1)"


# ── Test 2: derive_p_g_from_irt monotonicity ─────────────────────────────────


class TestDeriveGuessFromIRT:
    """P(G)_eff must be monotonically decreasing as P_irt increases."""

    def test_high_p_irt_gives_low_p_g(self):
        """Easy item for high-ability student → low effective guess rate."""
        # θ=2.0 (high), b=-2.0 (easy), a=1.0 → P_irt ≈ 0.982
        p_g = derive_p_g_from_irt(theta=2.0, b=-2.0, a=1.0)
        assert p_g < DEFAULT_P_G  # should be well below default 0.25

    def test_low_p_irt_gives_high_p_g(self):
        """Hard item for low-ability student → high effective guess rate."""
        # θ=-2.0, b=2.0, a=1.0 → P_irt ≈ 0.018
        p_g = derive_p_g_from_irt(theta=-2.0, b=2.0, a=1.0)
        assert p_g > P_G_IRT_FLOOR  # should be near base default

    def test_monotonically_decreasing_in_p_irt(self):
        """As θ increases (student gets stronger relative to item), P(G)_eff decreases."""
        p_g_values = [
            derive_p_g_from_irt(theta=t, b=0.0, a=1.0)
            for t in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        ]
        for i in range(len(p_g_values) - 1):
            assert p_g_values[i] >= p_g_values[i + 1], (
                f"P(G) not decreasing: index {i} has {p_g_values[i]:.4f}, "
                f"index {i+1} has {p_g_values[i+1]:.4f}"
            )

    def test_output_within_clamp_bounds(self):
        """P(G)_eff must always be in [P_G_CLAMP_MIN, P_G_CLAMP_MAX]."""
        for theta in [-4, -2, 0, 2, 4]:
            for b in [-2, 0, 2]:
                p_g = derive_p_g_from_irt(theta, b, a=1.0)
                assert P_G_CLAMP_MIN <= p_g <= P_G_CLAMP_MAX

    def test_formula_matches_manual_calculation(self):
        """Spot-check: P_irt = 0.5 → P(G)_eff = P_G_IRT_BASE×0.5 + P_G_IRT_FLOOR."""
        # θ = b → P_irt = σ(0) = 0.5 exactly
        p_g = derive_p_g_from_irt(theta=1.0, b=1.0, a=1.0)
        expected = P_G_IRT_BASE * 0.5 + P_G_IRT_FLOOR
        assert p_g == pytest.approx(expected, rel=1e-6)


# ── Test 3: derive_p_s_from_irt monotonicity ─────────────────────────────────


class TestDeriveSlipFromIRT:
    """P(S)_eff must be monotonically decreasing as P_irt increases."""

    def test_high_p_irt_gives_low_p_s(self):
        """Easy item for high-ability student → low effective slip rate."""
        p_s = derive_p_s_from_irt(theta=2.0, b=-2.0, a=1.0)
        assert p_s < DEFAULT_P_S

    def test_low_p_irt_gives_high_p_s(self):
        """Hard item for low-ability student → high effective slip rate."""
        p_s = derive_p_s_from_irt(theta=-2.0, b=2.0, a=1.0)
        assert p_s > P_S_IRT_FLOOR

    def test_monotonically_decreasing_in_theta(self):
        """P(S)_eff must decrease as θ increases for fixed item."""
        p_s_values = [
            derive_p_s_from_irt(theta=t, b=0.0, a=1.0)
            for t in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        ]
        for i in range(len(p_s_values) - 1):
            assert p_s_values[i] >= p_s_values[i + 1]

    def test_output_within_clamp_bounds(self):
        for theta in [-4, -2, 0, 2, 4]:
            for b in [-2, 0, 2]:
                p_s = derive_p_s_from_irt(theta, b, a=1.0)
                assert P_S_CLAMP_MIN <= p_s <= P_S_CLAMP_MAX

    def test_formula_matches_manual_at_midpoint(self):
        """P_irt = 0.5 → P(S)_eff = P_S_IRT_BASE × 0.5 + P_S_IRT_FLOOR."""
        p_s = derive_p_s_from_irt(theta=1.0, b=1.0, a=1.0)
        expected = P_S_IRT_BASE * 0.5 + P_S_IRT_FLOOR
        assert p_s == pytest.approx(expected, rel=1e-6)


# ── Test 4: derive_p_t_from_difficulty ────────────────────────────────────────


class TestDerivePTFromDifficulty:
    """P(T) must be monotonically decreasing in mean_b (harder → slower learning)."""

    def test_negative_b_gives_higher_p_t(self):
        """Easy concept (mean_b < 0) → faster learning → P(T) > DEFAULT_P_T."""
        p_t = derive_p_t_from_difficulty(mean_b=-2.0)
        assert p_t > DEFAULT_P_T

    def test_zero_b_gives_approx_default(self):
        """Neutral difficulty (mean_b = 0) → P(T) should equal DEFAULT_P_T exactly
        (after normalization by sigmoid(0) = 0.5)."""
        p_t = derive_p_t_from_difficulty(mean_b=0.0)
        assert p_t == pytest.approx(DEFAULT_P_T, rel=1e-6)

    def test_positive_b_gives_lower_p_t(self):
        """Hard concept (mean_b > 0) → slower learning → P(T) < DEFAULT_P_T."""
        p_t = derive_p_t_from_difficulty(mean_b=2.0)
        assert p_t < DEFAULT_P_T

    def test_monotonically_decreasing_in_b(self):
        """P(T) must decrease as mean_b increases."""
        p_t_values = [derive_p_t_from_difficulty(b) for b in [-2.0, -1.0, 0.0, 1.0, 2.0]]
        for i in range(len(p_t_values) - 1):
            assert p_t_values[i] >= p_t_values[i + 1]

    def test_output_within_clamp_bounds(self):
        for b in [-5.0, -2.0, 0.0, 2.0, 5.0]:
            p_t = derive_p_t_from_difficulty(b)
            assert P_T_CLAMP_MIN <= p_t <= P_T_CLAMP_MAX


# ── Test 5: seed_p_l0_from_irt ───────────────────────────────────────────────


class TestSeedPL0FromIRT:
    """IRT cold-start P(L0) seeding must replicate mastery_initializer.py logic."""

    def test_output_in_seed_prior_bounds(self):
        """Result must be in [SEED_PRIOR_MIN, SEED_PRIOR_MAX]."""
        p_l0 = seed_p_l0_from_irt(
            theta=0.5,
            bloom_difficulties=[-1.0, 0.0, 1.0],
            n_correct=2,
            n_attempted=3,
        )
        assert SEED_PRIOR_MIN <= p_l0 <= SEED_PRIOR_MAX

    def test_higher_theta_gives_higher_p_l0(self):
        """A more able student (higher θ) should get a higher P(L0) for the same concept."""
        p_l0_low  = seed_p_l0_from_irt(theta=-1.0, bloom_difficulties=[0.0], n_correct=0, n_attempted=0)
        p_l0_high = seed_p_l0_from_irt(theta=2.0,  bloom_difficulties=[0.0], n_correct=0, n_attempted=0)
        assert p_l0_high > p_l0_low

    def test_perfect_accuracy_increases_p_l0(self):
        """Higher observed accuracy with the same theta → higher P(L0)."""
        p_l0_low  = seed_p_l0_from_irt(theta=0.0, bloom_difficulties=[0.0], n_correct=0, n_attempted=5)
        p_l0_high = seed_p_l0_from_irt(theta=0.0, bloom_difficulties=[0.0], n_correct=5, n_attempted=5)
        assert p_l0_high > p_l0_low

    def test_empty_difficulties_raises(self):
        with pytest.raises(IRTParameterRangeError, match="bloom_difficulties"):
            seed_p_l0_from_irt(theta=0.0, bloom_difficulties=[], n_correct=0, n_attempted=0)

    def test_n_correct_exceeds_n_attempted_raises(self):
        with pytest.raises(IRTParameterRangeError):
            seed_p_l0_from_irt(theta=0.0, bloom_difficulties=[0.0], n_correct=5, n_attempted=3)

    def test_zero_attempts_uses_theta_implied_only(self):
        """With n_attempted=0, the result should be the theta-implied accuracy
        (weight = 0/(0+K) = 0 → full theta prior)."""
        from bkt.config import MASTERY_PRIOR_STRENGTH
        theta = 0.5
        b = 0.0
        p_irt = irt_p_correct(theta, b, MASTERY_REFERENCE_DISCRIMINATION)
        # With n=0, shrinkage weight=0 → result = theta_implied = p_irt, clamped
        p_l0 = seed_p_l0_from_irt(theta=theta, bloom_difficulties=[b], n_correct=0, n_attempted=0)
        expected = max(SEED_PRIOR_MIN, min(SEED_PRIOR_MAX, p_irt))
        assert p_l0 == pytest.approx(expected, rel=1e-6)


# ── Test 6: from_irt factory ──────────────────────────────────────────────────


class TestFromIRTFactory:
    """from_irt() must produce valid, IRT-derived BKTSkillParameters."""

    def test_produces_valid_params(self):
        """from_irt() output must pass BKTSkillParameters validation."""
        params = from_irt(theta=0.5, b=0.0, a=1.0)
        # Validation happens in __post_init__ — if no exception, we're good
        assert isinstance(params, BKTSkillParameters)
        assert params.irt_derived is True

    def test_p_l0_passed_through(self):
        """When p_l0 is explicitly passed, it should be used (clamped)."""
        params = from_irt(theta=0.5, b=0.0, a=1.0, p_l0=0.70)
        assert params.p_l0 == pytest.approx(0.70, abs=1e-6)

    def test_p_l0_defaults_when_none(self):
        """When p_l0=None, fallback to DEFAULT_P_L0."""
        params = from_irt(theta=0.5, b=0.0, a=1.0, p_l0=None)
        assert params.p_l0 == pytest.approx(DEFAULT_P_L0)

    def test_skill_id_stored(self):
        params = from_irt(theta=0.5, b=0.0, a=1.0, skill_id="test_skill")
        assert params.skill_id == "test_skill"

    def test_higher_theta_gives_lower_p_g_and_p_s(self):
        """for the same item, higher student ability → lower P(G) and P(S)."""
        params_low  = from_irt(theta=-1.0, b=0.0, a=1.0)
        params_high = from_irt(theta=2.0,  b=0.0, a=1.0)
        assert params_high.p_g < params_low.p_g
        assert params_high.p_s < params_low.p_s
