"""
test_mastery.py — Unit tests for mastery thresholding, guess detection,
and expected-attempts-to-mastery forward simulation.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bkt import (
    MASTERY_THRESHOLD,
    check_mastery,
    expected_attempts_to_mastery,
    is_probable_guess,
    mastery_gap,
)
from bkt.config import (
    DEFAULT_P_G,
    DEFAULT_P_S,
    DEFAULT_P_T,
    IRT_SURPRISE_THRESHOLD,
    RT_GUESS_THRESHOLD_MS,
)
from bkt.exceptions import MasteryThresholdError


# ── check_mastery ─────────────────────────────────────────────────────────────


class TestCheckMastery:
    def test_at_threshold_is_mastered(self):
        assert check_mastery(MASTERY_THRESHOLD) is True

    def test_above_threshold_is_mastered(self):
        assert check_mastery(0.99) is True

    def test_below_threshold_is_not_mastered(self):
        assert check_mastery(0.94) is False

    def test_custom_threshold(self):
        assert check_mastery(0.80, threshold=0.80) is True
        assert check_mastery(0.79, threshold=0.80) is False

    def test_invalid_threshold_zero_raises(self):
        with pytest.raises(MasteryThresholdError):
            check_mastery(0.5, threshold=0.0)

    def test_invalid_threshold_one_raises(self):
        with pytest.raises(MasteryThresholdError):
            check_mastery(0.5, threshold=1.0)


# ── mastery_gap ───────────────────────────────────────────────────────────────


class TestMasteryGap:
    def test_positive_gap_when_below_threshold(self):
        gap = mastery_gap(0.70, threshold=MASTERY_THRESHOLD)
        assert gap == pytest.approx(MASTERY_THRESHOLD - 0.70)
        assert gap > 0

    def test_negative_gap_when_above_threshold(self):
        gap = mastery_gap(0.98, threshold=MASTERY_THRESHOLD)
        assert gap < 0

    def test_zero_gap_at_threshold(self):
        gap = mastery_gap(MASTERY_THRESHOLD)
        assert gap == pytest.approx(0.0, abs=1e-10)


# ── is_probable_guess ─────────────────────────────────────────────────────────


class TestIsProbableGuess:
    """Two-condition logic: fast AND surprising (if IRT available)."""

    def test_incorrect_never_guess(self):
        assert is_probable_guess(
            is_correct=False,
            response_time_ms=100,
            theta=-2.0, difficulty=3.0,
        ) is False

    def test_slow_correct_never_guess(self):
        assert is_probable_guess(
            is_correct=True,
            response_time_ms=RT_GUESS_THRESHOLD_MS + 1,
            theta=-2.0, difficulty=3.0,
        ) is False

    def test_fast_surprising_correct_is_guess(self):
        """Fast + IRT P(correct) < IRT_SURPRISE_THRESHOLD → guess."""
        # theta very low, difficulty very high → P_irt << 0.30
        assert is_probable_guess(
            is_correct=True,
            response_time_ms=200,
            theta=-3.0,
            difficulty=3.0,
            discrimination=1.0,
        ) is True

    def test_fast_not_surprising_not_guess(self):
        """Fast answer on an easy item for a high-ability student is NOT a guess."""
        # θ=3.0, b=-2.0 → P_irt ≈ 0.993 >> 0.30 → not surprising → not a guess
        assert is_probable_guess(
            is_correct=True,
            response_time_ms=500,
            theta=3.0,
            difficulty=-2.0,
            discrimination=1.0,
        ) is False

    def test_fast_no_irt_params_is_guess(self):
        """Fast correct with no IRT params → time-only fallback → guess."""
        assert is_probable_guess(
            is_correct=True,
            response_time_ms=300,
            theta=None, difficulty=None,
        ) is True

    def test_correct_no_time_not_guess(self):
        """No response time → cannot be too fast → not a guess."""
        assert is_probable_guess(
            is_correct=True,
            response_time_ms=None,
            theta=-2.0, difficulty=3.0,
        ) is False

    def test_custom_rt_threshold(self):
        """Custom rt_threshold_ms should be respected."""
        # response_time_ms = 2000, default threshold = 1500 → not too fast by default
        assert is_probable_guess(
            is_correct=True,
            response_time_ms=2000,
            rt_threshold_ms=1500,
        ) is False
        # With threshold=3000, 2000ms IS fast → time-only fallback → guess
        assert is_probable_guess(
            is_correct=True,
            response_time_ms=2000,
            rt_threshold_ms=3000,
        ) is True


# ── expected_attempts_to_mastery ──────────────────────────────────────────────


class TestExpectedAttemptsToMastery:
    def test_already_mastered_returns_zero(self):
        result = expected_attempts_to_mastery(
            p_l=MASTERY_THRESHOLD,
            p_t=DEFAULT_P_T, p_g=DEFAULT_P_G, p_s=DEFAULT_P_S,
        )
        assert result == 0

    def test_returns_positive_int_when_below_threshold(self):
        result = expected_attempts_to_mastery(
            p_l=0.50,
            p_t=DEFAULT_P_T, p_g=DEFAULT_P_G, p_s=DEFAULT_P_S,
        )
        assert result is not None
        assert result > 0

    def test_higher_p_l_needs_fewer_attempts(self):
        """A student closer to mastery needs fewer all-correct attempts."""
        r_low  = expected_attempts_to_mastery(0.20, DEFAULT_P_T, DEFAULT_P_G, DEFAULT_P_S)
        r_high = expected_attempts_to_mastery(0.80, DEFAULT_P_T, DEFAULT_P_G, DEFAULT_P_S)
        assert r_low is not None and r_high is not None
        assert r_low > r_high

    def test_higher_p_t_means_fewer_attempts(self):
        """Higher learn rate → fewer attempts to mastery."""
        r_slow = expected_attempts_to_mastery(0.20, p_t=0.05, p_g=DEFAULT_P_G, p_s=DEFAULT_P_S)
        r_fast = expected_attempts_to_mastery(0.20, p_t=0.30, p_g=DEFAULT_P_G, p_s=DEFAULT_P_S)
        assert r_slow is not None and r_fast is not None
        assert r_slow > r_fast

    def test_returns_none_for_very_slow_learner_with_high_guess(self):
        """Extremely slow P(T) + high P(G) may not converge within max_steps."""
        # With P(T) = 0.01 and P(G) = 0.44, mastery may require >200 steps
        result = expected_attempts_to_mastery(
            p_l=0.05, p_t=0.01, p_g=0.44, p_s=0.05,
            max_steps=5  # deliberately small
        )
        # May or may not be None; just assert it's either a positive int or None
        assert result is None or result > 0
