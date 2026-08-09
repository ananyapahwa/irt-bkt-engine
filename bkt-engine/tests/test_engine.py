"""
test_engine.py — Unit tests for BKTEngine core update mathematics.

Tests verify:
  1. Consecutive correct answers push P(L) past the mastery threshold.
  2. A single incorrect answer from a high P(L) penalizes correctly and
     is cushioned by P(S) (does not drop catastrophically).
  3. The IRT override hooks (override_p_g, override_p_s) replace skill defaults.
  4. Guess detection inflates P(G) and produces a weaker upward update.
  5. batch_update threads state correctly.
  6. Edge cases: invalid states and parameter degeneracy raise correctly.
"""

import math
import pytest
from datetime import datetime, timezone

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bkt import (
    BKTEngine,
    BKTSkillParameters,
    BKTState,
    MASTERY_THRESHOLD,
)
from bkt.config import (
    DEFAULT_P_G,
    DEFAULT_P_L0,
    DEFAULT_P_S,
    DEFAULT_P_T,
    P_G_CLAMP_MIN,
    P_G_CLAMP_MAX,
    P_L_CLAMP_MAX,
    P_L_CLAMP_MIN,
    P_S_CLAMP_MIN,
    P_S_CLAMP_MAX,
    P_G_INFLATED,
)
from bkt.exceptions import InvalidParameterError, InvalidStateError
from bkt.state import initial_state
from bkt.parameters import defaults


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_params(**kwargs) -> BKTSkillParameters:
    """Construct BKTSkillParameters with defaults overridden by kwargs."""
    base = dict(p_l0=DEFAULT_P_L0, p_t=DEFAULT_P_T, p_g=DEFAULT_P_G, p_s=DEFAULT_P_S)
    base.update(kwargs)
    return BKTSkillParameters(**base)


def make_state(p_l: float = DEFAULT_P_L0, student_id: str = "s1", skill_id: str = "k1") -> BKTState:
    """Construct a BKTState with given p_l."""
    return initial_state(student_id, skill_id, p_l0=p_l)


# ── Test 1: Consecutive correct answers reach mastery ────────────────────────


class TestConsecutiveCorrectAnswers:
    """Consecutive correct answers must push P(L) past MASTERY_THRESHOLD."""

    def test_all_correct_reaches_mastery(self):
        """Starting from DEFAULT_P_L0=0.20, repeated correct answers
        must eventually reach MASTERY_THRESHOLD=0.95."""
        params = defaults()
        state = BKTEngine.initial_state("s1", "k1", params)
        assert not state.is_mastered

        for _ in range(50):  # upper bound — should converge well before this
            if state.is_mastered:
                break
            result = BKTEngine.update(state, is_correct=True, params=params)
            state = result.new_state

        assert state.is_mastered, (
            f"Expected mastery after repeated correct answers, but P(L) = {state.p_l:.4f}"
        )
        assert state.p_l >= MASTERY_THRESHOLD

    def test_mastery_reached_within_reasonable_attempts(self):
        """With standard parameters, mastery should be reached in under 30
        all-correct answers (empirically, ~15-20 with defaults)."""
        params = defaults()
        state = BKTEngine.initial_state("s1", "k1", params)
        attempts_to_mastery = None

        for i in range(1, 31):
            result = BKTEngine.update(state, is_correct=True, params=params)
            state = result.new_state
            if state.is_mastered:
                attempts_to_mastery = i
                break

        assert attempts_to_mastery is not None, (
            "Expected mastery within 30 correct answers, but it was never reached"
        )

    def test_p_l_monotonically_increases_on_all_correct(self):
        """P(L) must strictly increase on every correct answer (no plateau
        before mastery with these standard parameters)."""
        params = defaults()
        state = BKTEngine.initial_state("s1", "k1", params)
        prev_p_l = state.p_l

        for _ in range(20):
            result = BKTEngine.update(state, is_correct=True, params=params)
            state = result.new_state
            assert state.p_l > prev_p_l, (
                f"P(L) did not increase: {prev_p_l:.6f} → {state.p_l:.6f}"
            )
            prev_p_l = state.p_l
            if state.is_mastered:
                break

    def test_delta_positive_for_correct(self):
        """BKTUpdateResult.delta_p_l must be positive for correct answers."""
        params = defaults()
        state = make_state(0.5)
        result = BKTEngine.update(state, is_correct=True, params=params)
        assert result.delta_p_l > 0


# ── Test 2: Single incorrect penalizes correctly, slip cushions ──────────────


class TestIncorrectAnswerPenalty:
    """A single incorrect answer must reduce P(L), but the slip rate P(S)
    provides cushioning — the drop should not be catastrophic."""

    def test_incorrect_reduces_p_l(self):
        """P(L) must decrease after an incorrect answer."""
        state = make_state(0.80)
        params = defaults()
        result = BKTEngine.update(state, is_correct=False, params=params)
        assert result.new_state.p_l < state.p_l, (
            f"Expected P(L) to decrease after incorrect answer, got "
            f"{state.p_l:.4f} → {result.new_state.p_l:.4f}"
        )

    def test_slip_cushions_incorrect_drop(self):
        """With a high P(L)=0.90 and default P(S)=0.10, one incorrect answer
        should not reduce P(L) below 0.50 (the slip parameter provides cushion).
        Mathematical reasoning: P(L|incorrect) = P(L)×P(S) / (P(L)×P(S)+(1-P(L))×(1-P(G)))
        At P(L)=0.90, P(S)=0.10, P(G)=0.25:
          numerator   = 0.90 × 0.10 = 0.090
          denominator = 0.090 + 0.10 × 0.75 = 0.090 + 0.075 = 0.165
          P(L|inc) = 0.090/0.165 ≈ 0.545
          After transition: P(L) = 0.545 + 0.455 × 0.10 ≈ 0.591
        """
        state = make_state(0.90)
        params = defaults()
        result = BKTEngine.update(state, is_correct=False, params=params)
        assert result.new_state.p_l > 0.50, (
            f"Slip cushion failed: P(L) dropped from 0.90 to {result.new_state.p_l:.4f} "
            f"on a single incorrect — expected > 0.50 with P(S)=0.10"
        )

    def test_delta_negative_for_incorrect(self):
        """BKTUpdateResult.delta_p_l must be negative for incorrect answers."""
        state = make_state(0.80)
        params = defaults()
        result = BKTEngine.update(state, is_correct=False, params=params)
        assert result.delta_p_l < 0

    def test_lower_slip_means_harder_incorrect_penalty(self):
        """A lower P(S) means incorrect answers carry less blame on the student
        (less likely they knew it and slipped) → larger downward update."""
        state = make_state(0.70)
        params_high_slip = make_params(p_s=0.20)
        params_low_slip  = make_params(p_s=0.05)

        result_high = BKTEngine.update(state, is_correct=False, params=params_high_slip)
        result_low  = BKTEngine.update(state, is_correct=False, params=params_low_slip)

        # Higher slip → P(L|incorrect) stays higher (more blame on slip, not ignorance)
        assert result_high.new_state.p_l > result_low.new_state.p_l

    def test_high_p_l_mastered_student_recovers_after_one_miss(self):
        """A student at near-mastery (P(L)=0.94) who gets one wrong then one right
        should still reach mastery within a few attempts."""
        params = defaults()
        state = make_state(0.94)

        # One incorrect
        result = BKTEngine.update(state, is_correct=False, params=params)
        state = result.new_state
        assert not state.is_mastered  # should drop below threshold

        # Then correct answers to recover
        for _ in range(10):
            if state.is_mastered:
                break
            result = BKTEngine.update(state, is_correct=True, params=params)
            state = result.new_state

        assert state.is_mastered, (
            f"Student at P(L)=0.94 with one miss should recover quickly; "
            f"final P(L)={state.p_l:.4f}"
        )


# ── Test 3: IRT override hooks ────────────────────────────────────────────────


class TestIRTOverrideHooks:
    """The override_p_g and override_p_s arguments must replace skill defaults."""

    def test_override_p_g_used_in_update(self):
        """When override_p_g is supplied, it must appear in result.p_g_used."""
        state = make_state(0.5)
        params = defaults()
        override_g = 0.10

        result = BKTEngine.update(state, is_correct=True, params=params,
                                  override_p_g=override_g)
        assert result.p_g_used == pytest.approx(override_g)
        assert result.override_p_g_applied is True

    def test_override_p_s_used_in_update(self):
        """When override_p_s is supplied, it must appear in result.p_s_used."""
        state = make_state(0.5)
        params = defaults()
        override_s = 0.05

        result = BKTEngine.update(state, is_correct=True, params=params,
                                  override_p_s=override_s)
        assert result.p_s_used == pytest.approx(override_s)
        assert result.override_p_s_applied is True

    def test_low_p_g_override_gives_stronger_upward_update(self):
        """A lower effective P(G) makes a correct answer stronger evidence of
        mastery → larger upward update compared to the default P(G)."""
        state = make_state(0.5)
        params = defaults()

        result_default = BKTEngine.update(state, is_correct=True, params=params)
        result_low_g   = BKTEngine.update(state, is_correct=True, params=params,
                                          override_p_g=P_G_CLAMP_MIN)

        assert result_low_g.new_state.p_l > result_default.new_state.p_l

    def test_high_p_g_override_gives_weaker_upward_update(self):
        """A higher effective P(G) makes a correct answer weaker evidence of mastery."""
        state = make_state(0.5)
        params = defaults()

        result_default  = BKTEngine.update(state, is_correct=True, params=params)
        result_high_g   = BKTEngine.update(state, is_correct=True, params=params,
                                           override_p_g=P_G_CLAMP_MAX)

        assert result_high_g.new_state.p_l < result_default.new_state.p_l

    def test_invalid_override_p_g_raises(self):
        state = make_state(0.5)
        params = defaults()
        with pytest.raises(InvalidParameterError):
            BKTEngine.update(state, is_correct=True, params=params, override_p_g=0.99)

    def test_invalid_override_p_s_raises(self):
        state = make_state(0.5)
        params = defaults()
        with pytest.raises(InvalidParameterError):
            BKTEngine.update(state, is_correct=True, params=params, override_p_s=-0.01)


# ── Test 4: Guess detection ───────────────────────────────────────────────────


class TestGuessDetection:
    """Guess-flagged correct answers must use inflated P(G) and produce weaker updates."""

    def test_fast_surprising_correct_flags_as_guess(self):
        """A correct answer that is very fast AND IRT-surprising triggers the flag."""
        state = make_state(0.40)
        params = defaults()

        # theta low, difficulty high → IRT P(correct) is very low → "surprising"
        result = BKTEngine.update(
            state, is_correct=True, params=params,
            response_time_ms=500,    # very fast (< 1500ms threshold)
            theta=-1.5,              # below-average ability
            difficulty=2.0,          # hard item
            discrimination=1.0,
        )
        assert result.was_flagged_guess is True
        assert result.p_g_used > DEFAULT_P_G  # inflated

    def test_slow_correct_never_flagged(self):
        """A correct answer with response_time_ms > threshold is never a guess."""
        state = make_state(0.40)
        params = defaults()
        result = BKTEngine.update(
            state, is_correct=True, params=params,
            response_time_ms=5000,   # well above threshold
            theta=-1.5,
            difficulty=2.0,
        )
        assert result.was_flagged_guess is False

    def test_incorrect_never_flagged_as_guess(self):
        """Incorrect answers are never classified as guesses."""
        state = make_state(0.40)
        params = defaults()
        result = BKTEngine.update(
            state, is_correct=False, params=params,
            response_time_ms=200,    # very fast incorrect
            theta=-1.5,
            difficulty=2.0,
        )
        assert result.was_flagged_guess is False

    def test_guess_flag_produces_weaker_upward_update(self):
        """A guess-flagged correct answer should produce a smaller upward update
        than an un-flagged correct answer from the same state."""
        state = make_state(0.40)
        params = defaults()

        # Un-flagged: normal speed, not surprising
        result_normal = BKTEngine.update(
            state, is_correct=True, params=params,
            response_time_ms=8000,
            theta=1.0, difficulty=0.0,
        )
        # Flagged: fast + surprising
        result_guess = BKTEngine.update(
            state, is_correct=True, params=params,
            response_time_ms=500,
            theta=-1.5, difficulty=2.0,
        )

        assert result_normal.new_state.p_l > result_guess.new_state.p_l


# ── Test 5: Batch update ──────────────────────────────────────────────────────


class TestBatchUpdate:
    """batch_update must thread state correctly and return all results."""

    def test_batch_threads_state(self):
        """Final state of batch_update must match sequential single updates."""
        params = defaults()
        observations = [(True, None), (True, None), (False, None), (True, None)]

        # Sequential
        state = make_state()
        for is_correct, rt in observations:
            r = BKTEngine.update(state, is_correct, params)
            state = r.new_state
        expected_p_l = state.p_l

        # Batch
        state_batch = make_state()
        final_state, results = BKTEngine.batch_update(state_batch, observations, params)

        assert final_state.p_l == pytest.approx(expected_p_l, abs=1e-10)
        assert len(results) == len(observations)

    def test_batch_mismatched_overrides_raises(self):
        params = defaults()
        state = make_state()
        observations = [(True, None), (False, None)]
        # 3 overrides for 2 observations → should raise
        irt_overrides = [(None, None), (None, None), (None, None)]
        with pytest.raises(InvalidParameterError):
            BKTEngine.batch_update(state, observations, params, irt_overrides=irt_overrides)


# ── Test 6: Mathematical spot-checks ─────────────────────────────────────────


class TestMathematicalSpotChecks:
    """Verify the exact BKT formula output against hand-computed values."""

    def test_correct_update_formula(self):
        """
        Hand-computed example:
          P(L) = 0.5, P(S) = 0.10, P(G) = 0.25, P(T) = 0.10
          Stage 1 (correct):
            numerator   = 0.5 × (1 - 0.10) = 0.5 × 0.90 = 0.45
            denominator = 0.45 + 0.5 × 0.25 = 0.45 + 0.125 = 0.575
            P(L|correct) = 0.45 / 0.575 ≈ 0.78261
          Stage 2:
            P(L) = 0.78261 + (1 - 0.78261) × 0.10 ≈ 0.78261 + 0.02174 ≈ 0.80435
        """
        state = make_state(0.5)
        params = make_params(p_l0=0.5, p_t=0.10, p_g=0.25, p_s=0.10)
        result = BKTEngine.update(state, is_correct=True, params=params)

        expected_conditional = 0.45 / 0.575
        expected_p_l = expected_conditional + (1 - expected_conditional) * 0.10

        assert result.p_conditional == pytest.approx(expected_conditional, rel=1e-6)
        assert result.new_state.p_l  == pytest.approx(expected_p_l, rel=1e-6)

    def test_incorrect_update_formula(self):
        """
        Hand-computed example:
          P(L) = 0.5, P(S) = 0.10, P(G) = 0.25, P(T) = 0.10
          Stage 1 (incorrect):
            numerator   = 0.5 × 0.10 = 0.05
            denominator = 0.05 + 0.5 × (1 - 0.25) = 0.05 + 0.375 = 0.425
            P(L|incorrect) = 0.05 / 0.425 ≈ 0.11765
          Stage 2:
            P(L) = 0.11765 + (1 - 0.11765) × 0.10 ≈ 0.11765 + 0.08824 ≈ 0.20588
        """
        state = make_state(0.5)
        params = make_params(p_l0=0.5, p_t=0.10, p_g=0.25, p_s=0.10)
        result = BKTEngine.update(state, is_correct=False, params=params)

        expected_conditional = 0.05 / 0.425
        expected_p_l = expected_conditional + (1 - expected_conditional) * 0.10

        assert result.p_conditional == pytest.approx(expected_conditional, rel=1e-6)
        assert result.new_state.p_l  == pytest.approx(expected_p_l, rel=1e-6)

    def test_attempt_and_correct_counts_increment(self):
        """Attempt and correct counts must track accurately."""
        params = defaults()
        state = make_state()
        assert state.attempt_count == 0
        assert state.correct_count == 0

        r1 = BKTEngine.update(state, is_correct=True, params=params)
        assert r1.new_state.attempt_count == 1
        assert r1.new_state.correct_count == 1

        r2 = BKTEngine.update(r1.new_state, is_correct=False, params=params)
        assert r2.new_state.attempt_count == 2
        assert r2.new_state.correct_count == 1  # still 1 correct


# ── Test 7: Invalid inputs raise correctly ────────────────────────────────────


class TestInvalidInputs:
    """Invalid states and parameters must raise specific exceptions, never NaN."""

    def test_degenerate_p_g_plus_p_s_raises(self):
        """P(G) + P(S) >= 1.0 should raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match="P\\(G\\) \\+ P\\(S\\)"):
            BKTSkillParameters(p_l0=0.2, p_t=0.1, p_g=0.6, p_s=0.5)

    def test_p_t_zero_raises(self):
        """P(T) = 0 is outside the open interval (0, 1) and must raise."""
        # Valid small p_t should NOT raise
        params = BKTSkillParameters(p_l0=0.2, p_t=0.0001, p_g=0.25, p_s=0.10)
        assert params.p_t == pytest.approx(0.0001)
        # p_t=0 exactly must raise
        with pytest.raises(InvalidParameterError):
            BKTSkillParameters(p_l0=0.2, p_t=0.0, p_g=0.25, p_s=0.10)

    def test_p_l_outside_bounds_raises(self):
        """BKTState with p_l=0 must raise InvalidStateError."""
        with pytest.raises(InvalidStateError):
            BKTState(student_id="s", skill_id="k", p_l=0.0)

    def test_p_l_above_clamp_raises(self):
        with pytest.raises(InvalidStateError):
            BKTState(student_id="s", skill_id="k", p_l=1.0)

    def test_update_result_has_no_nan(self):
        """No field in BKTUpdateResult should ever be NaN or Inf."""
        params = defaults()
        state = make_state()
        result = BKTEngine.update(state, is_correct=True, params=params)

        for field_val in [result.p_conditional, result.p_g_used, result.p_s_used,
                          result.p_t_used, result.new_state.p_l, result.previous_p_l]:
            assert math.isfinite(field_val), f"Non-finite value in result: {field_val}"
