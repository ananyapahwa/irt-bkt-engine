"""
test_orchestrator.py — End-to-end integration tests validating the full
IRT → BKT orchestration pipeline as demonstrated in mock_orchestrator.py.

These tests do not touch any database or external service — all IRT data
is injected as in-memory stubs, exactly as the orchestrator would receive
them from the IRT engine in production.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bkt import (
    BKTEngine,
    MASTERY_THRESHOLD,
    derive_p_g_from_irt,
    derive_p_s_from_irt,
    from_irt,
    seed_p_l0_from_irt,
)
from bkt.config import (
    DEFAULT_P_L0,
    MASTERY_PRIOR_STRENGTH,
    MASTERY_REFERENCE_DISCRIMINATION,
)

# Import the orchestrator from examples/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from mock_orchestrator import (
    BKTOrchestrator,
    ConceptIRTInfo,
    IRTItemBank,
    IRTStudentProfile,
    PracticeQuestion,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def average_student() -> IRTStudentProfile:
    return IRTStudentProfile(student_id="test_s1", theta=0.0, theta_se=0.5, converged=True)


@pytest.fixture
def strong_student() -> IRTStudentProfile:
    return IRTStudentProfile(student_id="test_s2", theta=2.0, theta_se=0.3, converged=True)


@pytest.fixture
def weak_student() -> IRTStudentProfile:
    return IRTStudentProfile(student_id="test_s3", theta=-2.0, theta_se=0.6, converged=True)


@pytest.fixture
def simple_item_bank() -> IRTItemBank:
    return IRTItemBank(items={
        "q1": (1.0, -1.0),   # easy item (Understand level)
        "q2": (1.0,  0.0),   # neutral (Apply)
        "q3": (1.0,  1.5),   # hard (Analyze/Evaluate)
    })


@pytest.fixture
def concept_info() -> dict:
    return {
        "skill_a": ConceptIRTInfo(
            concept_id="skill_a",
            item_difficulties=[-1.0, 0.0, 1.5],
            n_attempted_diagnostic=3,
            n_correct_diagnostic=2,
        ),
    }


# ── Test 1: IRT-seeded initial states ─────────────────────────────────────────


class TestIRTSeededInitialStates:
    """Initial states must be seeded from IRT mastery initializer output."""

    def test_initial_p_l_in_seed_bounds(self, average_student, simple_item_bank, concept_info):
        from bkt.config import SEED_PRIOR_MIN, SEED_PRIOR_MAX
        orch = BKTOrchestrator(average_student, simple_item_bank, concept_info)
        for state in orch.states.values():
            assert SEED_PRIOR_MIN <= state.p_l <= SEED_PRIOR_MAX

    def test_strong_student_has_higher_initial_p_l(
        self, strong_student, weak_student, simple_item_bank, concept_info
    ):
        """Higher θ → higher IRT-seeded P(L0) for the same concept."""
        orch_strong = BKTOrchestrator(strong_student, simple_item_bank, concept_info)
        orch_weak   = BKTOrchestrator(weak_student,   simple_item_bank, concept_info)

        p_l_strong = orch_strong.states["skill_a"].p_l
        p_l_weak   = orch_weak.states["skill_a"].p_l

        assert p_l_strong > p_l_weak, (
            f"Strong student P(L0)={p_l_strong:.4f} should exceed "
            f"weak student P(L0)={p_l_weak:.4f}"
        )


# ── Test 2: Per-item IRT override hooks ───────────────────────────────────────


class TestPerItemIRTOverrides:
    """The orchestrator must inject per-item P(G) and P(S) overrides."""

    def test_easy_item_gives_low_p_g_override(
        self, strong_student, simple_item_bank, concept_info
    ):
        """For a strong student on q1 (easy item), override_p_g must be < default."""
        orch = BKTOrchestrator(strong_student, simple_item_bank, concept_info)
        q = PracticeQuestion("q1", "skill_a", response_time_ms=5000, is_correct=True)
        result = orch.process_answer(q)

        from bkt.config import DEFAULT_P_G
        assert result.p_g_used < DEFAULT_P_G, (
            f"Easy item for strong student should have P(G) < {DEFAULT_P_G:.4f}, "
            f"got {result.p_g_used:.4f}"
        )

    def test_hard_item_gives_higher_p_g_override(
        self, weak_student, simple_item_bank, concept_info
    ):
        """For a weak student on q3 (hard item), override_p_g should be near base default."""
        orch = BKTOrchestrator(weak_student, simple_item_bank, concept_info)
        q = PracticeQuestion("q3", "skill_a", response_time_ms=5000, is_correct=True)
        result = orch.process_answer(q)

        # Hard item for weak student → P(G)_eff ≈ base default
        from bkt.config import DEFAULT_P_G, P_G_IRT_FLOOR
        assert result.p_g_used > P_G_IRT_FLOOR

    def test_override_p_g_applied_flag(self, average_student, simple_item_bank, concept_info):
        """When item is in the item bank, override_p_g_applied must be True."""
        orch = BKTOrchestrator(average_student, simple_item_bank, concept_info)
        q = PracticeQuestion("q2", "skill_a", response_time_ms=4000, is_correct=True)
        result = orch.process_answer(q)
        assert result.override_p_g_applied is True
        assert result.override_p_s_applied is True


# ── Test 3: Full session — correct answers lead to mastery ───────────────────


class TestFullSessionMastery:
    """Repeated correct answers from a strong student should reach mastery."""

    def test_strong_student_masters_easy_concept(self, strong_student, simple_item_bank):
        """A strong student answering q1 (easy) repeatedly should master quickly."""
        concept_info = {
            "easy_skill": ConceptIRTInfo(
                concept_id="easy_skill",
                item_difficulties=[-2.0, -1.5, -1.0],  # very easy
                n_attempted_diagnostic=3,
                n_correct_diagnostic=3,  # perfect in diagnostic
            )
        }
        item_bank = IRTItemBank(items={"q_easy": (1.0, -2.0)})
        orch = BKTOrchestrator(strong_student, item_bank, concept_info)

        state = orch.states["easy_skill"]
        for _ in range(20):
            if state.is_mastered:
                break
            q = PracticeQuestion("q_easy", "easy_skill", response_time_ms=6000, is_correct=True)
            result = orch.process_answer(q)
            state = result.new_state

        assert state.is_mastered, (
            f"Strong student on easy concept should master, final P(L)={state.p_l:.4f}"
        )

    def test_all_correct_session_monotonically_increases(
        self, average_student, simple_item_bank, concept_info
    ):
        """P(L) must monotonically increase across all-correct answers (until mastery)."""
        orch = BKTOrchestrator(average_student, simple_item_bank, concept_info)
        prev_p_l = orch.states["skill_a"].p_l

        for i in range(15):
            q = PracticeQuestion("q2", "skill_a", response_time_ms=5000, is_correct=True)
            result = orch.process_answer(q)
            if result.new_state.is_mastered:
                # Once clamped at P_L_CLAMP_MAX, further increases are not possible.
                break
            assert result.new_state.p_l > prev_p_l, (
                f"P(L) decreased on correct answer at step {i}: "
                f"{prev_p_l:.4f} → {result.new_state.p_l:.4f}"
            )
            prev_p_l = result.new_state.p_l


# ── Test 4: Incorrect answers penalize correctly ──────────────────────────────


class TestIncorrectAnswerInOrchestrator:
    def test_incorrect_decreases_p_l(self, average_student, simple_item_bank, concept_info):
        orch = BKTOrchestrator(average_student, simple_item_bank, concept_info)
        prev_p_l = orch.states["skill_a"].p_l

        q = PracticeQuestion("q2", "skill_a", response_time_ms=8000, is_correct=False)
        result = orch.process_answer(q)
        assert result.new_state.p_l < prev_p_l


# ── Test 5: Guess detection in orchestrator ───────────────────────────────────


class TestGuessDetectionInOrchestrator:
    def test_fast_surprising_answer_flagged(self, weak_student, simple_item_bank, concept_info):
        """Weak student + hard item + very fast correct → should flag as guess."""
        orch = BKTOrchestrator(weak_student, simple_item_bank, concept_info)
        # q3 is hard (b=1.5), weak student has θ=-2.0 → P_irt ≈ 0.017 << 0.30
        q = PracticeQuestion("q3", "skill_a", response_time_ms=300, is_correct=True)
        result = orch.process_answer(q)
        assert result.was_flagged_guess is True

    def test_normal_speed_correct_not_flagged(self, average_student, simple_item_bank, concept_info):
        orch = BKTOrchestrator(average_student, simple_item_bank, concept_info)
        q = PracticeQuestion("q2", "skill_a", response_time_ms=7000, is_correct=True)
        result = orch.process_answer(q)
        assert result.was_flagged_guess is False


# ── Test 6: New concept not in diagnostic ────────────────────────────────────


class TestNewConceptNotInDiagnostic:
    """Concepts not seen in diagnostic must gracefully initialize with IRT defaults."""

    def test_unknown_concept_initializes_without_error(
        self, average_student, simple_item_bank, concept_info
    ):
        orch = BKTOrchestrator(average_student, simple_item_bank, concept_info)
        # "new_skill" was never in concept_info
        q = PracticeQuestion("q2", "new_skill", response_time_ms=5000, is_correct=True)
        result = orch.process_answer(q)
        assert result.new_state.p_l > 0

    def test_unknown_item_uses_concept_params(
        self, average_student, simple_item_bank, concept_info
    ):
        """An item not in the item bank should use concept-level params (no crash)."""
        orch = BKTOrchestrator(average_student, simple_item_bank, concept_info)
        q = PracticeQuestion("q_unknown", "skill_a", response_time_ms=5000, is_correct=True)
        result = orch.process_answer(q)
        # Should not raise; override flags should be False (no item in bank)
        assert result.override_p_g_applied is False
        assert result.override_p_s_applied is False


# ── Test 7: Mastery summary ───────────────────────────────────────────────────


class TestMasterySummary:
    def test_summary_contains_all_concepts(
        self, average_student, simple_item_bank, concept_info
    ):
        orch = BKTOrchestrator(average_student, simple_item_bank, concept_info)
        summary = orch.mastery_summary()
        for concept_id in concept_info:
            assert concept_id in summary

    def test_summary_fields_present(
        self, average_student, simple_item_bank, concept_info
    ):
        orch = BKTOrchestrator(average_student, simple_item_bank, concept_info)
        q = PracticeQuestion("q1", "skill_a", response_time_ms=5000, is_correct=True)
        orch.process_answer(q)
        summary = orch.mastery_summary()
        info = summary["skill_a"]
        assert "p_l" in info
        assert "is_mastered" in info
        assert "attempts" in info
        assert "accuracy" in info
        assert "est_attempts_to_mastery" in info
        assert info["attempts"] == 1
        assert info["accuracy"] == pytest.approx(1.0)
