"""
mock_orchestrator.py — End-to-end demonstration of how an orchestrator service
wires the IRT engine's output into the BKT engine for a live student session.

This script simulates a realistic scenario:
  1. IRT pipeline has already run → ThetaResult + per-item parameters are known.
  2. Student's initial mastery has been seeded → P(L0) per concept.
  3. Student takes a practice quiz → questions arrive one at a time.
  4. For each question, the orchestrator:
       a. Fetches the item's IRT parameters.
       b. Derives per-item BKT parameters (P(G), P(S) from IRT).
       c. Calls BKTEngine.update() with the observation + IRT overrides.
       d. Checks if the student has reached mastery.
  5. Final summary is printed.

This is the ONLY place that knows about both the IRT engine and the BKT engine.
Neither engine imports the other — this orchestrator is the integration seam.
"""

from __future__ import annotations

import sys
import os
# Allow running from examples/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ── BKT engine imports ────────────────────────────────────────────────────────
from bkt import (
    BKTEngine,
    BKTSkillParameters,
    BKTState,
    BKTUpdateResult,
    MASTERY_THRESHOLD,
    derive_p_g_from_irt,
    derive_p_s_from_irt,
    expected_attempts_to_mastery,
    from_irt,
    irt_p_correct,
    seed_p_l0_from_irt,
)
from bkt.config import (
    DEFAULT_P_L0,
    DEFAULT_P_T,
    MASTERY_REFERENCE_DISCRIMINATION,
    MASTERY_PRIOR_STRENGTH,
)

# ── IRT output stubs (would be imported from irt.theta / irt.item_parameters) ─
# In production, replace these with the real IRT engine types.


@dataclass
class IRTStudentProfile:
    """Represents the output of the IRT pipeline for one student."""
    student_id: str
    theta: float              # from ThetaResult.theta
    theta_se: Optional[float] # from ThetaResult.standard_error
    converged: bool


@dataclass
class IRTItemBank:
    """Maps question_id → (discrimination a, difficulty b)."""
    items: Dict[str, Tuple[float, float]]  # question_id → (a, b)

    def get(self, question_id: str) -> Optional[Tuple[float, float]]:
        return self.items.get(question_id)


@dataclass
class ConceptIRTInfo:
    """Aggregated IRT info for a concept — needed for P(T) and P(L0) seeding."""
    concept_id: str
    item_difficulties: List[float]   # b-values for all items in this concept
    n_attempted_diagnostic: int      # how many diagnostic items the student answered
    n_correct_diagnostic: int        # how many were correct


@dataclass
class PracticeQuestion:
    """One question in a live practice session."""
    question_id: str
    concept_id: str
    response_time_ms: int
    is_correct: bool


# ── Orchestrator class ────────────────────────────────────────────────────────


class BKTOrchestrator:
    """
    Wires IRT engine output → BKT engine input for a student session.

    Responsibilities:
    - Maintains a BKTState per concept for the student.
    - For each answered question, resolves IRT parameters and derives
      dynamic BKT parameters (P(G), P(S) per item; P(L0) from diagnostic).
    - Calls BKTEngine.update() and tracks mastery.
    - Never computes any BKT math itself — delegates everything to engine.py.
    - Never computes any IRT math itself — consumes pre-computed IRT outputs.
    """

    def __init__(
        self,
        student: IRTStudentProfile,
        item_bank: IRTItemBank,
        concept_info: Dict[str, ConceptIRTInfo],
        mastery_threshold: float = MASTERY_THRESHOLD,
    ) -> None:
        self.student = student
        self.item_bank = item_bank
        self.concept_info = concept_info
        self.mastery_threshold = mastery_threshold

        # Initialize one BKTState per concept from IRT-derived P(L0)
        self.states: Dict[str, BKTState] = {}
        self.params: Dict[str, BKTSkillParameters] = {}
        self._initialize_states()

    def _initialize_states(self) -> None:
        """Seed initial BKT states from IRT pipeline output."""
        for concept_id, info in self.concept_info.items():
            # P(L0) from IRT: Beta-Bernoulli shrinkage blend of diagnostic performance
            p_l0 = seed_p_l0_from_irt(
                theta=self.student.theta,
                bloom_difficulties=info.item_difficulties,
                n_correct=info.n_correct_diagnostic,
                n_attempted=info.n_attempted_diagnostic,
                a_ref=MASTERY_REFERENCE_DISCRIMINATION,
                prior_strength=MASTERY_PRIOR_STRENGTH,
            )

            # Mean item difficulty for P(T) scaling
            mean_b = (
                sum(info.item_difficulties) / len(info.item_difficulties)
                if info.item_difficulties else 0.0
            )

            # Build skill parameters using concept-level IRT data.
            # P(G) and P(S) here use concept mean difficulty + student theta
            # as a baseline — they will be refined per-item during the session.
            params = from_irt(
                theta=self.student.theta,
                b=mean_b,             # concept mean b as baseline
                a=MASTERY_REFERENCE_DISCRIMINATION,
                mean_b=mean_b,
                p_l0=p_l0,
                skill_id=concept_id,
            )
            self.params[concept_id] = params
            self.states[concept_id] = BKTEngine.initial_state(
                student_id=self.student.student_id,
                skill_id=concept_id,
                params=params,
                mastery_threshold=self.mastery_threshold,
            )

    def process_answer(self, question: PracticeQuestion) -> BKTUpdateResult:
        """Process one student answer and update the relevant concept's BKT state.

        For each answer, the orchestrator:
        1. Looks up the item's IRT parameters from the item bank.
        2. Derives per-item P(G) and P(S) from student θ + item (a, b).
        3. Calls BKTEngine.update() with those per-item overrides.
        4. Updates the stored state.

        Parameters
        ----------
        question : PracticeQuestion
            The answered question with correctness and response time.

        Returns
        -------
        BKTUpdateResult
            Full audit record of this update.
        """
        concept_id = question.concept_id
        state = self.states.get(concept_id)
        params = self.params.get(concept_id)

        if state is None or params is None:
            # Concept not seen in diagnostic — create a state with defaults
            params = from_irt(
                theta=self.student.theta,
                b=0.0,  # unknown concept difficulty → assume neutral
                a=MASTERY_REFERENCE_DISCRIMINATION,
                skill_id=concept_id,
            )
            self.params[concept_id] = params
            state = BKTEngine.initial_state(
                student_id=self.student.student_id,
                skill_id=concept_id,
                params=params,
                mastery_threshold=self.mastery_threshold,
            )
            self.states[concept_id] = state

        # Per-item IRT parameter overrides (the IRT hook)
        override_p_g: Optional[float] = None
        override_p_s: Optional[float] = None
        item_a: Optional[float] = None
        item_b: Optional[float] = None

        item_params = self.item_bank.get(question.question_id)
        if item_params is not None:
            item_a, item_b = item_params
            override_p_g = derive_p_g_from_irt(self.student.theta, item_b, item_a)
            override_p_s = derive_p_s_from_irt(self.student.theta, item_b, item_a)

        result = BKTEngine.update(
            state=state,
            is_correct=question.is_correct,
            params=params,
            override_p_g=override_p_g,
            override_p_s=override_p_s,
            response_time_ms=question.response_time_ms,
            theta=self.student.theta,
            difficulty=item_b,
            discrimination=item_a,
            mastery_threshold=self.mastery_threshold,
        )
        self.states[concept_id] = result.new_state
        return result

    def mastery_summary(self) -> Dict[str, dict]:
        """Return a mastery summary for all tracked concepts."""
        summary = {}
        for concept_id, state in self.states.items():
            params = self.params[concept_id]
            est = expected_attempts_to_mastery(
                state.p_l, params.p_t, params.p_g, params.p_s, self.mastery_threshold
            )
            summary[concept_id] = {
                "p_l": round(state.p_l, 4),
                "is_mastered": state.is_mastered,
                "attempts": state.attempt_count,
                "accuracy": round(state.accuracy, 3) if state.accuracy is not None else None,
                "est_attempts_to_mastery": est,
            }
        return summary


# ── Mock session ─────────────────────────────────────────────────────────────


def run_mock_session() -> None:
    """Simulate a student session and print the mastery trajectory."""

    print("=" * 70)
    print("BKT Orchestrator — Mock Student Session")
    print("=" * 70)

    # ── IRT pipeline output (would come from the IRT engine in production) ──
    student = IRTStudentProfile(
        student_id="student_42",
        theta=0.65,          # slightly above-average ability
        theta_se=0.38,
        converged=True,
    )

    item_bank = IRTItemBank(items={
        # question_id: (discrimination a, difficulty b)
        # Ohm's Law items (mostly Remember/Understand level → easy)
        "q_ohm_1":  (1.20, -1.5),   # Remember — easy
        "q_ohm_2":  (0.95, -0.8),   # Understand
        "q_ohm_3":  (1.10,  0.0),   # Apply — neutral
        "q_ohm_4":  (0.85,  0.5),   # Apply+
        "q_ohm_5":  (1.30,  1.2),   # Analyze — harder
        # Circuit items (mixed difficulty)
        "q_circ_1": (1.00, -1.0),
        "q_circ_2": (1.15,  0.3),
        "q_circ_3": (0.90,  1.5),   # Evaluate — hard
    })

    concept_info = {
        "ohms_law": ConceptIRTInfo(
            concept_id="ohms_law",
            item_difficulties=[-1.5, -0.8, 0.0, 0.5, 1.2],
            n_attempted_diagnostic=4,
            n_correct_diagnostic=3,   # 75% in diagnostic
        ),
        "circuit_analysis": ConceptIRTInfo(
            concept_id="circuit_analysis",
            item_difficulties=[-1.0, 0.3, 1.5],
            n_attempted_diagnostic=3,
            n_correct_diagnostic=1,   # 33% in diagnostic — weak
        ),
    }

    orchestrator = BKTOrchestrator(
        student=student,
        item_bank=item_bank,
        concept_info=concept_info,
        mastery_threshold=MASTERY_THRESHOLD,
    )

    print(f"\nStudent: {student.student_id}  |  θ = {student.theta:.3f}")
    print(f"Mastery threshold: {MASTERY_THRESHOLD:.0%}")
    print("\n── Initial States (IRT-seeded P(L0)) ──")
    for cid, state in orchestrator.states.items():
        params = orchestrator.params[cid]
        print(f"  {cid:30s}  P(L0)={state.p_l:.4f}  "
              f"P(T)={params.p_t:.4f}  P(G)={params.p_g:.4f}  P(S)={params.p_s:.4f}")

    # ── Simulated practice session ──
    # A mix of correct/incorrect answers with realistic response times.
    # Response times under 1500ms on "surprising" items will trigger guess detection.
    session: List[PracticeQuestion] = [
        # Ohm's Law — mostly getting it right
        PracticeQuestion("q_ohm_1", "ohms_law",       response_time_ms=8000,  is_correct=True),
        PracticeQuestion("q_ohm_2", "ohms_law",       response_time_ms=6500,  is_correct=True),
        PracticeQuestion("q_ohm_3", "ohms_law",       response_time_ms=5200,  is_correct=False),  # slip
        PracticeQuestion("q_ohm_4", "ohms_law",       response_time_ms=9100,  is_correct=True),
        PracticeQuestion("q_ohm_5", "ohms_law",       response_time_ms=12000, is_correct=True),
        PracticeQuestion("q_ohm_1", "ohms_law",       response_time_ms=4500,  is_correct=True),
        PracticeQuestion("q_ohm_2", "ohms_law",       response_time_ms=3800,  is_correct=True),
        PracticeQuestion("q_ohm_3", "ohms_law",       response_time_ms=5500,  is_correct=True),
        # Circuit Analysis — struggling
        PracticeQuestion("q_circ_1", "circuit_analysis", response_time_ms=15000, is_correct=False),
        PracticeQuestion("q_circ_2", "circuit_analysis", response_time_ms=18000, is_correct=False),
        PracticeQuestion("q_circ_3", "circuit_analysis", response_time_ms=900,   is_correct=True),   # suspiciously fast correct
        PracticeQuestion("q_circ_1", "circuit_analysis", response_time_ms=11000, is_correct=True),
        PracticeQuestion("q_circ_2", "circuit_analysis", response_time_ms=9500,  is_correct=False),
        PracticeQuestion("q_circ_3", "circuit_analysis", response_time_ms=13000, is_correct=True),
    ]

    print(f"\n── Session Transcript ({len(session)} questions) ──")
    print(f"{'#':>3}  {'Concept':<25} {'Q':<12} {'Ans':>5}  "
          f"{'P(G)_eff':>9} {'P(S)_eff':>9} {'P(L)_old':>9} {'P(L)_new':>9} {'Guess?':>7}")
    print("-" * 100)

    for i, q in enumerate(session, 1):
        result = orchestrator.process_answer(q)
        guess_flag = "⚠ GUESS" if result.was_flagged_guess else ""
        mastered_flag = " ✓ MASTERED" if result.new_state.is_mastered else ""
        print(
            f"{i:>3}  {q.concept_id:<25} {q.question_id:<12} "
            f"{'✓' if q.is_correct else '✗':>5}  "
            f"{result.p_g_used:>9.4f} {result.p_s_used:>9.4f} "
            f"{result.previous_p_l:>9.4f} {result.new_state.p_l:>9.4f} "
            f"{guess_flag:>7}{mastered_flag}"
        )

    print("\n── Final Mastery Summary ──")
    summary = orchestrator.mastery_summary()
    for concept_id, info in summary.items():
        status = "✓ MASTERED" if info["is_mastered"] else f"~{info['est_attempts_to_mastery']} more to go"
        print(f"  {concept_id:30s}  P(L)={info['p_l']:.4f}  "
              f"acc={info['accuracy']:.1%}  {status}")


if __name__ == "__main__":
    run_mock_session()
