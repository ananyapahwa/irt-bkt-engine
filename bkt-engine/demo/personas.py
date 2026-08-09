import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import List, Dict

from examples.mock_orchestrator import (
    IRTStudentProfile,
    IRTItemBank,
    ConceptIRTInfo,
    PracticeQuestion,
)

@dataclass
class Persona:
    id: str
    name: str
    description: str
    student: IRTStudentProfile
    item_bank: IRTItemBank
    concept_info: Dict[str, ConceptIRTInfo]
    session: List[PracticeQuestion]

def build_personas() -> List[Persona]:
    # Concept for all
    concept_math = ConceptIRTInfo(
        concept_id="algebra_01",
        item_difficulties=[-1.0, 0.0, 1.0, 2.5],
        n_attempted_diagnostic=10,
        n_correct_diagnostic=9, # Default high for persona A, will override
    )

    # Item Bank (same for all personas for simplicity)
    # Bloom levels -> difficulties: remember(-2.0), understand(-1.0), apply(0.0), analyze(1.0), evaluate(2.0), create(2.5)
    item_bank = IRTItemBank(items={
        "q_easy": (1.0, -1.0),     # (a, b) - understand
        "q_medium": (1.1, 0.0),    # apply
        "q_hard": (1.2, 1.0),      # analyze
        "q_very_hard": (1.5, 2.5), # create
    })

    # Persona A: The Struggling Learner
    # High initial diagnostic to get P(L0) > 0.70, then bad performance
    student_a = IRTStudentProfile(
        student_id="student_A",
        theta=-1.2,
        theta_se=0.5,
        converged=True
    )
    # Give them a high diagnostic so they start > 0.70 to show the drop
    concept_math_a = ConceptIRTInfo(
        concept_id="algebra_01",
        item_difficulties=[-1.0, -1.0, -1.0, -1.0], # very easy diagnostic
        n_attempted_diagnostic=10,
        n_correct_diagnostic=9,
    )
    session_a = [
        PracticeQuestion("q_medium", "algebra_01", 4500, True), # 1 correct
        PracticeQuestion("q_medium", "algebra_01", 8000, False), # 2 incorrect
        PracticeQuestion("q_medium", "algebra_01", 9000, False), # 3 incorrect
        PracticeQuestion("q_hard", "algebra_01", 12000, False), # 4 incorrect
        PracticeQuestion("q_hard", "algebra_01", 15000, False), # 5 incorrect
    ]
    persona_a = Persona(
        id="student_A",
        name="The Struggling Learner",
        description="Starts with decent P(L0), but drops below 0.70 due to consecutive failures, triggering RAG remediation.",
        student=student_a,
        item_bank=item_bank,
        concept_info={"algebra_01": concept_math_a},
        session=session_a
    )

    # Persona B: The Lucky Guesser
    # Low/med P(L0), answers a very hard question correctly in very short time (< 1500ms)
    student_b = IRTStudentProfile(
        student_id="student_B",
        theta=-0.8,
        theta_se=0.5,
        converged=True
    )
    concept_math_b = ConceptIRTInfo(
        concept_id="algebra_01",
        item_difficulties=[0.0, 0.0, 1.0, 2.5],
        n_attempted_diagnostic=5,
        n_correct_diagnostic=3,
    )
    session_b = [
        PracticeQuestion("q_easy", "algebra_01", 5000, True),
        PracticeQuestion("q_very_hard", "algebra_01", 800, True), # 800ms < 1500ms RT_GUESS_THRESHOLD
    ]
    persona_b = Persona(
        id="student_B",
        name="The Lucky Guesser",
        description="Answers a very hard item (b=2.5) correctly in <1000ms, triggering anti-gaming Guess Flag.",
        student=student_b,
        item_bank=item_bank,
        concept_info={"algebra_01": concept_math_b},
        session=session_b
    )

    # Persona C: The Fast Learner
    # High P(L0), consistently correct, converges to mastery >= 0.95
    student_c = IRTStudentProfile(
        student_id="student_C",
        theta=1.5,
        theta_se=0.5,
        converged=True
    )
    concept_math_c = ConceptIRTInfo(
        concept_id="algebra_01",
        item_difficulties=[0.0, 1.0, 2.5],
        n_attempted_diagnostic=5,
        n_correct_diagnostic=4,
    )
    session_c = [
        PracticeQuestion("q_medium", "algebra_01", 5000, True),
        PracticeQuestion("q_hard", "algebra_01", 6000, True),
        PracticeQuestion("q_very_hard", "algebra_01", 8000, True),
    ]
    persona_c = Persona(
        id="student_C",
        name="The Fast Learner",
        description="High ability student who consistently answers correctly, converging to mastery.",
        student=student_c,
        item_bank=item_bank,
        concept_info={"algebra_01": concept_math_c},
        session=session_c
    )

    return [persona_a, persona_b, persona_c]
