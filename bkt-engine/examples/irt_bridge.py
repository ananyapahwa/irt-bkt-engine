"""
irt_bridge.py — Standalone reference showing exactly how IRT engine output
maps to BKT engine input, with full inline math documentation.

This file is a REFERENCE EXAMPLE, not production code. It demonstrates
the formulas and shows how an orchestrator service would extract data
from the IRT engine's output types (ThetaResult, QuestionIRTParameters,
MasteryInitializationResult) and pass them into the BKT engine.

Reading this file should answer the question: "given the IRT pipeline's
output, how do I construct BKT parameters for a student answering a
specific item on a specific skill?"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

# ── Minimal stubs for IRT engine types ───────────────────────────────────────
# In production, these would be imported directly from the IRT engine:
#   from irt.theta import ThetaResult
#   from irt.item_parameters import QuestionIRTParameters
#   from irt.mastery_initializer import MasteryInitializationResult, ConceptMastery
#
# Here we define lightweight dataclass stubs so this file is self-contained
# and runnable without installing the IRT engine as a dependency.


@dataclass
class IRTThetaResult:
    """Stub for irt.theta.ThetaResult."""
    theta: float           # student ability on logit scale, typically in [-4, 4]
    converged: bool
    standard_error: Optional[float]
    n_responses: int


@dataclass
class IRTItemParameters:
    """Stub for irt.item_parameters.QuestionIRTParameters."""
    question_id: str
    discrimination: float  # a — how well the item separates strong/weak students
    difficulty: float      # b — logit-scale difficulty (same scale as theta)


@dataclass
class IRTConceptMastery:
    """Stub for irt.mastery_initializer.ConceptMastery."""
    concept_id: str
    initial_mastery: float    # the shrinkage-blended prior — this IS P(L0)
    observed_accuracy: float
    theta_implied_accuracy: float
    n_attempted: int
    n_correct: int
    weight_observed: float


# ── BKT engine imports ────────────────────────────────────────────────────────
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
    MASTERY_REFERENCE_DISCRIMINATION,
    MASTERY_THRESHOLD,
    P_T_CLAMP_MAX,
    P_T_CLAMP_MIN,
    SEED_PRIOR_MAX,
    SEED_PRIOR_MIN,
)


# ── Bridge function 1: P(L0) from MasteryInitializationResult ────────────────


def p_l0_from_mastery_result(concept_mastery: IRTConceptMastery) -> float:
    """Extract P(L0) directly from the IRT engine's mastery initializer output.

    The IRT engine's mastery_initializer.py already computes:
        initial_mastery = shrinkage_blend(observed_accuracy, theta_implied_accuracy)
    clamped to (SEED_PRIOR_MIN, SEED_PRIOR_MAX).

    That value IS the BKT P(L0) — no further computation needed. This
    function is a named pass-through for clarity in orchestrator code.

    Parameters
    ----------
    concept_mastery : IRTConceptMastery
        One concept's entry from MasteryInitializationResult.concept_masteries.

    Returns
    -------
    float in [SEED_PRIOR_MIN, SEED_PRIOR_MAX]
    """
    p_l0 = concept_mastery.initial_mastery
    # Defensive clamp in case IRT engine bounds ever change
    return max(SEED_PRIOR_MIN, min(SEED_PRIOR_MAX, p_l0))


# ── Bridge function 2: P(G) from theta + item parameters ─────────────────────


def p_g_for_item(
    theta_result: IRTThetaResult,
    item: IRTItemParameters,
) -> float:
    """Derive per-item effective P(G) from IRT parameters.

    Mathematical derivation:
        P_irt = σ(a × (θ − b))    [2PL item characteristic curve]
        P(G)_eff = P_G_IRT_BASE × (1 − P_irt) + P_G_IRT_FLOOR

    Intuition:
        - If IRT predicts the student will almost certainly answer correctly
          (P_irt ≈ 1), the effective guess rate collapses to its floor.
          A correct answer here is strong evidence of mastery, not guessing.
        - If IRT predicts near-certain failure (P_irt ≈ 0), the effective
          guess rate rises to its base default. A correct answer on a very
          hard item carries much less mastery signal.

    Example:
        θ = 1.5, b = -0.5 (easy item), a = 1.2
        P_irt = σ(1.2 × 2.0) = σ(2.4) ≈ 0.917
        P(G)_eff = 0.25 × (1 - 0.917) + 0.0625 = 0.25 × 0.083 + 0.0625 ≈ 0.083
        (Much lower than the default 0.25 — the student's P(G) is minimal
         for this easy item given their high ability.)

    Parameters
    ----------
    theta_result : IRTThetaResult
        From the IRT pipeline's estimate_theta().
    item : IRTItemParameters
        From the IRT pipeline's build_question_parameters().

    Returns
    -------
    float — effective P(G) for this student × item pair.
    """
    return derive_p_g_from_irt(
        theta=theta_result.theta,
        b=item.difficulty,
        a=item.discrimination,
    )


# ── Bridge function 3: P(S) from theta + item parameters ─────────────────────


def p_s_for_item(
    theta_result: IRTThetaResult,
    item: IRTItemParameters,
) -> float:
    """Derive per-item effective P(S) from IRT parameters.

    Mathematical derivation:
        P_irt = σ(a × (θ − b))
        P(S)_eff = P_S_IRT_BASE × (1 − P_irt) + P_S_IRT_FLOOR

    Intuition:
        - A high-ability student on an easy item (P_irt ≈ 1) is very unlikely
          to slip → effective P(S) collapses to its floor.
        - A low-ability student on a hard item (P_irt ≈ 0) has a higher
          effective slip rate — even their "known" skills are tested under
          stress by this difficult item.

    Example:
        θ = -1.0, b = 1.5 (hard item), a = 0.8
        P_irt = σ(0.8 × (-2.5)) = σ(-2.0) ≈ 0.119
        P(S)_eff = 0.10 × (1 - 0.119) + 0.05 = 0.10 × 0.881 + 0.05 ≈ 0.138
        (Somewhat higher than the 0.10 default — this hard item has higher
         slip risk for a below-average student.)
    """
    return derive_p_s_from_irt(
        theta=theta_result.theta,
        b=item.difficulty,
        a=item.discrimination,
    )


# ── Bridge function 4: P(T) from concept's mean difficulty ───────────────────


def p_t_for_concept(items_in_concept: List[IRTItemParameters]) -> float:
    """Derive learn-rate P(T) from the mean IRT difficulty of a concept's items.

    Mathematical derivation:
        mean_b = mean over all items [b_i]
        P(T) = DEFAULT_P_T × σ(-mean_b) / σ(0)
             = DEFAULT_P_T × 2 × σ(-mean_b)    [since σ(0) = 0.5]

    The sigmoid normalization ensures mean_b = 0 (neutral difficulty)
    preserves exactly DEFAULT_P_T. Bloom difficulty values from the IRT
    engine range from -2.0 (Remember) to +2.5 (Create):
        mean_b = -2.0 (all-Remember concept): P(T) ≈ DEFAULT_P_T × 1.76 → faster
        mean_b =  0.0 (all-Apply concept):    P(T) = DEFAULT_P_T exactly
        mean_b = +2.5 (all-Create concept):   P(T) ≈ DEFAULT_P_T × 0.24 → slower

    Parameters
    ----------
    items_in_concept : List[IRTItemParameters]
        All items tagged to this concept. Must be non-empty.

    Returns
    -------
    float — P(T) for this concept, in [P_T_CLAMP_MIN, P_T_CLAMP_MAX].
    """
    if not items_in_concept:
        from bkt.config import DEFAULT_P_T
        return DEFAULT_P_T
    mean_b = sum(item.difficulty for item in items_in_concept) / len(items_in_concept)
    return derive_p_t_from_difficulty(mean_b)


# ── Full bridge: assemble BKTSkillParameters from IRT outputs ────────────────


def bkt_params_from_irt(
    theta_result: IRTThetaResult,
    current_item: IRTItemParameters,
    concept_mastery: Optional[IRTConceptMastery],
    concept_items: Optional[List[IRTItemParameters]] = None,
    skill_id: Optional[str] = None,
) -> BKTSkillParameters:
    """Assemble a complete BKTSkillParameters from IRT engine outputs.

    This is the one-stop-shop function the orchestrator calls when it has
    all IRT data available. It wires together the four bridge functions
    above into a single call.

    Parameters
    ----------
    theta_result : IRTThetaResult
        Student ability from the IRT engine.
    current_item : IRTItemParameters
        The specific item being answered now (used for P(G) and P(S)).
    concept_mastery : IRTConceptMastery | None
        This concept's entry from MasteryInitializationResult.concept_masteries.
        If None (e.g. concept not in the diagnostic), P(L0) falls back to
        DEFAULT_P_L0.
    concept_items : List[IRTItemParameters] | None
        All items for this concept (used for P(T) scaling). If None,
        uses the current item's difficulty as an approximation.
    skill_id : str | None
        Identifier for logging.

    Returns
    -------
    BKTSkillParameters — fully IRT-derived, validated.
    """
    p_l0 = p_l0_from_mastery_result(concept_mastery) if concept_mastery else None
    mean_b = (
        sum(item.difficulty for item in concept_items) / len(concept_items)
        if concept_items else None
    )

    return from_irt(
        theta=theta_result.theta,
        b=current_item.difficulty,
        a=current_item.discrimination,
        mean_b=mean_b,
        p_l0=p_l0,
        skill_id=skill_id,
    )


# ── Demo: show the mapping for a concrete student × item pair ─────────────────


if __name__ == "__main__":
    print("=" * 65)
    print("IRT → BKT Parameter Bridge — Derivation Reference")
    print("=" * 65)

    # Simulate a student with above-average ability answering three items
    # of increasing difficulty (Bloom: Remember → Apply → Evaluate)
    theta_result = IRTThetaResult(theta=0.8, converged=True, standard_error=0.4, n_responses=20)

    items = [
        IRTItemParameters("q1", discrimination=1.2, difficulty=-2.0),   # Remember
        IRTItemParameters("q2", discrimination=0.9, difficulty=0.0),    # Apply
        IRTItemParameters("q3", discrimination=1.1, difficulty=2.0),    # Evaluate
    ]

    concept_mastery = IRTConceptMastery(
        concept_id="concept_ohms_law",
        initial_mastery=0.62,
        observed_accuracy=0.70,
        theta_implied_accuracy=0.58,
        n_attempted=10,
        n_correct=7,
        weight_observed=10 / (10 + 3.0),  # K = MASTERY_PRIOR_STRENGTH = 3.0
    )

    print(f"\nStudent θ = {theta_result.theta} (standard_error = {theta_result.standard_error})")
    print(f"Concept initial mastery (from IRT): {concept_mastery.initial_mastery:.4f}  → P(L0)")

    print(f"\n{'Item':<6} {'a':>6} {'b':>6} {'P_irt':>8} {'P(G)_eff':>10} {'P(S)_eff':>10}")
    print("-" * 55)
    for item in items:
        p_irt = irt_p_correct(theta_result.theta, item.difficulty, item.discrimination)
        p_g = p_g_for_item(theta_result, item)
        p_s = p_s_for_item(theta_result, item)
        print(f"{item.question_id:<6} {item.discrimination:>6.2f} {item.difficulty:>6.2f} "
              f"{p_irt:>8.4f} {p_g:>10.4f} {p_s:>10.4f}")

    p_t = p_t_for_concept(items)
    print(f"\nMean item difficulty: {sum(i.difficulty for i in items)/len(items):.2f}")
    print(f"P(T) scaled by concept difficulty: {p_t:.4f}")

    # Full parameter set for answering item q2
    params = bkt_params_from_irt(
        theta_result=theta_result,
        current_item=items[1],
        concept_mastery=concept_mastery,
        concept_items=items,
        skill_id="concept_ohms_law",
    )
    print(f"\nFull BKTSkillParameters for concept_ohms_law / item q2:")
    print(f"  P(L0) = {params.p_l0:.4f}  (from IRT mastery initializer)")
    print(f"  P(T)  = {params.p_t:.4f}  (difficulty-scaled learn rate)")
    print(f"  P(G)  = {params.p_g:.4f}  (IRT-derived per-item guess rate)")
    print(f"  P(S)  = {params.p_s:.4f}  (IRT-derived per-item slip rate)")
    print(f"  irt_derived = {params.irt_derived}")
