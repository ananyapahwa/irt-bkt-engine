from dataclasses import dataclass
from typing import Literal
from bkt.engine import BKTUpdateResult

EventKind = Literal["CONTINUE", "RAG_REMEDIATION", "MASTERY_ACHIEVED", "GUESS_FLAGGED"]

@dataclass
class SystemEvent:
    kind: EventKind
    p_l: float
    message: str

def classify_event(result: BKTUpdateResult, rag_threshold: float = 0.70) -> SystemEvent:
    if result.new_state.is_mastered:
        return SystemEvent("MASTERY_ACHIEVED", result.new_state.p_l, "🎓 MASTERY ACHIEVED")
    if result.was_flagged_guess:
        return SystemEvent("GUESS_FLAGGED", result.new_state.p_l, "⚡ GUESS FLAGGED (P(G) inflated)")
    if result.new_state.p_l < rag_threshold:
        return SystemEvent("RAG_REMEDIATION", result.new_state.p_l, "🚨 RAG REMEDIATION TRIGGERED")
    return SystemEvent("CONTINUE", result.new_state.p_l, "Continue")
