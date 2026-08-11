"""
session.py — Multi-turn conversation state for the RAG AI tutor.
"""

from typing import Dict, List, Optional
from rag.models import TutoringTurn
from rag.config import MAX_TUTORING_TURNS

class TutoringSession:
    """
    Manages the conversational state of the AI tutor during a session.
    A session groups the student's interactions and limits the number of
    tutoring turns for a specific concept to avoid endless rabbit holes.
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        
        # Track history of tutoring turns.
        # Key: concept_id, Value: List[TutoringTurn]
        self.history: Dict[str, List[TutoringTurn]] = {}
        
    def get_turn_number(self, concept_id: str) -> int:
        """Get the next turn number for a concept (1-indexed)."""
        return len(self.history.get(concept_id, [])) + 1
        
    def is_turn_limit_reached(self, concept_id: str) -> bool:
        """Check if the maximum number of turns has been reached for this concept."""
        return self.get_turn_number(concept_id) > MAX_TUTORING_TURNS
        
    def add_turn(self, turn: TutoringTurn):
        """Record a completed tutoring turn."""
        if turn.concept_id not in self.history:
            self.history[turn.concept_id] = []
        self.history[turn.concept_id].append(turn)
        
    def get_history(self, concept_id: str) -> List[TutoringTurn]:
        """Get the full tutoring history for a concept in this session."""
        return self.history.get(concept_id, [])
        
    def reset_concept(self, concept_id: str):
        """Reset the conversation history for a concept (e.g., when they pass a new question on it)."""
        if concept_id in self.history:
            self.history[concept_id] = []
