"""
models.py — Data types for the RAG engine.
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TextbookChunk:
    """A raw chunk of textbook content before embedding."""
    concept_id: str
    concept_name: str
    chapter: str
    chunk_index: int
    text: str
    source: str

@dataclass
class EmbeddedChunk:
    """A textbook chunk with its vector embedding."""
    chunk: TextbookChunk
    embedding: List[float]

@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store, optionally with a similarity score/distance."""
    chunk: TextbookChunk
    distance: Optional[float] = None

@dataclass
class ConceptContent:
    """Raw content entry for a concept, used during ingestion."""
    concept_id: str
    concept_name: str
    chapter: str
    sections: List[str]  # Each section will become one TextbookChunk

@dataclass
class TutoringContext:
    """All context required to generate a tutor response."""
    concept_id: str
    concept_name: str
    student_answer: str
    correct_answer: str
    misconception_tag: Optional[str]
    mastery_probability: float
    theta: float
    retrieved_chunks: List[RetrievedChunk]
    turn_number: int = 1

@dataclass
class TutoringResponse:
    """The output from the AI tutor."""
    response_text: str
    retrieved_chunk_ids: List[str]
    turn_number: int

@dataclass
class TutoringTurn:
    """Audit record for a single tutoring turn (to be saved to the database)."""
    session_id: str
    concept_id: str
    student_answer: Optional[str]
    misconception_tag: Optional[str]
    retrieved_chunk_ids: List[str]
    theta: Optional[float]
    mastery_at_turn: float
    tutor_response: str
    turn_number: int
