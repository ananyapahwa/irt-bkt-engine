"""
RAG Engine module.
"""

from .config import RAG_MASTERY_THRESHOLD, MAX_TUTORING_TURNS, TOP_K_CHUNKS, EMBEDDING_MODEL
from .models import TextbookChunk, EmbeddedChunk, RetrievedChunk, ConceptContent, TutoringContext, TutoringResponse, TutoringTurn
from .chunker import chunk_concept_content
from .embedder import embed_chunks, embed_query
from .vector_store import ingest, retrieve, clear_collection
from .tutor import generate_tutoring_response

__all__ = [
    "RAG_MASTERY_THRESHOLD",
    "MAX_TUTORING_TURNS",
    "TOP_K_CHUNKS",
    "EMBEDDING_MODEL",
    "TextbookChunk",
    "EmbeddedChunk",
    "RetrievedChunk",
    "ConceptContent",
    "TutoringContext",
    "TutoringResponse",
    "TutoringTurn",
    "chunk_concept_content",
    "embed_chunks",
    "embed_query",
    "ingest",
    "retrieve",
    "clear_collection",
    "generate_tutoring_response"
]
