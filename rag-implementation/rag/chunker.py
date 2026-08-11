"""
chunker.py — Concept-scoped chunking logic.
"""

from typing import List
from .models import ConceptContent, TextbookChunk

def chunk_concept_content(content: ConceptContent) -> List[TextbookChunk]:
    """
    Split a concept's raw content into TextbookChunks.
    For this implementation, each section in ConceptContent.sections becomes
    exactly one chunk. This follows the textbook's own structural boundaries
    rather than arbitrary token counts.
    """
    chunks = []
    for idx, section_text in enumerate(content.sections):
        chunk = TextbookChunk(
            concept_id=content.concept_id,
            concept_name=content.concept_name,
            chapter=content.chapter,
            chunk_index=idx,
            text=section_text.strip(),
            source=f"NCERT Class 9 Science - {content.chapter}"
        )
        chunks.append(chunk)
    return chunks
