"""
embedder.py — sentence-transformers wrapper for generating text embeddings.
"""

from typing import List
from sentence_transformers import SentenceTransformer
from .models import TextbookChunk, EmbeddedChunk
from .config import EMBEDDING_MODEL

# Load model lazily
_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed_chunks(chunks: List[TextbookChunk]) -> List[EmbeddedChunk]:
    """Generate embeddings for a list of textbook chunks."""
    if not chunks:
        return []
    
    model = _get_model()
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    
    embedded_chunks = []
    for i, chunk in enumerate(chunks):
        # convert numpy array to list
        embedded_chunks.append(EmbeddedChunk(chunk=chunk, embedding=embeddings[i].tolist()))
        
    return embedded_chunks

def embed_query(query: str) -> List[float]:
    """Generate embedding for a single query string."""
    model = _get_model()
    embedding = model.encode([query], show_progress_bar=False)[0]
    return embedding.tolist()
