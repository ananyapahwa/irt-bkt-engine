"""
vector_store.py — ChromaDB wrapper for storing and retrieving textbook chunks.
"""

import os
import chromadb
from typing import List, Optional
from .models import EmbeddedChunk, RetrievedChunk, TextbookChunk
from .config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

# Load client lazily
_client = None

def _get_collection():
    global _client
    if _client is None:
        # Create dir if not exists
        if not os.path.exists(CHROMA_PERSIST_DIR):
            os.makedirs(CHROMA_PERSIST_DIR)
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    return _client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def ingest(embedded_chunks: List[EmbeddedChunk]):
    """Write embedded chunks into ChromaDB."""
    if not embedded_chunks:
        return

    collection = _get_collection()
    
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for ec in embedded_chunks:
        chunk_id = f"{ec.chunk.concept_id}_{ec.chunk.chunk_index}"
        ids.append(chunk_id)
        embeddings.append(ec.embedding)
        documents.append(ec.chunk.text)
        metadatas.append({
            "concept_id": ec.chunk.concept_id,
            "concept_name": ec.chunk.concept_name,
            "chapter": ec.chunk.chapter,
            "chunk_index": ec.chunk.chunk_index,
            "source": ec.chunk.source
        })
        
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

def retrieve(concept_id: str, query: Optional[str] = None, top_k: int = 3) -> List[RetrievedChunk]:
    """
    Retrieve chunks for a given concept_id.
    If query is provided, semantic search is performed (filtered by concept_id).
    If query is None, simply retrieves the first top_k chunks for the concept.
    """
    collection = _get_collection()
    
    # If no query is provided, we still need some embedding to query Chroma, 
    # or we can just fetch. Since chroma's `get` doesn't do top_k easily without vector,
    # we use get with where clause to fetch all chunks for the concept.
    if not query:
        result = collection.get(
            where={"concept_id": concept_id},
            include=["documents", "metadatas"]
        )
        retrieved = []
        if result["ids"]:
            # sort by chunk_index to keep order
            combined = list(zip(result["ids"], result["documents"], result["metadatas"]))
            combined.sort(key=lambda x: x[2]["chunk_index"])
            for r_id, r_doc, r_meta in combined[:top_k]:
                chunk = TextbookChunk(
                    concept_id=r_meta["concept_id"],
                    concept_name=r_meta["concept_name"],
                    chapter=r_meta["chapter"],
                    chunk_index=r_meta["chunk_index"],
                    text=r_doc,
                    source=r_meta["source"]
                )
                retrieved.append(RetrievedChunk(chunk=chunk, distance=0.0))
        return retrieved
        
    else:
        # Semantic search with query
        from .embedder import embed_query
        query_embedding = embed_query(query)
        
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"concept_id": concept_id},
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved = []
        if result["ids"] and len(result["ids"][0]) > 0:
            for i in range(len(result["ids"][0])):
                r_id = result["ids"][0][i]
                r_doc = result["documents"][0][i]
                r_meta = result["metadatas"][0][i]
                r_dist = result["distances"][0][i]
                
                chunk = TextbookChunk(
                    concept_id=r_meta["concept_id"],
                    concept_name=r_meta["concept_name"],
                    chapter=r_meta["chapter"],
                    chunk_index=r_meta["chunk_index"],
                    text=r_doc,
                    source=r_meta["source"]
                )
                retrieved.append(RetrievedChunk(chunk=chunk, distance=r_dist))
                
        return retrieved

def clear_collection():
    """Clear all data from the collection (for re-ingestion)."""
    global _client
    if _client is not None:
        try:
            _client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass
    # Re-create
    _get_collection()
