"""
service.py — FastAPI service for the RAG engine.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from rag.vector_store import retrieve, ingest, clear_collection
from rag.embedder import embed_chunks
from rag.chunker import chunk_concept_content
from rag.tutor import generate_tutoring_response
from rag.models import TutoringContext, RetrievedChunk
from data.seed_content import get_seed_content

app = FastAPI(title="Synapse RAG Engine API")

class RetrieveRequest(BaseModel):
    concept_id: str
    query: Optional[str] = None
    top_k: int = 3

class TutorRequest(BaseModel):
    concept_id: str
    concept_name: str
    student_answer: str
    correct_answer: str
    misconception_tag: Optional[str] = None
    mastery_probability: float
    theta: float
    turn_number: int = 1

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "rag-engine"}

@app.post("/ingest")
def ingest_seed_data():
    """Clear collection and ingest all seed data."""
    try:
        clear_collection()
        contents = get_seed_content()
        total_chunks = 0
        for content in contents:
            chunks = chunk_concept_content(content)
            embedded = embed_chunks(chunks)
            ingest(embedded)
            total_chunks += len(chunks)
        return {"status": "success", "concepts_ingested": len(contents), "chunks_ingested": total_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve")
def retrieve_chunks(req: RetrieveRequest):
    """Retrieve top-k chunks for a concept."""
    try:
        chunks = retrieve(concept_id=req.concept_id, query=req.query, top_k=req.top_k)
        return {"chunks": [c.chunk.__dict__ for c in chunks]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tutor")
def get_tutoring(req: TutorRequest):
    """Trigger the AI tutor for a specific context."""
    try:
        # Retrieve chunks
        chunks = retrieve(concept_id=req.concept_id, top_k=3)
        
        # Build context
        context = TutoringContext(
            concept_id=req.concept_id,
            concept_name=req.concept_name,
            student_answer=req.student_answer,
            correct_answer=req.correct_answer,
            misconception_tag=req.misconception_tag,
            mastery_probability=req.mastery_probability,
            theta=req.theta,
            retrieved_chunks=chunks,
            turn_number=req.turn_number
        )
        
        # Generate response
        response = generate_tutoring_response(context)
        
        return {
            "response_text": response.response_text,
            "retrieved_chunk_ids": response.retrieved_chunk_ids,
            "turn_number": response.turn_number
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
