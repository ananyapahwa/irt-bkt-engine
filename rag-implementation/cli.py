"""
cli.py — Command-line interface for the RAG engine.
"""

import argparse
import sys
from typing import Optional, Sequence

from rag.models import TutoringContext
from rag.vector_store import ingest, retrieve, clear_collection
from rag.embedder import embed_chunks
from rag.chunker import chunk_concept_content
from rag.tutor import generate_tutoring_response
from data.seed_content import get_seed_content

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synapse RAG Engine CLI")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ingest", 
        action="store_true", 
        help="Clear vector store and ingest synthetic textbook data"
    )
    group.add_argument(
        "--retrieve", 
        metavar="CONCEPT_ID",
        help="Retrieve top chunks for a given concept ID (e.g. 'E01')"
    )
    group.add_argument(
        "--tutor",
        metavar="CONCEPT_ID",
        help="Test the AI tutor response for a given concept ID"
    )
    
    return parser

def do_ingest():
    print("Clearing existing Chroma collection...")
    clear_collection()
    
    print("Loading synthetic seed content...")
    contents = get_seed_content()
    
    print(f"Chunking and embedding {len(contents)} concepts...")
    for content in contents:
        chunks = chunk_concept_content(content)
        embedded_chunks = embed_chunks(chunks)
        ingest(embedded_chunks)
        print(f"  Ingested '{content.concept_name}' ({len(chunks)} chunks)")
        
    print("Ingestion complete.")

def do_retrieve(concept_id: str):
    print(f"Retrieving chunks for concept_id: {concept_id}...")
    chunks = retrieve(concept_id=concept_id, top_k=3)
    
    if not chunks:
        print("No chunks found. Did you run --ingest first?")
        return
        
    for i, c in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} (Index: {c.chunk.chunk_index}) ---")
        print(f"Source: {c.chunk.source}")
        print(f"Text: {c.chunk.text}")

def do_tutor(concept_id: str):
    print(f"Retrieving chunks for concept_id: {concept_id}...")
    chunks = retrieve(concept_id=concept_id, top_k=3)
    
    if not chunks:
        print("No chunks found. Did you run --ingest first?")
        return
        
    print("Building tutor context...")
    context = TutoringContext(
        concept_id=concept_id,
        concept_name=chunks[0].chunk.concept_name if chunks else concept_id,
        student_answer="I thought voltage causes resistance.",
        correct_answer="Voltage is potential difference; resistance opposes current.",
        misconception_tag="confuses_voltage_and_resistance",
        mastery_probability=0.45,
        theta=-0.5, # slightly struggling
        retrieved_chunks=chunks,
        turn_number=1
    )
    
    print("Generating tutor response (this may take a moment if calling LLM)...")
    response = generate_tutoring_response(context)
    print("\n--- Tutor Response ---")
    print(response.response_text)
    print("\n----------------------")

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    if args.ingest:
        do_ingest()
    elif args.retrieve:
        do_retrieve(args.retrieve)
    elif args.tutor:
        do_tutor(args.tutor)
        
    return 0
