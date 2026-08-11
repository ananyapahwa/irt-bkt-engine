import sys
import os

# Ensure the rag package can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.seed_content import get_seed_content
from rag.chunker import chunk_concept_content
from rag.embedder import embed_chunks
from rag.vector_store import ingest, clear_collection

def main():
    print("Clearing existing vector database collection...")
    clear_collection()
    
    print("Loading seed content...")
    concepts = get_seed_content()
    
    total_chunks_ingested = 0
    for concept in concepts:
        print(f"Processing concept: {concept.concept_id} - {concept.concept_name}")
        chunks = chunk_concept_content(concept)
        print(f"  Generated {len(chunks)} chunks.")
        
        embedded = embed_chunks(chunks)
        ingest(embedded)
        total_chunks_ingested += len(embedded)
        
    print(f"\nSeeding complete! Successfully ingested {total_chunks_ingested} chunks into ChromaDB.")

if __name__ == "__main__":
    main()
