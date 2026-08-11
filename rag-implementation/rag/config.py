"""
config.py — Configuration for the RAG engine.
"""

import os
from dotenv import load_dotenv

# Load .env file if present — use the .env in the rag-implementation directory
# regardless of the working directory the server is started from.
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(_env_path)

# ── RAG tutoring triggers and constraints ─────────────────────────────────────
# The mastery threshold below which RAG tutoring is triggered.
# This is set to 0.85, lower than BKT's formal mastery threshold of 0.95,
# to catch concepts that are weak but not formally "unmastered" before they
# become a big problem.
RAG_MASTERY_THRESHOLD = 0.85

# Maximum number of tutoring turns allowed per weak-concept cycle.
MAX_TUTORING_TURNS = 5

# Maximum number of words for the tutor response.
MAX_TUTOR_RESPONSE_WORDS = 120

# ── Vector Store and Embeddings ───────────────────────────────────────────────
# The embedding model to use for chunking.
# all-MiniLM-L6-v2 is small, fast, and sufficient for general text.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of chunks to retrieve for a given concept.
TOP_K_CHUNKS = 3

# ChromaDB collection name
CHROMA_COLLECTION_NAME = "ncert_class9_science"

# ChromaDB persist directory
# Default is ./chroma_db in the rag-implementation folder.
_default_db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", _default_db_path)

# ── LLM Configuration (Ollama) ────────────────────────────────────────────────
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
