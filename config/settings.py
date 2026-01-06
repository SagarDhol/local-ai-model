import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_PATH = BASE_DIR / "vector_store" / "faiss_index"

# Model settings
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# RAG settings
TOP_K_RESULTS = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH.parent, exist_ok=True)