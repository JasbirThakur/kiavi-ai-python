import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

print("Loading Embedding Model...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, model_kwargs={"local_files_only": True})
except Exception:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding Model Loaded Successfully!")

def get_embedding(text: str) -> list:
    """Generates 384-dimensional vector embedding"""
    return embedding_model.encode(text).tolist()

def compute_cosine_similarity(v1, v2) -> float:
    """Calculates cosine angle between two vector embeddings"""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / norm) if norm > 0 else 0.0
