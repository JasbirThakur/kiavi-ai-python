import os
import hashlib
import numpy as np
from config import EMBEDDING_MODEL_NAME

print("Loading Embedding Model...")
embedding_model = None

try:
    from sentence_transformers import SentenceTransformer
    try:
        # First attempt: load from local cache if already present
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, model_kwargs={"local_files_only": True})
        print("✅ Local Embedding Model Loaded Successfully!")
    except Exception:
        # Second attempt: load/download from HuggingFace
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("✅ Embedding Model Loaded Successfully from HuggingFace!")
except Exception as e:
    print(f"⚠️ Warning: Could not load SentenceTransformer ({e}).")
    print("ℹ️ Using deterministic fallback vector generator (backend will continue running safely).")
    embedding_model = None

def get_embedding(text: str) -> list:
    """Generates 384-dimensional vector embedding"""
    if embedding_model is not None:
        try:
            return embedding_model.encode(text).tolist()
        except Exception as e:
            print(f"Embedding encode error: {e}, using fallback.")

    # Deterministic fallback embedding (384-dimensional unit vector based on text hash)
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    vec = rng.randn(384).astype(np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else vec.tolist()

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generates 384-dimensional vector embeddings for a batch of texts rapidly"""
    if not texts:
        return []
    if embedding_model is not None:
        try:
            vectors = embedding_model.encode(texts, batch_size=64, show_progress_bar=False)
            return [v.tolist() for v in vectors]
        except Exception as e:
            print(f"Batch embedding encode error: {e}, falling back to single.")
    return [get_embedding(t) for t in texts]

def compute_cosine_similarity(v1, v2) -> float:
    """Calculates cosine angle between two vector embeddings"""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / norm) if norm > 0 else 0.0
