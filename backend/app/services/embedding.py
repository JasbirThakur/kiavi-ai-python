import os
import hashlib
import numpy as np
from typing import List
from app.config.settings import EMBEDDING_MODEL_NAME
from app.utils.logger import logger

embedding_model = None

try:
    from sentence_transformers import SentenceTransformer
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, model_kwargs={"local_files_only": True})
        logger.info("✅ Local Embedding Model Loaded Successfully!")
    except Exception:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("✅ Embedding Model Loaded Successfully from HuggingFace!")
except Exception as e:
    logger.warning(f"⚠️ Warning: Could not load SentenceTransformer ({e}). Using deterministic fallback vector generator.")
    embedding_model = None

def get_embedding(text: str) -> List[float]:
    """Generates 384-dimensional vector embedding"""
    if embedding_model is not None:
        try:
            return embedding_model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Embedding encode error: {e}, using fallback.")

    # Deterministic fallback embedding (384-dimensional unit vector based on text hash)
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    vec = rng.randn(384).astype(np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else vec.tolist()

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generates 384-dimensional vector embeddings for a batch of texts rapidly"""
    if not texts:
        return []
    if embedding_model is not None:
        try:
            vectors = embedding_model.encode(texts, batch_size=64, show_progress_bar=False)
            return [v.tolist() for v in vectors]
        except Exception as e:
            logger.error(f"Batch embedding encode error: {e}, falling back to single.")
    return [get_embedding(t) for t in texts]

def compute_cosine_similarity(v1, v2) -> float:
    """Calculates cosine angle between two vector embeddings"""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / norm) if norm > 0 else 0.0

