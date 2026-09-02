"""Sentence embedding generator for LocalGPT RAG."""

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_embedding_model_instance = None


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Singleton getter for SentenceTransformer embedding model."""
    global _embedding_model_instance
    if _embedding_model_instance is None:
        _embedding_model_instance = SentenceTransformer(model_name)
    return _embedding_model_instance


def embed_texts(texts: List[str], model: SentenceTransformer = None) -> np.ndarray:
    """Generate normalized sentence embeddings for a list of text strings."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    if model is None:
        model = get_embedding_model()

    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str, model: SentenceTransformer = None) -> np.ndarray:
    """Generate normalized embedding for a single search query."""
    if model is None:
        model = get_embedding_model()

    emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    return np.array(emb[0], dtype=np.float32)
