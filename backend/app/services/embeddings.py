"""Sentence embedding generator for LocalGPT RAG."""

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from typing import List, Union
import numpy as np

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_embedding_model_instance = None


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Singleton getter for SentenceTransformer embedding model."""
    global _embedding_model_instance
    if _embedding_model_instance is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model_instance = SentenceTransformer(model_name)
        except Exception as e:
            print(f"[RAG Embedding Load Notice]: {e}")
            _embedding_model_instance = None
    return _embedding_model_instance


def embed_texts(texts: List[str], model=None) -> np.ndarray:
    """Generate normalized sentence embeddings for a list of text strings with fast batching."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    if model is None:
        model = get_embedding_model()

    if model is not None:
        embeddings = model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.ascontiguousarray(embeddings, dtype=np.float32)
    else:
        # Fast TF-IDF / Hash fallback if sentence-transformers is loading
        vectors = []
        for text in texts:
            vec = np.zeros(384, dtype=np.float32)
            for i, word in enumerate(text.lower().split()[:384]):
                vec[hash(word) % 384] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


def embed_query(query: str, model=None) -> np.ndarray:
    """Generate normalized embedding for a single search query."""
    if model is None:
        model = get_embedding_model()

    if model is not None:
        emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        return np.ascontiguousarray(emb[0], dtype=np.float32)
    else:
        vec = np.zeros(384, dtype=np.float32)
        for word in query.lower().split()[:384]:
            vec[hash(word) % 384] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
