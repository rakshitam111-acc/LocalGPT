"""FAISS Vector Store manager for LocalGPT RAG."""

import json
import os
from typing import Any, Dict, List, Optional
import faiss
import numpy as np


class FAISSVectorStore:
    """Manages FAISS index and metadata storage for document chunks."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product on normalized vectors = Cosine Similarity
        self.chunks: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add text chunks and corresponding embedding vectors to FAISS."""
        if len(chunks) == 0 or embeddings.shape[0] == 0:
            return

        # Ensure float32 format
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
        """Search the FAISS index for the top-k most similar document chunks."""
        if self.index.ntotal == 0 or len(self.chunks) == 0:
            return []

        k = min(top_k, self.index.ntotal)
        query_vec = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)

        scores, indices = self.index.search(query_vec, k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                chunk = dict(self.chunks[idx])
                chunk["similarity_score"] = float(score)
                results.append(chunk)

        return results

    def get_indexed_documents(self) -> List[str]:
        """Return unique list of indexed document names."""
        docs = set()
        for c in self.chunks:
            if "source" in c:
                docs.add(c["source"])
        return sorted(list(docs))

    def get_total_chunks(self) -> int:
        """Return total number of chunks currently indexed."""
        return len(self.chunks)

    def clear(self):
        """Reset the vector index and chunks list."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []

    def save(self, directory: str):
        """Save FAISS index and metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        index_file = os.path.join(directory, "index.faiss")
        meta_file = os.path.join(directory, "chunks.json")

        faiss.write_index(self.index, index_file)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load(self, directory: str) -> bool:
        """Load FAISS index and metadata from disk if they exist."""
        index_file = os.path.join(directory, "index.faiss")
        meta_file = os.path.join(directory, "chunks.json")

        if os.path.exists(index_file) and os.path.exists(meta_file):
            self.index = faiss.read_index(index_file)
            with open(meta_file, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            return True
        return False
