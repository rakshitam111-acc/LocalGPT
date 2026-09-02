"""Retrieval-Augmented Generation (RAG) pipeline for LocalGPT."""

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple
from document_loader import load_document, chunk_text
from embeddings import embed_texts, embed_query
from vector_store import FAISSVectorStore

DOCS_STORAGE_DIR = os.path.join(os.path.dirname(__file__), "data", "documents")


class RAGPipeline:
    """End-to-end RAG manager handling document ingestion, indexing, and retrieval."""

    def __init__(self, storage_dir: str = DOCS_STORAGE_DIR):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.vector_store = FAISSVectorStore(dimension=384)
        # Attempt to load existing index if saved
        self.vector_store.load(self.storage_dir)

    def ingest_file(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 60) -> int:
        """Load, chunk, embed, and index a single document file."""
        pages_data = load_document(file_path)
        if not pages_data:
            return 0

        chunks = chunk_text(pages_data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        self.vector_store.add_chunks(chunks, embeddings)
        self.vector_store.save(self.storage_dir)
        return len(chunks)

    def ingest_uploaded_file(self, uploaded_file) -> Tuple[str, int]:
        """Save a Streamlit UploadedFile to data/documents and index it."""
        file_path = os.path.join(self.storage_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)

        num_chunks = self.ingest_file(file_path)
        return uploaded_file.name, num_chunks

    def retrieve_context(self, query: str, top_k: int = 4) -> Tuple[str, List[Dict[str, Any]]]:
        """Search relevant document chunks and format them into context string with citations."""
        if self.vector_store.get_total_chunks() == 0:
            return "", []

        query_vec = embed_query(query)
        retrieved_chunks = self.vector_store.search(query_vec, top_k=top_k)

        if not retrieved_chunks:
            return "", []

        context_parts = []
        sources = []

        for idx, chunk in enumerate(retrieved_chunks):
            source_label = f"{chunk['source']} (Page {chunk['page']})"
            context_parts.append(
                f"[Source {idx+1}: {source_label}]\n{chunk['text']}"
            )
            sources.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "similarity": f"{chunk.get('similarity_score', 0.0):.2f}",
                "snippet": chunk["text"][:200] + ("..." if len(chunk["text"]) > 200 else ""),
            })

        formatted_context = "\n\n".join(context_parts)
        return formatted_context, sources

    def get_indexed_documents(self) -> List[str]:
        """Get names of all currently indexed documents."""
        return self.vector_store.get_indexed_documents()

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics for RAG index."""
        return {
            "total_documents": len(self.get_indexed_documents()),
            "total_chunks": self.vector_store.get_total_chunks(),
            "documents": self.get_indexed_documents(),
        }

    def clear_index(self):
        """Clear all indexed documents and reset store."""
        self.vector_store.clear()
        if os.path.exists(self.storage_dir):
            for fname in os.listdir(self.storage_dir):
                fpath = os.path.join(self.storage_dir, fname)
                try:
                    if os.path.isfile(fpath):
                        os.unlink(fpath)
                except Exception:
                    pass
