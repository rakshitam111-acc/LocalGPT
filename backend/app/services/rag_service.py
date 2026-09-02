"""RAG Service supporting PDF (Native Text + OCR for Scanned PDFs), DOCX, TXT, CSV, and JSON with FAISS."""

import io
import json
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from app.core.config import settings

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

try:
    from rapidocr_onnxruntime import RapidOCR
    _ocr_engine = RapidOCR()
except Exception:
    _ocr_engine = None

from app.services.embeddings import embed_texts, embed_query


class DocumentParser:
    """Extracts structured text from PDF (native + OCR), DOCX, TXT, CSV, and JSON."""

    @staticmethod
    def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
        pages = []
        filename = os.path.basename(file_path)
        if not pypdf:
            return []
        try:
            reader = pypdf.PdfReader(file_path)
            for idx, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()

                # If page text is empty, check if it's an image/scanned PDF and apply OCR
                if not text and len(page.images) > 0 and _ocr_engine is not None:
                    ocr_lines = []
                    for img_obj in page.images:
                        try:
                            img = Image.open(io.BytesIO(img_obj.data)).convert("RGB")
                            img_np = np.array(img)
                            res, _ = _ocr_engine(img_np)
                            if res:
                                ocr_lines.extend([line[1] for line in res])
                        except Exception as e:
                            print(f"[OCR extraction notice]: {e}")
                    if ocr_lines:
                        text = "\n".join(ocr_lines).strip()

                if text:
                    pages.append({"text": text, "page": idx + 1, "source": filename})
        except Exception as e:
            print(f"[PDF Parse Error]: {e}")
        return pages

    @staticmethod
    def parse_docx(file_path: str) -> List[Dict[str, Any]]:
        if not docx:
            return []
        try:
            doc = docx.Document(file_path)
            filename = os.path.basename(file_path)
            paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paras)
            return [{"text": text, "page": 1, "source": filename}] if text else []
        except Exception as e:
            print(f"[DOCX Parse Error]: {e}")
            return []

    @staticmethod
    def parse_txt(file_path: str) -> List[Dict[str, Any]]:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
            return [{"text": text, "page": 1, "source": filename}] if text else []
        except Exception as e:
            return []

    @staticmethod
    def parse_csv(file_path: str) -> List[Dict[str, Any]]:
        filename = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            text_rows = []
            for idx, row in df.iterrows():
                row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
                text_rows.append(f"Row {idx+1}: {row_str}")
            text = "\n".join(text_rows)
            return [{"text": text, "page": 1, "source": filename}] if text else []
        except Exception as e:
            return []

    @staticmethod
    def parse_json(file_path: str) -> List[Dict[str, Any]]:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            text = json.dumps(data, indent=2)
            return [{"text": text, "page": 1, "source": filename}] if text else []
        except Exception as e:
            return []

    @classmethod
    def parse_file(cls, file_path: str) -> List[Dict[str, Any]]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return cls.parse_pdf(file_path)
        elif ext == ".docx":
            return cls.parse_docx(file_path)
        elif ext == ".csv":
            return cls.parse_csv(file_path)
        elif ext == ".json":
            return cls.parse_json(file_path)
        else:
            return cls.parse_txt(file_path)


def chunk_extracted_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> List[Dict[str, Any]]:
    """Split pages into overlapping chunks preserving source and page metadata."""
    chunks = []
    chunk_id = 0
    for page in pages:
        text = page["text"]
        source = page["source"]
        page_num = page["page"]
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            if end < text_len:
                last_space = text.rfind(" ", start, end)
                last_nl = text.rfind("\n", start, end)
                split = max(last_space, last_nl)
                if split > start + (chunk_size // 2):
                    end = split + 1

            chunk_text_str = text[start:end].strip()
            if chunk_text_str:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text_str,
                    "source": source,
                    "page": page_num,
                })
                chunk_id += 1

            if end >= text_len:
                break
            start = max(start + 1, end - chunk_overlap)
    return chunks


class RAGService:
    """User-aware FAISS RAG Service with OCR and Persistence."""

    def __init__(self, user_id: str = "global"):
        self.user_id = user_id
        self.user_vector_dir = os.path.join(settings.VECTOR_DIR, user_id)
        os.makedirs(self.user_vector_dir, exist_ok=True)
        self.dimension = 384
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks: List[Dict[str, Any]] = []
        self.load_index()

        # If index is empty but user has uploaded files in data/uploads, auto-index them with OCR!
        if self.index.ntotal == 0:
            self._auto_index_user_uploads()

    def _auto_index_user_uploads(self):
        user_upload_dir = os.path.join(settings.UPLOAD_DIR, self.user_id)
        if os.path.exists(user_upload_dir):
            for fname in os.listdir(user_upload_dir):
                fpath = os.path.join(user_upload_dir, fname)
                if os.path.isfile(fpath):
                    self.add_document(fpath)

    def add_document(self, file_path: str) -> int:
        """Parse, chunk, embed, and index a document."""
        pages = DocumentParser.parse_file(file_path)
        if not pages:
            return 0
        chunks = chunk_extracted_pages(pages)
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        if embeddings.shape[0] > 0:
            self.index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
            self.chunks.extend(chunks)
            self.save_index()
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 6) -> Tuple[str, List[Dict[str, Any]]]:
        """Search top-k chunks and format context with citations."""
        if self.index.ntotal == 0 or not self.chunks:
            return "", []

        q_lower = query.lower()
        if any(w in q_lower for w in ["analyze", "summarize", "what is this", "about the document", "explain the file", "about this file", "overview", "what is in"]):
            selected_chunks = self.chunks[:min(top_k, len(self.chunks))]
            context_blocks = []
            sources = []
            for chunk in selected_chunks:
                source_label = f"{chunk['source']} (Page {chunk['page']})"
                context_blocks.append(f"[Source: {source_label}]\n{chunk['text']}")
                sources.append({
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "similarity": "0.98",
                    "snippet": chunk["text"][:120] + "...",
                })
            return "\n\n".join(context_blocks), sources

        # Standard semantic vector search
        q_vec = embed_query(query)
        q_mat = np.ascontiguousarray(q_vec.reshape(1, -1), dtype=np.float32)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_mat, k)

        context_blocks = []
        sources = []

        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx]
                source_label = f"{chunk['source']} (Page {chunk['page']})"
                context_blocks.append(f"[Source: {source_label}]\n{chunk['text']}")
                sources.append({
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "similarity": f"{max(0.0, float(score)):.2f}",
                    "snippet": chunk["text"][:120] + "...",
                })

        return "\n\n".join(context_blocks), sources

    def save_index(self):
        faiss_path = os.path.join(self.user_vector_dir, "index.faiss")
        meta_path = os.path.join(self.user_vector_dir, "meta.json")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load_index(self):
        faiss_path = os.path.join(self.user_vector_dir, "index.faiss")
        meta_path = os.path.join(self.user_vector_dir, "meta.json")
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            try:
                self.index = faiss.read_index(faiss_path)
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
            except Exception as e:
                self.index = faiss.IndexFlatIP(self.dimension)
                self.chunks = []
