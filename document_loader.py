"""Document loader and text chunker for LocalGPT.

Supports PDF (pypdf), Word (.docx), and plain text (.txt) files.
"""

import os
from typing import Any, Dict, List
import pypdf
import docx


def load_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF preserving page numbers."""
    pages_data = []
    reader = pypdf.PdfReader(file_path)
    filename = os.path.basename(file_path)

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages_data.append({
                "text": text.strip(),
                "page": page_idx + 1,
                "source": filename,
            })
    return pages_data


def load_text_from_docx(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from Word .docx file."""
    doc = docx.Document(file_path)
    filename = os.path.basename(file_path)
    full_text = []

    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())

    text = "\n\n".join(full_text)
    if not text:
        return []
    return [{
        "text": text,
        "page": 1,
        "source": filename,
    }]


def load_text_from_txt(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from plain text file."""
    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()

    if not text:
        return []
    return [{
        "text": text,
        "page": 1,
        "source": filename,
    }]


def load_document(file_path: str) -> List[Dict[str, Any]]:
    """Load document based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_text_from_pdf(file_path)
    elif ext == ".docx":
        return load_text_from_docx(file_path)
    elif ext in [".txt", ".md", ".py", ".json", ".csv"]:
        return load_text_from_txt(file_path)
    else:
        # Fallback to plain text read
        return load_text_from_txt(file_path)


def chunk_text(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 60,
) -> List[Dict[str, Any]]:
    """Split extracted page text into smaller overlapping chunks with metadata."""
    chunks = []
    chunk_id = 0

    for page_item in pages_data:
        text = page_item["text"]
        page_num = page_item["page"]
        source_name = page_item["source"]

        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)

            # Try to break at newline or space if not at the end
            if end < text_len:
                last_space = text.rfind(" ", start, end)
                last_newline = text.rfind("\n", start, end)
                split_point = max(last_space, last_newline)
                if split_point > start + (chunk_size // 2):
                    end = split_point + 1

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_content,
                    "source": source_name,
                    "page": page_num,
                    "char_start": start,
                    "char_end": end,
                })
                chunk_id += 1

            if end >= text_len:
                break
            start = max(start + 1, end - chunk_overlap)

    return chunks
