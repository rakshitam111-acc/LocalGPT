"""Document management endpoints: Upload (PDF, DOCX, TXT, CSV, JSON), Indexing, List, and Delete."""

import os
import shutil
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from app.api.auth import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.db.models import Document, User
from app.services.rag_service import RAGService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload and index multiple PDF, DOCX, TXT, CSV, or JSON documents into RAG vector store."""
    user_upload_dir = os.path.join(settings.UPLOAD_DIR, current_user.id)
    os.makedirs(user_upload_dir, exist_ok=True)

    rag_service = RAGService(user_id=current_user.id)
    uploaded_records = []

    for file in files:
        safe_filename = os.path.basename(file.filename)
        dest_path = os.path.join(user_upload_dir, safe_filename)

        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(dest_path)
        ext = os.path.splitext(safe_filename)[1].lower().replace(".", "")

        # Ingest into RAG vector index
        chunks_count = rag_service.add_document(dest_path)

        # Check if record already exists
        existing_doc = db.query(Document).filter(
            Document.user_id == current_user.id,
            Document.filename == safe_filename,
        ).first()

        if existing_doc:
            existing_doc.file_size = file_size
            existing_doc.total_chunks = chunks_count
            db.commit()
            doc_record = existing_doc
        else:
            doc_record = Document(
                user_id=current_user.id,
                filename=safe_filename,
                file_type=ext or "txt",
                file_size=file_size,
                total_chunks=chunks_count,
            )
            db.add(doc_record)
            db.commit()
            db.refresh(doc_record)

        uploaded_records.append({
            "id": doc_record.id,
            "filename": doc_record.filename,
            "file_type": doc_record.file_type,
            "file_size": doc_record.file_size,
            "total_chunks": doc_record.total_chunks,
            "created_at": doc_record.created_at.isoformat() if doc_record.created_at else None,
        })

    return {
        "message": f"Successfully processed {len(files)} document(s)",
        "documents": uploaded_records,
    }


@router.get("")
def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all indexed documents for the current user."""
    docs = db.query(Document).filter(Document.user_id == current_user.id).order_by(Document.created_at.desc()).all()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "file_type": d.file_type,
            "file_size": d.file_size,
            "total_chunks": d.total_chunks,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in docs
    ]


@router.delete("/{doc_id}")
def delete_document(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a document from database and remove its chunks from FAISS vector store."""
    doc = db.query(Document).filter(Document.id == doc_id, Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    rag_service = RAGService(user_id=current_user.id)
    rag_service.delete_document(doc.filename)

    # Remove physical file if exists
    user_file = os.path.join(settings.UPLOAD_DIR, current_user.id, doc.filename)
    if os.path.exists(user_file):
        try:
            os.remove(user_file)
        except Exception:
            pass

    db.delete(doc)
    db.commit()
    return {"message": "Document deleted successfully", "id": doc_id}


@router.delete("")
def clear_all_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Clear all documents and reset the user's vector store."""
    docs = db.query(Document).filter(Document.user_id == current_user.id).all()
    for d in docs:
        db.delete(d)
    db.commit()

    rag_service = RAGService(user_id=current_user.id)
    rag_service.clear()

    user_upload_dir = os.path.join(settings.UPLOAD_DIR, current_user.id)
    if os.path.exists(user_upload_dir):
        shutil.rmtree(user_upload_dir, ignore_errors=True)

    return {"message": "All documents cleared successfully"}
