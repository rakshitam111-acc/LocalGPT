"""Conversations CRUD, message actions, and real-time SSE streaming chat with Web Search, Vision & RAG."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.api.auth import get_current_user
from app.core.database import get_db
from app.db.models import Conversation, Message, User
from app.services.llm_service import stream_hosted_llm
from app.services.memory_service import MemoryService
from app.services.rag_service import RAGService
from app.services.web_search_service import WebSearchService
from app.services.vision_service import VisionService

router = APIRouter(tags=["Chat"])


# Pydantic Schemas
class CreateConversationRequest(BaseModel):
    title: Optional[str] = "New Chat"
    system_prompt: Optional[str] = "You are an intelligent, helpful, and concise AI assistant."
    provider: Optional[str] = "groq"
    model: Optional[str] = "llama-3.3-70b-versatile"
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


class UpdateConversationRequest(BaseModel):
    title: Optional[str] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class ChatStreamRequest(BaseModel):
    conversation_id: str
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = 1024
    use_rag: Optional[bool] = True
    use_web_search: Optional[bool] = False
    image_data: Optional[str] = None  # Base64 string for Vision models


class MessageFeedbackRequest(BaseModel):
    feedback: Optional[str] = None  # 'like', 'dislike', or None


@router.get("/conversations")
def list_conversations(
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all conversations for the authenticated user."""
    query = db.query(Conversation).filter(Conversation.user_id == current_user.id)
    if search and search.strip():
        query = query.filter(Conversation.title.ilike(f"%{search.strip()}%"))

    convs = query.order_by(Conversation.updated_at.desc()).all()
    return [
        {
            "id": c.id,
            "title": c.title,
            "system_prompt": c.system_prompt,
            "provider": c.provider,
            "model": c.model,
            "temperature": float(c.temperature or 0.7),
            "top_p": float(c.top_p or 0.9),
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            "message_count": len(c.messages),
        }
        for c in convs
    ]


@router.post("/conversations")
def create_conversation(
    data: CreateConversationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new conversation session."""
    conv = Conversation(
        user_id=current_user.id,
        title=data.title or "New Chat",
        system_prompt=data.system_prompt,
        provider=data.provider or "ollama",
        model=data.model or "llama3.2:latest",
        temperature=str(data.temperature or 0.7),
        top_p=str(data.top_p or 0.9),
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return {
        "id": conv.id,
        "title": conv.title,
        "system_prompt": conv.system_prompt,
        "provider": conv.provider,
        "model": conv.model,
        "temperature": float(conv.temperature),
        "top_p": float(conv.top_p),
        "created_at": conv.created_at.isoformat(),
        "messages": [],
    }


@router.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get conversation details with full message history."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "id": conv.id,
        "title": conv.title,
        "system_prompt": conv.system_prompt,
        "provider": conv.provider,
        "model": conv.model,
        "temperature": float(conv.temperature or 0.7),
        "top_p": float(conv.top_p or 0.9),
        "created_at": conv.created_at.isoformat(),
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "sources": m.get_sources(),
                "created_at": m.created_at.isoformat(),
                "feedback": m.feedback,
            }
            for m in conv.messages
        ],
    }


@router.patch("/conversations/{conversation_id}")
def update_conversation(
    conversation_id: str,
    data: UpdateConversationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update conversation title or model parameters."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if data.title is not None:
        conv.title = data.title
    if data.system_prompt is not None:
        conv.system_prompt = data.system_prompt
    if data.model is not None:
        conv.model = data.model
    if data.provider is not None:
        conv.provider = data.provider
    if data.temperature is not None:
        conv.temperature = str(data.temperature)
    if data.top_p is not None:
        conv.top_p = str(data.top_p)

    db.commit()
    db.refresh(conv)
    return {"message": "Conversation updated successfully", "id": conv.id}


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a conversation and all its messages."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(conv)
    db.commit()
    return {"message": "Conversation deleted successfully"}


@router.post("/chat/stream")
async def chat_stream(
    req: ChatStreamRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Real-time SSE streaming AI generation with multi-turn memory, RAG, Web Search & Vision."""
    conv = db.query(Conversation).filter(Conversation.id == req.conversation_id, Conversation.user_id == current_user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # 1. Save user turn message
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=req.message,
    )
    db.add(user_msg)

    # Automatically set conversation title on first message
    if len(conv.messages) == 0 or conv.title == "New Chat":
        conv.title = req.message[:28] + ("..." if len(req.message) > 28 else "")

    conv.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(conv)

    # 2. Retrieve RAG document context if enabled
    rag_context = ""
    sources = []
    if req.use_rag:
        rag_service = RAGService(user_id=current_user.id)
        rag_context, sources = rag_service.retrieve(req.message, top_k=4)

    # 3. Perform Live Web Search if enabled (Perplexity-Style)
    web_sources = []
    if req.use_web_search:
        search_res = WebSearchService.format_search_context(req.message, max_results=4)
        if search_res.get("context"):
            rag_context = (rag_context + "\n\n" + search_res["context"]).strip()
            web_sources = search_res.get("sources", [])

    # 4. Format Multi-Turn Prompt Messages
    prompt_messages = MemoryService.build_chat_messages(
        system_prompt=conv.system_prompt or "You are an intelligent, helpful AI assistant.",
        history=conv.messages[:-1],
        current_message=req.message,
        rag_context=rag_context,
    )

    # If Vision image provided, extract OCR visual context & attach safely
    provider = req.provider or conv.provider or "ollama"
    model = req.model or conv.model or "llama3.2:latest"

    if req.image_data:
        raw_b64 = req.image_data.split(",")[-1]
        
        # Extract visual OCR text from image using RapidOCR
        image_ocr_text = ""
        try:
            import base64
            from PIL import Image
            import io
            import numpy as np
            from app.services.rag_service import _ocr_engine
            
            img_bytes = base64.b64decode(raw_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if _ocr_engine:
                ocr_res, _ = _ocr_engine(np.array(img))
                if ocr_res:
                    image_ocr_text = "\n".join([line[1] for line in ocr_res]).strip()
        except Exception as ocr_err:
            print(f"[Image OCR Notice]: {ocr_err}")

        # If image has text or visual labels, inject into prompt
        if image_ocr_text:
            prompt_messages[-1]["content"] += f"\n\n[Visual Content / Text Extracted from Attached Image]:\n{image_ocr_text}"
        else:
            prompt_messages[-1]["content"] += "\n\n[User attached an image with visual graphics/elements for analysis]."

        # Only pass raw images if model explicitly supports vision
        if "vision" in model.lower() or "llava" in model.lower() or "moondream" in model.lower():
            if provider == "ollama":
                prompt_messages[-1]["images"] = [raw_b64]
            else:
                prompt_messages[-1]["content"] = [
                    {"type": "text", "text": req.message},
                    {"type": "image_url", "image_url": {"url": req.image_data}},
                ]

    temperature = req.temperature if req.temperature is not None else float(conv.temperature or 0.7)
    top_p = req.top_p if req.top_p is not None else float(conv.top_p or 0.9)
    user_settings = current_user.get_settings()

    async def sse_event_generator():
        # First event: emit document RAG sources if found
        if sources:
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # Second event: emit live web sources if found
        if web_sources:
            yield f"data: {json.dumps({'type': 'web_sources', 'sources': web_sources})}\n\n"

        full_response_chunks = []
        async for chunk in stream_hosted_llm(
            messages=prompt_messages,
            provider=provider,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=req.max_tokens or 1024,
            user_settings=user_settings,
        ):
            full_response_chunks.append(chunk)
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

        # Save assistant message to database
        complete_content = "".join(full_response_chunks)
        from app.core.database import SessionLocal
        save_db = SessionLocal()
        try:
            assistant_msg = Message(
                conversation_id=conv.id,
                role="assistant",
                content=complete_content,
            )
            all_sources = sources + [{"title": w["title"], "url": w["url"], "domain": w["domain"]} for w in web_sources]
            if all_sources:
                assistant_msg.set_sources(all_sources)
            save_db.add(assistant_msg)
            save_db.commit()
            msg_id = assistant_msg.id
        finally:
            save_db.close()

        # Emit completion metadata
        yield f"data: {json.dumps({'type': 'done', 'message_id': msg_id, 'created_at': datetime.utcnow().isoformat()})}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
