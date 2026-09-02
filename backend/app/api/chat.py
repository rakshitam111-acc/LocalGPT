"""Conversations CRUD, message actions, and real-time SSE streaming chat."""

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
        provider=data.provider or "groq",
        model=data.model or "llama-3.3-70b-versatile",
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


@router.get("/conversations/{conv_id}")
def get_conversation_details(
    conv_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get single conversation with its full message history."""
    conv = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == current_user.id).first()
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
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "sources": m.get_sources(),
                "feedback": m.feedback,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in conv.messages
        ],
    }


@router.patch("/conversations/{conv_id}")
def update_conversation(
    conv_id: str,
    data: UpdateConversationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Rename or update settings for a conversation."""
    conv = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == current_user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if data.title is not None:
        conv.title = data.title.strip()
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

    conv.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(conv)
    return {"message": "Conversation updated successfully", "id": conv.id, "title": conv.title}


@router.delete("/conversations/{conv_id}")
def delete_conversation(
    conv_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a conversation and all its messages."""
    conv = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == current_user.id).first()
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
    """Real-time SSE streaming AI generation with multi-turn memory & RAG retrieval."""
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

    # 3. Format Multi-Turn Prompt Messages
    prompt_messages = MemoryService.build_chat_messages(
        system_prompt=conv.system_prompt or "You are a helpful AI assistant.",
        history=conv.messages[:-1],  # exclude current message since we pass it explicitly
        current_message=req.message,
        rag_context=rag_context,
    )

    provider = req.provider or conv.provider or "groq"
    model = req.model or conv.model or "llama-3.3-70b-versatile"
    temperature = req.temperature if req.temperature is not None else float(conv.temperature or 0.7)
    top_p = req.top_p if req.top_p is not None else float(conv.top_p or 0.9)
    user_settings = current_user.get_settings()

    async def sse_event_generator():
        # First event: emit sources if found
        if sources:
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

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
            if sources:
                assistant_msg.set_sources(sources)
            save_db.add(assistant_msg)
            save_db.commit()
            msg_id = assistant_msg.id
        finally:
            save_db.close()

        yield f"data: {json.dumps({'type': 'done', 'message_id': msg_id})}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/messages/{msg_id}/feedback")
def set_message_feedback(
    msg_id: str,
    data: MessageFeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Set thumbs up/down feedback on an assistant message."""
    msg = db.query(Message).join(Conversation).filter(
        Message.id == msg_id,
        Conversation.user_id == current_user.id,
    ).first()
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")

    msg.feedback = data.feedback
    db.commit()
    return {"message": "Feedback updated", "msg_id": msg_id, "feedback": msg.feedback}
