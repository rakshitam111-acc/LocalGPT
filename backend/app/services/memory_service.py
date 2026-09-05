"""Memory & context building service for multi-turn chat and RAG injection."""

from typing import Any, Dict, List, Optional
from app.db.models import Message


class MemoryService:
    @staticmethod
    def build_chat_messages(
        system_prompt: str,
        history: List[Message],
        current_message: str,
        rag_context: Optional[str] = None,
        max_history_messages: int = 16,
    ) -> List[Dict[str, str]]:
        """Construct standard OpenAI-compatible messages list with system prompt and RAG context."""
        messages: List[Dict[str, str]] = []

        # 1. System Prompt
        sys_content = (
            "You are an intelligent, helpful, and highly capable AI assistant. "
            "Always provide direct, relevant, accurate, and helpful answers to the user's questions. "
            "Format your answers with clean markdown and concise structure."
        )
        if system_prompt and system_prompt.strip() and system_prompt != "You are an intelligent, helpful AI assistant.":
            sys_content = system_prompt.strip()
        messages.append({"role": "system", "content": sys_content})

        # 2. Historical Messages (truncated to recent limit)
        recent_history = history[-max_history_messages:] if len(history) > max_history_messages else history
        for msg in recent_history:
            if msg.role in ["user", "assistant"] and msg.content:
                messages.append({"role": msg.role, "content": msg.content})

        # 3. Current User Query with RAG context
        if rag_context and rag_context.strip():
            augmented_content = (
                f"Retrieved Knowledge Excerpts:\n{rag_context}\n\n"
                f"User Question: {current_message}\n\n"
                "Instructions: If the retrieved knowledge is relevant, incorporate it and cite the source name/page. If the excerpts are not directly relevant to the question, answer the user's question directly and thoroughly using your knowledge."
            )
            messages.append({"role": "user", "content": augmented_content})
        else:
            messages.append({"role": "user", "content": current_message})

        return messages
