"""Conversation memory and multi-turn context builder for LocalGPT."""

from typing import Any, Dict, List, Optional


class ConversationMemory:
    """Manages short-term and multi-turn conversation memory with token budgeting."""

    def __init__(self, max_history_turns: int = 10):
        self.max_history_turns = max_history_turns

    def build_prompt_messages(
        self,
        system_prompt: str,
        messages_history: List[Dict[str, Any]],
        current_user_message: str,
        rag_context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Construct a structured message list for Qwen chat template."""
        formatted_messages = [
            {"role": "system", "content": system_prompt.strip()}
        ]

        # Truncate history to max_history_turns
        recent_history = messages_history[- (self.max_history_turns * 2):] if messages_history else []

        for msg in recent_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ["user", "assistant"] and content:
                formatted_messages.append({"role": role, "content": content})

        # Inject RAG context into the latest user message if available
        if rag_context and rag_context.strip():
            augmented_content = (
                "You have access to the following reference documents to answer the question:\n\n"
                f"--- BEGIN DOCUMENT CONTEXT ---\n{rag_context}\n--- END DOCUMENT CONTEXT ---\n\n"
                "Instructions: Answer accurately using the context above. If the context contains the answer, "
                "cite the source file and page. If the context is insufficient, state what is known and answer politely.\n\n"
                f"Question: {current_user_message}"
            )
            formatted_messages.append({"role": "user", "content": augmented_content})
        else:
            formatted_messages.append({"role": "user", "content": current_user_message})

        return formatted_messages
