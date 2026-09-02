"""High-level chat coordinator for LocalGPT."""

from typing import Any, Dict, Generator, List, Optional, Tuple
from database import add_message, get_messages, delete_messages_from
from memory import ConversationMemory
from model import generate_stream, extract_prompt_internals
from rag import RAGPipeline


class ChatCoordinator:
    """Orchestrates multi-turn conversations, RAG context retrieval, and streaming generation."""

    def __init__(self, model, tokenizer, rag_pipeline: Optional[RAGPipeline] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.memory = ConversationMemory(max_history_turns=10)
        self.rag = rag_pipeline if rag_pipeline is not None else RAGPipeline()

    def process_turn_stream(
        self,
        conv_id: str,
        user_message: str,
        system_prompt: str,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        use_rag: bool = True,
    ) -> Tuple[Generator[str, None, None], List[Dict[str, Any]], str]:
        """Prepare context, retrieve RAG documents if enabled, and return streaming generator and sources."""
        # 1. Retrieve RAG context if enabled
        rag_context = ""
        sources = []
        if use_rag and self.rag.vector_store.get_total_chunks() > 0:
            rag_context, sources = self.rag.retrieve_context(user_message, top_k=4)

        # 2. Get past messages
        history = get_messages(conv_id)

        # 3. Build multi-turn messages
        formatted_messages = self.memory.build_prompt_messages(
            system_prompt=system_prompt,
            messages_history=history,
            current_user_message=user_message,
            rag_context=rag_context,
        )

        # 4. Apply Qwen chat template
        prompt_text = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 5. Create streaming generator
        token_stream = generate_stream(
            prompt=prompt_text,
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        return token_stream, sources, prompt_text

    def save_completed_turn(
        self,
        conv_id: str,
        user_message: str,
        assistant_response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        xray_data: Optional[Dict[str, Any]] = None,
    ):
        """Persist user turn and assistant response into database."""
        add_message(conv_id, role="user", content=user_message)
        add_message(conv_id, role="assistant", content=assistant_response, sources=sources, xray_data=xray_data)

    def regenerate_turn(
        self,
        conv_id: str,
        msg_id: str,
    ) -> Optional[str]:
        """Delete from the given assistant message onwards to allow regenerating."""
        delete_messages_from(conv_id, msg_id)
        messages = get_messages(conv_id)
        if messages and messages[-1]["role"] == "user":
            return messages[-1]["content"]
        return None
