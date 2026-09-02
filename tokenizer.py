"""Tokenizer utilities for LLM X-Ray.

Handles tokenization, token ID extraction, visual token breakdowns,
and vocabulary inspection for Qwen/Qwen2.5-1.5B-Instruct.
"""

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from typing import Any, Dict, List
from transformers import AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def load_tokenizer(model_id: str = DEFAULT_MODEL_ID):
    """Load and return the Hugging Face AutoTokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return tokenizer


def get_token_breakdown(text: str, tokenizer) -> Dict[str, Any]:
    """Tokenize the input text and extract detailed token-level information."""
    if not text:
        return {
            "tokens": [],
            "token_ids": [],
            "tokens_formatted": "",
            "ids_formatted": "",
            "details": [],
            "vocab_size": getattr(tokenizer, "vocab_size", len(tokenizer)) if tokenizer else 0,
            "seq_len": 0,
        }

    input_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in input_ids]

    details: List[Dict[str, Any]] = []
    for idx, (tok, tid) in enumerate(zip(tokens, input_ids)):
        details.append({
            "index": idx,
            "token": tok,
            "display_token": repr(tok)[1:-1],  # Escapes newlines/whitespace cleanly
            "token_id": tid,
            "length": len(tok),
        })

    tokens_formatted = " | ".join([f"'{d['display_token']}'" for d in details])
    ids_formatted = " | ".join([str(d["token_id"]) for d in details])

    return {
        "tokens": tokens,
        "token_ids": input_ids,
        "tokens_formatted": tokens_formatted,
        "ids_formatted": ids_formatted,
        "details": details,
        "vocab_size": getattr(tokenizer, "vocab_size", len(tokenizer)),
        "seq_len": len(input_ids),
    }


def format_chat_prompt(prompt: str, tokenizer) -> str:
    """Format user prompt into Qwen chat template."""
    messages = [
        {"role": "system", "content": "You are a helpful and concise assistant."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
