"""In-Memory Hugging Face PyTorch Model Service for 100% self-hosted zero-API execution."""

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import threading
from typing import AsyncGenerator, Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

_model = None
_tokenizer = None
_lock = threading.Lock()


def get_model_and_tokenizer():
    """Lazy load PyTorch model and tokenizer into memory."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        with _lock:
            if _model is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.bfloat16 if device == "cuda" else torch.float32

                print(f"[HF Service] Loading {MODEL_ID} into memory on {device} ({dtype})...")
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
                _model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                if device != "cuda":
                    _model = _model.to(device)
                _model.eval()
                print(f"[HF Service] {MODEL_ID} loaded successfully!")
    return _model, _tokenizer


def format_messages_to_prompt(messages: List[Dict[str, str]], tokenizer) -> str:
    """Format chat messages into Qwen ChatML prompt."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted


async def stream_local_hf(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> AsyncGenerator[str, None]:
    """Generate and stream tokens directly from in-memory PyTorch model."""
    import asyncio
    
    model, tokenizer = get_model_and_tokenizer()
    prompt = format_messages_to_prompt(messages, tokenizer)
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    loop = asyncio.get_event_loop()
    
    def get_next_chunk():
        try:
            return next(streamer)
        except StopIteration:
            return None

    while True:
        chunk = await loop.run_in_executor(None, get_next_chunk)
        if chunk is None:
            break
        yield chunk

    thread.join()
