"""Model loading, inference, streaming, and internal state extraction for LocalGPT."""

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import threading
from typing import Any, Dict, Generator, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, TextIteratorStreamer
from tokenizer import DEFAULT_MODEL_ID, load_tokenizer


def get_device() -> str:
    """Detect and return the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_id: str = DEFAULT_MODEL_ID, device: str = None):
    """Load the causal language model with eager attention for introspection."""
    if device is None:
        device = get_device()

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    if device != "cuda":
        model = model.to(device)

    model.eval()
    return model, device


def get_model_metadata(model) -> Dict[str, Any]:
    """Extract structural architecture metadata from the model configuration."""
    config = model.config
    num_layers = getattr(config, "num_hidden_layers", 28)
    hidden_size = getattr(config, "hidden_size", 1536)
    num_attention_heads = getattr(config, "num_attention_heads", 12)
    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    vocab_size = getattr(config, "vocab_size", 151936)
    intermediate_size = getattr(config, "intermediate_size", 8960)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_type": getattr(config, "model_type", "qwen2"),
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "vocab_size": vocab_size,
        "intermediate_size": intermediate_size,
        "total_params": total_params,
        "total_params_formatted": f"{total_params / 1e9:.2f}B ({total_params:,})",
        "trainable_params": trainable_params,
    }


def generate_stream(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> Generator[str, None, None]:
    """Stream generated tokens in real-time using TextIteratorStreamer."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        if top_k > 0:
            gen_kwargs["top_k"] = top_k
        if top_p < 1.0:
            gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for text_chunk in streamer:
        yield text_chunk

    thread.join()


def extract_prompt_internals(
    text: str,
    model,
    tokenizer,
    top_k_preds: int = 10,
) -> Dict[str, Any]:
    """Run forward pass with full introspection: embeddings, hidden states, attentions, logits."""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    seq_len = input_ids.shape[1]
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    # 1. Extract Input Embeddings
    embedding_layer = model.get_input_embeddings()
    with torch.no_grad():
        input_embeddings_tensor = embedding_layer(input_ids)
        input_embeddings = input_embeddings_tensor[0].float().cpu().numpy()

    embedding_stats = []
    for idx, (tok, vec) in enumerate(zip(tokens, input_embeddings)):
        embedding_stats.append({
            "token_index": idx,
            "token": tok,
            "display_token": repr(tok)[1:-1],
            "token_id": int(input_ids[0, idx].item()),
            "mean": float(np.mean(vec)),
            "std": float(np.std(vec)),
            "min": float(np.min(vec)),
            "max": float(np.max(vec)),
            "l2_norm": float(np.linalg.norm(vec)),
        })

    # 2. Forward pass with hidden states and attention outputs
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    # 3. Extract Hidden States across all layers
    hidden_states_list = [h[0].float().cpu().numpy() for h in outputs.hidden_states]

    # 4. Extract Attention matrices across all layers
    attentions_list = []
    if outputs.attentions is not None and len(outputs.attentions) > 0:
        attentions_list = [att[0].float().cpu().numpy() for att in outputs.attentions]

    # 5. Extract Logits & Probabilities for the next token (position -1)
    last_token_logits = outputs.logits[0, -1, :].float()
    probs = F.softmax(last_token_logits, dim=-1)

    top_probs, top_indices = torch.topk(probs, k=top_k_preds)
    top_logits = last_token_logits[top_indices]

    predictions = []
    for rank in range(top_k_preds):
        tid = int(top_indices[rank].item())
        prob = float(top_probs[rank].item())
        logit = float(top_logits[rank].item())
        tok_str = tokenizer.decode([tid])
        predictions.append({
            "rank": rank + 1,
            "token_id": tid,
            "token": tok_str,
            "display_token": repr(tok_str)[1:-1],
            "probability": prob,
            "probability_percent": f"{prob * 100:.2f}%",
            "logit": logit,
        })

    return {
        "text": text,
        "input_ids": input_ids[0].tolist(),
        "tokens": tokens,
        "seq_len": seq_len,
        "hidden_size": input_embeddings.shape[1],
        "input_embeddings": input_embeddings,
        "embedding_stats": embedding_stats,
        "hidden_states": hidden_states_list,
        "num_hidden_layers": len(hidden_states_list) - 1,
        "attentions": attentions_list,
        "num_attention_layers": len(attentions_list),
        "num_heads": attentions_list[0].shape[0] if attentions_list else 0,
        "predictions": predictions,
    }


def generate_step_by_step(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 25,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    top_candidates_count: int = 5,
) -> Dict[str, Any]:
    """Auto-regressive token generation capturing internal stats at each generation step."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    generated_token_ids: List[int] = []
    generated_tokens: List[str] = []
    step_records: List[Dict[str, Any]] = []

    current_ids = input_ids.clone()

    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(input_ids=current_ids, return_dict=True)
            next_token_logits = outputs.logits[0, -1, :].float()

            if temperature > 0:
                scaled_logits = next_token_logits / temperature

                if top_k > 0:
                    indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k)[0][..., -1, None]
                    scaled_logits[indices_to_remove] = -float("Inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    scaled_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            next_token_id = int(next_token.item())
            next_token_prob = float(probs[next_token_id].item())
            next_token_str = tokenizer.decode([next_token_id])

            top_step_probs, top_step_indices = torch.topk(F.softmax(next_token_logits, dim=-1), k=top_candidates_count)
            top_candidates = []
            for rank in range(top_candidates_count):
                c_id = int(top_step_indices[rank].item())
                c_prob = float(top_step_probs[rank].item())
                c_str = tokenizer.decode([c_id])
                top_candidates.append({
                    "rank": rank + 1,
                    "token": c_str,
                    "display_token": repr(c_str)[1:-1],
                    "token_id": c_id,
                    "probability": c_prob,
                    "probability_percent": f"{c_prob * 100:.1f}%",
                })

            generated_token_ids.append(next_token_id)
            generated_tokens.append(next_token_str)
            accumulated_response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            accumulated_full = tokenizer.decode(current_ids[0].tolist() + [next_token_id], skip_special_tokens=True)

            step_records.append({
                "step": step + 1,
                "token_id": next_token_id,
                "token": next_token_str,
                "display_token": repr(next_token_str)[1:-1],
                "probability": next_token_prob,
                "probability_percent": f"{next_token_prob * 100:.2f}%",
                "accumulated_response": accumulated_response,
                "accumulated_full": accumulated_full,
                "top_candidates": top_candidates,
            })

            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

            if next_token_id == tokenizer.eos_token_id or (
                hasattr(tokenizer, "eot_token_id") and next_token_id == tokenizer.eot_token_id
            ):
                break

    full_generated_response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    return {
        "prompt": prompt,
        "generated_tokens": generated_tokens,
        "generated_token_ids": generated_token_ids,
        "full_response": full_generated_response,
        "step_records": step_records,
        "total_tokens_generated": len(generated_token_ids),
    }
