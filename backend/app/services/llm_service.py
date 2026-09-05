"""LLM Gateway supporting In-Memory PyTorch Model (Hugging Face Spaces), Local Ollama, and Cloud Endpoints."""

import json
import os
from typing import AsyncGenerator, Dict, List, Optional
import httpx
from app.core.config import settings

PROVIDER_ENDPOINTS = {
    "local_hf": "in-memory-pytorch",
    "ollama": "http://localhost:11434/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


async def get_dynamic_models() -> List[Dict]:
    """Fetch locally installed Ollama models and merge with In-Memory PyTorch models."""
    models = [
        {
            "id": "llama3.2:latest",
            "name": "Llama 3.2 (Local Ollama — Lightning Fast)",
            "provider": "ollama",
            "category": "⚡ Ultra Fast (30+ tok/s)",
            "speed": "Fastest",
        },
        {
            "id": "llama3.1:8b",
            "name": "Llama 3.1 8B (Local Ollama)",
            "provider": "ollama",
            "category": "🧠 High Intelligence",
            "speed": "Smart",
        },
        {
            "id": "qwen-2.5-1.5b-local",
            "name": "Qwen 2.5 1.5B (In-Memory PyTorch)",
            "provider": "local_hf",
            "category": "🧠 Self-Hosted (0 APIs)",
            "speed": "Standard",
        },
        {
            "id": "llama-3.3-70b-versatile",
            "name": "Llama 3.3 70B (Groq Cloud — Instant)",
            "provider": "groq",
            "category": "⚡ 300+ tok/s Cloud",
            "speed": "Instant",
        },
    ]

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                ollama_models = data.get("models", [])
                existing_ids = {m["id"] for m in models}
                for om in ollama_models:
                    model_name = om.get("name", "")
                    if model_name and model_name not in existing_ids:
                        models.append({
                            "id": model_name,
                            "name": f"{model_name} (Local Ollama)",
                            "provider": "ollama",
                            "category": "⚡ Local Ollama",
                            "speed": "Fast",
                        })
    except Exception:
        pass
    return models


def resolve_api_key(provider: str, user_settings: Optional[Dict] = None) -> Optional[str]:
    """Resolve API key for external providers (Local models don't require one)."""
    if provider in ["local_hf", "ollama"]:
        return "local"

    if user_settings:
        keys_dict = user_settings.get("api_keys", {})
        if provider in keys_dict and keys_dict[provider]:
            return keys_dict[provider]

    if provider == "groq":
        return settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    elif provider == "openai":
        return settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    elif provider == "openrouter":
        return settings.OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY")
    return None


async def stream_hosted_llm(
    messages: List[Dict[str, str]],
    provider: str = "ollama",
    model: str = "llama3.2:latest",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
    user_settings: Optional[Dict] = None,
) -> AsyncGenerator[str, None]:
    """Stream completions from In-Memory PyTorch model, Local Ollama, or Cloud APIs."""
    
    # Auto-detect Ollama model names
    if any(m in model.lower() for m in ["llama3.2", "llama3.1", "mistral", "gemma", "phi", "deepseek", "nomic"]):
        provider = "ollama"

    api_key = resolve_api_key(provider, user_settings)
    if provider in ["groq", "openai", "openrouter"] and not api_key:
        # Fallback to local Ollama if no cloud API key is configured
        provider = "ollama"
        model = "llama3.2:latest"

    # 1. Native Fast Ollama Streaming with keep_alive and optimized CPU threads
    if provider == "ollama":
        base_raw = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        clean_base = base_raw.replace("/v1", "").rstrip("/")
        ollama_url = f"{clean_base}/api/chat"

        cpu_threads = min(8, max(2, os.cpu_count() or 4))
        target_model = model if model != "qwen-2.5-1.5b-local" else "llama3.2:latest"
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": True,
            "keep_alive": "120m",  # Keep model resident in RAM for instant 0-second loading!
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
                "num_thread": cpu_threads,
                "num_ctx": 2048,
            }
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", ollama_url, json=payload) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            if line and line.strip():
                                try:
                                    data = json.loads(line)
                                    chunk = data.get("message", {}).get("content", "")
                                    if chunk:
                                        yield chunk
                                    if data.get("done", False):
                                        break
                                except Exception:
                                    continue
                        return
                    else:
                        yield f"\n\n**Ollama HTTP Error {response.status_code}:** Could not connect to local Ollama model `{model}`."
                        return
        except Exception as e:
            yield f"\n\n*(Notice: Ollama connection: {e}. Switching to In-Memory PyTorch engine...)*\n\n"
            provider = "local_hf"

    # 2. In-Memory PyTorch / Hugging Face Spaces Model Execution
    if provider == "local_hf" or model == "qwen-2.5-1.5b-local":
        try:
            from app.services.local_model_service import stream_local_hf
            async for chunk in stream_local_hf(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            ):
                yield chunk
            return
        except Exception as e:
            yield f"\n\n**Local PyTorch Execution Error:** {str(e)}"
            return

    # 3. Cloud Provider Streaming (Groq / OpenAI / OpenRouter)
    base_url = PROVIDER_ENDPOINTS.get(provider, "https://api.groq.com/openai/v1")
    api_key = resolve_api_key(provider, user_settings)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": True,
    }

    url = f"{base_url}/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    yield f"\n\n**API Provider Error ({response.status_code}):** {err_body.decode(errors='replace')}"
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except Exception:
                            continue
    except Exception as e:
        yield f"\n\n**Streaming Error:** {str(e)}"
