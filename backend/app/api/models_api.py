"""Model listing and dynamic Ollama model detection."""

from fastapi import APIRouter
from app.services.llm_service import get_dynamic_models, PROVIDER_ENDPOINTS

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("")
async def list_available_models():
    """List available local Ollama and hosted models."""
    models = await get_dynamic_models()
    return {
        "models": models,
        "providers": list(PROVIDER_ENDPOINTS.keys()),
    }
