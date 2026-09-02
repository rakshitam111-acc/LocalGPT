"""API routes for LLM X-Ray Interpretability and Code Sandbox."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.api.auth import get_current_user
from app.db.models import User
from app.services.xray_service import XRayService
from app.services.code_sandbox_service import CodeSandboxService

router = APIRouter(prefix="/xray", tags=["X-Ray"])


class XRayInspectRequest(BaseModel):
    prompt: str
    response: str
    context: Optional[str] = ""


class CodeExecuteRequest(BaseModel):
    code: str
    language: Optional[str] = "python"
    timeout: Optional[int] = 5


@router.post("/inspect")
def inspect_generation(
    data: XRayInspectRequest,
    current_user: User = Depends(get_current_user),
):
    """Compute neural interpretability telemetry: token logits, attention, and faithfulness."""
    tokens = XRayService.inspect_tokens(data.response)
    attention = XRayService.compute_attention_heatmap(data.prompt, data.response)
    faithfulness = XRayService.calculate_faithfulness(data.response, data.context or "")

    return {
        "tokens": tokens[:50],  # Return first 50 tokens for display
        "attention": attention,
        "faithfulness": faithfulness,
    }


@router.post("/execute-code")
def execute_code(
    data: CodeExecuteRequest,
    current_user: User = Depends(get_current_user),
):
    """Execute Python, C, C++, or JavaScript code safely in the sandbox."""
    res = CodeSandboxService.execute_code(
        code_str=data.code,
        language=data.language or "python",
        timeout_seconds=data.timeout or 5,
    )
    return res
