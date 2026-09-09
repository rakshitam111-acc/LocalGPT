"""API routes for AI Image & Video Generation."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.api.auth import get_current_user_optional
from app.db.models import User
from app.services.media_service import MediaService

router = APIRouter(prefix="/media", tags=["AI Media Studio"])


class ImageGenerateRequest(BaseModel):
    prompt: str
    style: Optional[str] = "photorealistic"
    aspect_ratio: Optional[str] = "1:1"
    model: Optional[str] = "flux"
    seed: Optional[int] = None


class VideoGenerateRequest(BaseModel):
    prompt: str
    image_url: Optional[str] = None
    style: Optional[str] = "cinematic"
    aspect_ratio: Optional[str] = "16:9"
    duration: Optional[int] = 5
    seed: Optional[int] = None


@router.post("/generate-image")
def generate_image(
    data: ImageGenerateRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """Generate high-resolution photorealistic AI Image."""
    try:
        res = MediaService.generate_image(
            prompt=data.prompt,
            style=data.style or "photorealistic",
            aspect_ratio=data.aspect_ratio or "1:1",
            model=data.model or "flux",
            seed=data.seed,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate-video")
def generate_video(
    data: VideoGenerateRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """Generate animated AI Video clip."""
    try:
        res = MediaService.generate_video(
            prompt=data.prompt,
            image_url=data.image_url,
            style=data.style or "cinematic",
            aspect_ratio=data.aspect_ratio or "16:9",
            duration=data.duration or 5,
            seed=data.seed,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
