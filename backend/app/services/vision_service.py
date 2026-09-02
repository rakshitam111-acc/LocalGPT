"""Multi-Modal Vision Service for Image Processing, OCR & Visual Question Answering."""

import base64
import io
import os
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image


class VisionService:
    """Processes user uploaded images and prepares multi-modal payloads for local & cloud LLMs."""

    @staticmethod
    def process_image(image_bytes: bytes, max_dim: int = 1024) -> Tuple[str, str]:
        """Resize image if needed and convert to JPEG base64 string."""
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert RGBA / P to RGB
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Resize if dimensions exceed max_dim
        width, height = img.size
        if max(width, height) > max_dim:
            scale = max_dim / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        out_buffer = io.BytesIO()
        img.save(out_buffer, format="JPEG", quality=85)
        b64_str = base64.b64encode(out_buffer.getvalue()).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{b64_str}"
        return b64_str, data_uri

    @classmethod
    def format_vision_payload(
        cls,
        prompt: str,
        image_bytes: bytes,
        provider: str = "ollama",
    ) -> Dict[str, Any]:
        """Format message structure according to LLM provider specifications."""
        b64_str, data_uri = cls.process_image(image_bytes)

        if provider == "ollama":
            return {
                "role": "user",
                "content": prompt or "Please analyze and describe this image in detail.",
                "images": [b64_str],
            }
        else:
            # OpenAI / Groq Vision format
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "Please analyze and describe this image in detail."},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
