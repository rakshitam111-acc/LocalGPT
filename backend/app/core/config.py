"""Application configuration settings for LocalGPT Phase 3."""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "LocalGPT AI Platform"
    VERSION: str = "3.0.0"
    API_V1_STR: str = "/api"

    # Security & Auth
    SECRET_KEY: str = os.getenv("SECRET_KEY", "localgpt-super-secret-key-change-in-production-2026")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # Google OAuth
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")

    # Database: PostgreSQL with SQLite fallback
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'platform.db'))}",
    )

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ]

    # LLM Defaults: Ollama Local by default!
    DEFAULT_PROVIDER: str = os.getenv("DEFAULT_PROVIDER", "ollama")  # ollama, groq, openai, openrouter
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "llama3.2:latest")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

    # Storage Paths
    DATA_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    UPLOAD_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploads"))
    VECTOR_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "vectors"))

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()

os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_DIR, exist_ok=True)
