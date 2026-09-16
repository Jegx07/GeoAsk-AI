"""GeoAsk AI — Configuration and settings."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file from project root
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Project ---
    app_name: str = "GeoAsk AI"
    app_version: str = "0.1.0"
    debug: bool = True

    # --- LLM Provider ---
    llm_provider: str = "gemini"  # "gemini" | "openai" | "ollama"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    gemini_vision_model: str = "gemini-3.6-flash"
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- File Upload ---
    max_upload_size_mb: int = 100
    upload_dir: str = "./uploads"
    reports_dir: str = "./reports"

    # --- GPU ---
    use_gpu: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    # --- Derived Properties ---

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def reports_path(self) -> Path:
        p = Path(self.reports_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def gpu_available(self) -> bool:
        if not self.use_gpu:
            return False
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()
