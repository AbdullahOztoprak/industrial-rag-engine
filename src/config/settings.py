"""
Centralized configuration management using Pydantic Settings.

Supports environment variables, .env files, and sensible defaults.
All configuration is validated at startup to fail fast on misconfiguration.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data" / "industrial_docs"


class Settings(BaseSettings):
    """Application settings with validation and environment variable support."""

    # --- Application ---
    app_name: str = "Industrial AI Knowledge Assistant"
    app_version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    # --- LLM Configuration ---
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    llm_model: str = Field(default="gpt-3.5-turbo", description="Default LLM model")
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="LLM temperature")
    llm_max_tokens: int = Field(default=1500, ge=100, le=8000, description="Max tokens per response")
    llm_request_timeout: int = Field(default=30, description="LLM request timeout in seconds")

    # --- RAG Configuration ---
    embedding_model: str = "text-embedding-ada-002"
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    retrieval_top_k: int = Field(default=4, ge=1, le=20)
    docs_directory: str = str(DATA_DIR)
    vector_store_path: Optional[str] = None

    # --- API Configuration ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    cors_origins: list[str] = ["http://localhost:8501", "http://localhost:3000"]
    rate_limit_per_minute: int = 60

    # --- Logging ---
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    log_file: Optional[str] = None

    # --- Security ---
    api_key_header: str = "X-API-Key"
    allowed_api_keys: list[str] = []

    # --- Streamlit UI ---
    ui_title: str = "Industrial AI Knowledge Assistant"
    ui_port: int = 8501

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Ensure chunk_overlap is less than chunk_size."""
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})")
        return v

    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
        """Basic validation of API key format."""
        if v is not None and not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Returns:
        Validated Settings instance. Cached after first call.
    """
    return Settings()
