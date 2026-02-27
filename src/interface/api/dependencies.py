"""
Dependency Injection for FastAPI routes.

Manages singleton services with proper lifecycle.
"""

from __future__ import annotations

from functools import lru_cache

from src.application.chat_service import ChatService
from src.config.settings import get_settings


@lru_cache()
def get_chat_service() -> ChatService:
    """
    Get a singleton ChatService instance.

    Uses lru_cache to ensure only one instance exists.
    In production, consider using a proper DI container.
    """
    settings = get_settings()
    service = ChatService(settings=settings)

    # Auto-initialize RAG if API key is available
    if settings.openai_api_key:
        service.initialize_rag()

    return service
