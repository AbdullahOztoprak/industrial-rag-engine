"""
API Routes for the Industrial AI Knowledge Assistant.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from src.config.settings import Settings, get_settings
from src.domain import (
    ChatRequest,
    ChatResponseDTO,
    HealthCheckResponse,
    IndustrialDomain,
)
from src.interface.api.dependencies import get_chat_service
from src.interface.api.app import get_uptime
from src.application.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Industrial AI Assistant"])


# ─── Health Check ─────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    settings: Settings = Depends(get_settings),
    chat_service: ChatService = Depends(get_chat_service),
) -> HealthCheckResponse:
    """
    Health check endpoint for monitoring and load balancers.

    Returns service status, version, and RAG initialization state.
    """
    return HealthCheckResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment.value,
        rag_loaded=chat_service.rag_status,
        uptime_seconds=round(get_uptime(), 2),
    )


# ─── Chat ─────────────────────────────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponseDTO)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponseDTO:
    """
    Process a chat message through the Industrial AI pipeline.

    The message goes through:
    1. Domain classification
    2. RAG context retrieval
    3. LLM generation with industrial system prompt
    4. Confidence scoring and safety analysis

    Returns a structured IndustrialResponse with sources, confidence, and safety warnings.
    """
    try:
        response = chat_service.process_message(request)
        return response
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}",
        )


# ─── RAG Management ──────────────────────────────────────────────────────────


@router.post("/rag/initialize", status_code=status.HTTP_200_OK)
async def initialize_rag(
    chat_service: ChatService = Depends(get_chat_service),
) -> dict[str, str]:
    """
    Initialize or reinitialize the RAG pipeline.

    Loads industrial documentation and builds the vector index.
    """
    success = chat_service.initialize_rag()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG initialization failed. Check logs for details.",
        )
    return {"status": "RAG pipeline initialized successfully"}


# ─── Industrial Topics ───────────────────────────────────────────────────────


@router.get("/topics")
async def get_industrial_topics() -> dict[str, list[dict[str, str]]]:
    """Get the list of supported industrial automation domains."""
    return {"domains": [{"name": domain.value, "key": domain.name} for domain in IndustrialDomain]}


# ─── Conversation Management ─────────────────────────────────────────────────


@router.delete("/conversations/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    chat_service: ChatService = Depends(get_chat_service),
) -> dict[str, str]:
    """Clear a conversation's history."""
    if chat_service.clear_conversation(conversation_id):
        return {"status": "Conversation cleared"}
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Conversation not found",
    )
