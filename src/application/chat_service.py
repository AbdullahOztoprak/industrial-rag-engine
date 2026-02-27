"""
Chat Service — Main orchestrator for industrial AI conversations.

Coordinates between:
- LLM Client (generation)
- RAG Service (retrieval)
- Industrial Analyzer (domain analysis, confidence, safety)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from src.config.settings import Settings, get_settings
from src.domain import (
    ChatMessage,
    ChatRequest,
    ChatResponseDTO,
    Conversation,
    IndustrialResponse,
    MessageRole,
)
from src.application.industrial_analyzer import IndustrialAnalyzer
from src.application.rag_service import RAGService
from src.infrastructure.llm_client import LLMClient, LLMError

logger = logging.getLogger(__name__)


class ChatService:
    """
    Central orchestrator for the Industrial AI Knowledge Assistant.

    This service implements the main conversation flow:
    1. Classify the domain of the incoming query
    2. Retrieve relevant context via RAG
    3. Build an augmented prompt with safety-aware system instructions
    4. Generate an LLM response
    5. Analyze the response for confidence, risk, and hallucination
    6. Return a structured IndustrialResponse
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        rag_service: Optional[RAGService] = None,
        analyzer: Optional[IndustrialAnalyzer] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm_client or LLMClient(self._settings)
        self._rag = rag_service or RAGService(settings=self._settings)
        self._analyzer = analyzer or IndustrialAnalyzer()
        self._conversations: dict[str, Conversation] = {}

    # ── Public API ───────────────────────────────────────────────────────

    def process_message(self, request: ChatRequest) -> ChatResponseDTO:
        """
        Process an incoming chat message through the full pipeline.

        Args:
            request: ChatRequest with user message and options.

        Returns:
            ChatResponseDTO containing the structured response.
        """
        start_time = time.perf_counter()

        # 1. Get or create conversation
        conversation = self._get_or_create_conversation(request.conversation_id)
        conversation.add_message(MessageRole.USER, request.message)

        # 2. Classify domain
        domain = request.domain_hint or self._analyzer.classify_domain(request.message)
        conversation.domain = domain

        # 3. Build system prompt
        system_prompt = self._analyzer.build_system_prompt(domain)

        # 4. Retrieve context (if RAG is enabled and initialized)
        sources = []
        augmented_query = request.message

        if request.use_rag and self._rag.is_initialized:
            retrieval = self._rag.retrieve(request.message)
            augmented_query = self._rag.build_augmented_prompt(request.message, retrieval)
            sources = self._rag.get_source_attributions(retrieval)

        # 5. Prepare messages for LLM
        recent_messages = self._get_recent_messages(conversation, limit=10)
        # Replace the last user message with the augmented version
        if recent_messages:
            recent_messages[-1] = ChatMessage(
                role=MessageRole.USER,
                content=augmented_query,
            )

        # 6. Generate LLM response
        try:
            temperature = request.temperature or self._settings.llm_temperature
            response_text = self._llm.generate(
                messages=recent_messages,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        except LLMError as e:
            logger.error(f"LLM generation failed: {e}")
            response_text = (
                "I apologize, but I'm unable to generate a response at this time. "
                "Please check the system configuration and try again."
            )
            # llm_latency = 0.0

        # 7. Analyze response quality
        confidence_level, confidence_score = self._analyzer.compute_confidence(
            response_text, sources, request.message
        )
        risk_level = self._analyzer.assess_risk(request.message, response_text)
        safety_warnings = self._analyzer.generate_safety_warnings(request.message, response_text)
        hallucination_flags = self._analyzer.detect_hallucination_flags(response_text)

        # 8. Store assistant response
        conversation.add_message(MessageRole.ASSISTANT, response_text)

        # 9. Build structured response
        total_time = (time.perf_counter() - start_time) * 1000

        industrial_response = IndustrialResponse(
            answer=response_text,
            confidence=confidence_level,
            confidence_score=confidence_score,
            risk_level=risk_level,
            domain=domain,
            sources=sources,
            safety_warnings=safety_warnings,
            hallucination_flags=hallucination_flags,
            model_used=self._llm.model_name,
            response_time_ms=round(total_time, 2),
        )

        logger.info(
            "Message processed",
            extra={
                "conversation_id": conversation.id,
                "domain": domain.value,
                "confidence": confidence_score,
                "risk": risk_level.value,
                "latency_ms": round(total_time, 2),
                "source_count": len(sources),
            },
        )

        return ChatResponseDTO(
            conversation_id=conversation.id,
            response=industrial_response,
            processing_time_ms=round(total_time, 2),
        )

    def initialize_rag(self) -> bool:
        """Initialize the RAG pipeline."""
        return self._rag.initialize()

    # ── Conversation Management ──────────────────────────────────────────

    def _get_or_create_conversation(self, conversation_id: Optional[str]) -> Conversation:
        """Get an existing conversation or create a new one."""
        if conversation_id and conversation_id in self._conversations:
            return self._conversations[conversation_id]

        conversation = Conversation()
        self._conversations[conversation.id] = conversation
        return conversation

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Retrieve a conversation by ID."""
        return self._conversations.get(conversation_id)

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation's history."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            return True
        return False

    @staticmethod
    def _get_recent_messages(conversation: Conversation, limit: int = 10) -> list[ChatMessage]:
        """Get the most recent messages from a conversation."""
        return conversation.messages[-limit:]

    # ── Metrics ──────────────────────────────────────────────────────────

    @property
    def active_conversations(self) -> int:
        return len(self._conversations)

    @property
    def rag_status(self) -> bool:
        return self._rag.is_initialized
