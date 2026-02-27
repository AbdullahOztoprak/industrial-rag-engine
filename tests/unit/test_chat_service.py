"""
Unit tests for ChatService orchestration.

Tests the full pipeline with mocked infrastructure dependencies.
"""

import pytest
from unittest.mock import MagicMock

from src.application.chat_service import ChatService
from src.domain import (
    ChatRequest,
    IndustrialDomain,
)


@pytest.fixture
def mock_llm_client():
    """Create a mocked LLM client."""
    client = MagicMock()
    client.generate.return_value = (
        "To configure PID control in a PLC, use the PID_Compact function block. "
        "Set Kp=2.0, Ti=5s for temperature control applications.",
        250.0,  # latency_ms
    )
    client.model_name = "gpt-3.5-turbo"
    return client


@pytest.fixture
def mock_rag_service():
    """Create a mocked RAG service."""
    from src.domain import DocumentChunk, RetrievalResult, SourceAttribution

    service = MagicMock()
    service.is_initialized = True
    service.retrieve.return_value = RetrievalResult(
        query="PID control",
        chunks=[
            DocumentChunk(
                content="PID control requires Kp, Ki, Kd tuning...",
                source="plc_guide.txt",
                metadata={"document_name": "plc_guide"},
            )
        ],
        relevance_scores=[0.85],
    )
    service.build_augmented_prompt.return_value = (
        "Context: PID control requires Kp, Ki, Kd tuning...\n" "Question: How to configure PID?"
    )
    service.get_source_attributions.return_value = [
        SourceAttribution(
            document="plc_guide",
            relevance_score=0.85,
            excerpt="PID control requires Kp, Ki, Kd tuning...",
        )
    ]
    return service


@pytest.fixture
def chat_service(mock_llm_client, mock_rag_service, test_settings):
    """Create a ChatService with mocked dependencies."""
    return ChatService(
        llm_client=mock_llm_client,
        rag_service=mock_rag_service,
        settings=test_settings,
    )


class TestChatServiceProcessing:
    """Tests for the main message processing pipeline."""

    def test_process_message_returns_response(self, chat_service):
        request = ChatRequest(message="How to configure PID control?")
        result = chat_service.process_message(request)

        assert result.conversation_id is not None
        assert result.response.answer is not None
        assert len(result.response.answer) > 0
        assert result.processing_time_ms > 0

    def test_process_message_classifies_domain(self, chat_service):
        request = ChatRequest(message="How to program ladder logic in a PLC?")
        result = chat_service.process_message(request)

        assert result.response.domain == IndustrialDomain.PLC_PROGRAMMING

    def test_process_message_with_domain_hint(self, chat_service):
        request = ChatRequest(
            message="How to configure communication?",
            domain_hint=IndustrialDomain.SCADA_SYSTEMS,
        )
        result = chat_service.process_message(request)

        assert result.response.domain == IndustrialDomain.SCADA_SYSTEMS

    def test_process_message_uses_rag(self, chat_service, mock_rag_service):
        request = ChatRequest(message="PID control guide", use_rag=True)
        chat_service.process_message(request)

        mock_rag_service.retrieve.assert_called_once()
        mock_rag_service.build_augmented_prompt.assert_called_once()

    def test_process_message_without_rag(self, chat_service, mock_rag_service):
        request = ChatRequest(message="What is automation?", use_rag=False)
        chat_service.process_message(request)

        mock_rag_service.retrieve.assert_not_called()

    def test_process_message_includes_sources(self, chat_service):
        request = ChatRequest(message="PID control")
        result = chat_service.process_message(request)

        assert len(result.response.sources) > 0
        assert result.response.sources[0].document == "plc_guide"

    def test_process_message_includes_confidence(self, chat_service):
        request = ChatRequest(message="PID control")
        result = chat_service.process_message(request)

        assert 0.0 <= result.response.confidence_score <= 1.0
        assert result.response.confidence is not None

    def test_process_message_model_info(self, chat_service):
        request = ChatRequest(message="test")
        result = chat_service.process_message(request)

        assert result.response.model_used == "gpt-3.5-turbo"


class TestChatServiceConversation:
    """Tests for conversation management."""

    def test_creates_new_conversation(self, chat_service):
        request = ChatRequest(message="Hello")
        result = chat_service.process_message(request)

        conv = chat_service.get_conversation(result.conversation_id)
        assert conv is not None
        assert len(conv.messages) == 2  # user + assistant

    def test_continues_existing_conversation(self, chat_service):
        # First message
        req1 = ChatRequest(message="What is a PLC?")
        result1 = chat_service.process_message(req1)
        conv_id = result1.conversation_id

        # Second message in same conversation
        req2 = ChatRequest(
            message="Tell me more about ladder logic",
            conversation_id=conv_id,
        )
        result2 = chat_service.process_message(req2)

        assert result2.conversation_id == conv_id
        conv = chat_service.get_conversation(conv_id)
        assert len(conv.messages) == 4  # 2 user + 2 assistant

    def test_clear_conversation(self, chat_service):
        request = ChatRequest(message="test")
        result = chat_service.process_message(request)

        assert chat_service.clear_conversation(result.conversation_id) is True
        assert chat_service.get_conversation(result.conversation_id) is None

    def test_clear_nonexistent_conversation(self, chat_service):
        assert chat_service.clear_conversation("nonexistent") is False


class TestChatServiceErrorHandling:
    """Tests for error handling in the pipeline."""

    def test_llm_failure_returns_error_message(self, mock_rag_service, test_settings):
        from src.infrastructure.llm_client import LLMError

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = LLMError("API timeout")
        mock_llm.model_name = "gpt-3.5-turbo"

        service = ChatService(
            llm_client=mock_llm,
            rag_service=mock_rag_service,
            settings=test_settings,
        )

        request = ChatRequest(message="test query")
        result = service.process_message(request)

        # Should still return a response (error message)
        assert result.response.answer is not None
        assert "unable to generate" in result.response.answer.lower()

    def test_rag_not_initialized_still_works(self, mock_llm_client, test_settings):
        """Test that the system works even without RAG."""
        mock_rag = MagicMock()
        mock_rag.is_initialized = False

        service = ChatService(
            llm_client=mock_llm_client,
            rag_service=mock_rag,
            settings=test_settings,
        )

        request = ChatRequest(message="test")
        result = service.process_message(request)

        assert result.response.answer is not None
        assert len(result.response.sources) == 0


class TestChatServiceMetrics:
    """Tests for service metrics."""

    def test_active_conversations_count(self, chat_service):
        assert chat_service.active_conversations == 0

        chat_service.process_message(ChatRequest(message="test1"))
        assert chat_service.active_conversations == 1

        chat_service.process_message(ChatRequest(message="test2"))
        assert chat_service.active_conversations == 2
