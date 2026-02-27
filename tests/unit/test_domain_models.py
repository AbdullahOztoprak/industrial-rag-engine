"""
Unit tests for domain models.

Tests Pydantic model validation, serialization, and business logic.
"""

import pytest
from datetime import datetime

from src.domain import (
    ChatMessage,
    ChatRequest,
    Conversation,
    ConfidenceLevel,
    DocumentChunk,
    IndustrialDomain,
    IndustrialResponse,
    MessageRole,
    RiskLevel,
    SafetyWarning,
    SourceAttribution,
    RetrievalResult,
)


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_create_user_message(self):
        msg = ChatMessage(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.id is not None
        assert isinstance(msg.timestamp, datetime)

    def test_create_assistant_message(self):
        msg = ChatMessage(role=MessageRole.ASSISTANT, content="Hi there")
        assert msg.role == MessageRole.ASSISTANT

    def test_message_metadata(self):
        msg = ChatMessage(
            role=MessageRole.USER,
            content="Test",
            metadata={"source": "api"},
        )
        assert msg.metadata["source"] == "api"


class TestConversation:
    """Tests for Conversation model."""

    def test_create_empty_conversation(self):
        conv = Conversation()
        assert len(conv.messages) == 0
        assert conv.domain == IndustrialDomain.GENERAL

    def test_add_message(self):
        conv = Conversation()
        msg = conv.add_message(MessageRole.USER, "Test question")
        assert len(conv.messages) == 1
        assert msg.content == "Test question"
        assert msg.role == MessageRole.USER

    def test_add_multiple_messages(self):
        conv = Conversation()
        conv.add_message(MessageRole.USER, "Q1")
        conv.add_message(MessageRole.ASSISTANT, "A1")
        conv.add_message(MessageRole.USER, "Q2")
        assert len(conv.messages) == 3

    def test_conversation_updates_timestamp(self):
        conv = Conversation()
        initial_time = conv.updated_at
        conv.add_message(MessageRole.USER, "Update")
        assert conv.updated_at >= initial_time


class TestIndustrialResponse:
    """Tests for IndustrialResponse model."""

    def test_basic_response(self):
        resp = IndustrialResponse(answer="Test answer")
        assert resp.answer == "Test answer"
        assert resp.confidence == ConfidenceLevel.MEDIUM
        assert resp.risk_level == RiskLevel.LOW

    def test_high_confidence_check(self):
        resp = IndustrialResponse(
            answer="test",
            confidence_score=0.9,
        )
        assert resp.is_high_confidence is True

    def test_low_confidence_check(self):
        resp = IndustrialResponse(
            answer="test",
            confidence_score=0.3,
        )
        assert resp.is_high_confidence is False

    def test_safety_concerns(self):
        resp = IndustrialResponse(
            answer="test",
            safety_warnings=[
                SafetyWarning(level=RiskLevel.HIGH, message="Danger"),
            ],
        )
        assert resp.has_safety_concerns is True

    def test_no_safety_concerns(self):
        resp = IndustrialResponse(answer="test")
        assert resp.has_safety_concerns is False

    def test_full_industrial_response(self):
        resp = IndustrialResponse(
            answer="PLC needs reset",
            problem_summary="PLC communication failure",
            root_cause="Network cable disconnected",
            solution="Reconnect cable and restart communication",
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.92,
            risk_level=RiskLevel.MEDIUM,
            domain=IndustrialDomain.PLC_PROGRAMMING,
            sources=[
                SourceAttribution(
                    document="plc_guide.txt",
                    relevance_score=0.85,
                    excerpt="Communication troubleshooting...",
                )
            ],
            safety_warnings=[
                SafetyWarning(
                    level=RiskLevel.MEDIUM,
                    message="Ensure PLC is in STOP before hardware changes",
                )
            ],
            model_used="gpt-4",
            response_time_ms=1234.5,
        )
        assert resp.is_high_confidence
        assert resp.has_safety_concerns
        assert len(resp.sources) == 1
        assert resp.domain == IndustrialDomain.PLC_PROGRAMMING


class TestChatRequest:
    """Tests for ChatRequest validation."""

    def test_valid_request(self):
        req = ChatRequest(message="Test query")
        assert req.message == "Test query"
        assert req.use_rag is True

    def test_empty_message_rejected(self):
        with pytest.raises(Exception):
            ChatRequest(message="")

    def test_request_with_domain_hint(self):
        req = ChatRequest(
            message="Test",
            domain_hint=IndustrialDomain.SCADA_SYSTEMS,
        )
        assert req.domain_hint == IndustrialDomain.SCADA_SYSTEMS

    def test_temperature_validation(self):
        req = ChatRequest(message="Test", temperature=1.5)
        assert req.temperature == 1.5

    def test_invalid_temperature_rejected(self):
        with pytest.raises(Exception):
            ChatRequest(message="Test", temperature=3.0)


class TestDocumentChunk:
    """Tests for DocumentChunk model."""

    def test_create_chunk(self):
        chunk = DocumentChunk(
            content="PLC programming basics",
            source="plc_guide.txt",
            chunk_index=0,
        )
        assert chunk.content == "PLC programming basics"
        assert chunk.source == "plc_guide.txt"

    def test_chunk_with_metadata(self):
        chunk = DocumentChunk(
            content="Test",
            source="test.txt",
            metadata={"page": 1, "section": "intro"},
        )
        assert chunk.metadata["page"] == 1


class TestRetrievalResult:
    """Tests for RetrievalResult model."""

    def test_empty_result(self):
        result = RetrievalResult(query="test")
        assert len(result.chunks) == 0
        assert len(result.relevance_scores) == 0

    def test_result_with_chunks(self):
        result = RetrievalResult(
            query="PLC",
            chunks=[
                DocumentChunk(content="c1", source="s1"),
                DocumentChunk(content="c2", source="s2"),
            ],
            relevance_scores=[0.9, 0.7],
        )
        assert len(result.chunks) == 2
        assert result.relevance_scores[0] == 0.9
