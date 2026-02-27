"""
Integration tests for API endpoints.

Tests FastAPI routes using TestClient with mocked service layer.
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from src.interface.api.app import create_app
from src.domain import (
    ChatResponseDTO,
    ConfidenceLevel,
    IndustrialDomain,
    IndustrialResponse,
    RiskLevel,
)


@pytest.fixture
def mock_chat_service():
    """Create a mocked ChatService for API testing."""
    service = MagicMock()
    service.rag_status = True

    # Mock process_message
    service.process_message.return_value = ChatResponseDTO(
        conversation_id="test-conv-123",
        response=IndustrialResponse(
            answer="PID control uses proportional, integral, derivative terms.",
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.85,
            risk_level=RiskLevel.LOW,
            domain=IndustrialDomain.PLC_PROGRAMMING,
            model_used="gpt-3.5-turbo",
            response_time_ms=250.0,
        ),
        processing_time_ms=300.0,
    )

    service.initialize_rag.return_value = True
    service.clear_conversation.return_value = True
    return service


@pytest.fixture
def client(mock_chat_service):
    """Create a FastAPI test client with mocked dependencies."""
    app = create_app()

    # Override dependency
    from src.interface.api.dependencies import get_chat_service

    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "environment" in data

    def test_health_shows_rag_status(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        assert "rag_loaded" in data


class TestChatEndpoint:
    """Tests for the chat endpoint."""

    def test_chat_basic_request(self, client, mock_chat_service):
        response = client.post(
            "/api/v1/chat",
            json={"message": "How to configure PID control?"},
        )
        assert response.status_code == 200

        data = response.json()
        assert "conversation_id" in data
        assert "response" in data
        assert data["response"]["answer"] is not None

    def test_chat_response_structure(self, client):
        response = client.post(
            "/api/v1/chat",
            json={"message": "What is a PLC?"},
        )
        data = response.json()

        resp = data["response"]
        assert "answer" in resp
        assert "confidence" in resp
        assert "confidence_score" in resp
        assert "risk_level" in resp
        assert "domain" in resp
        assert "sources" in resp
        assert "safety_warnings" in resp
        assert "model_used" in resp

    def test_chat_with_domain_hint(self, client, mock_chat_service):
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Configure communication",
                "domain_hint": "SCADA Systems",
            },
        )
        assert response.status_code == 200
        mock_chat_service.process_message.assert_called_once()

    def test_chat_with_rag_disabled(self, client, mock_chat_service):
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "What is automation?",
                "use_rag": False,
            },
        )
        assert response.status_code == 200

    def test_chat_empty_message_rejected(self, client):
        response = client.post(
            "/api/v1/chat",
            json={"message": ""},
        )
        assert response.status_code == 422  # Validation error

    def test_chat_with_conversation_id(self, client, mock_chat_service):
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Follow up question",
                "conversation_id": "existing-conv-123",
            },
        )
        assert response.status_code == 200


class TestRAGEndpoint:
    """Tests for the RAG management endpoint."""

    def test_initialize_rag(self, client, mock_chat_service):
        response = client.post("/api/v1/rag/initialize")
        assert response.status_code == 200
        mock_chat_service.initialize_rag.assert_called_once()

    def test_initialize_rag_failure(self, client, mock_chat_service):
        mock_chat_service.initialize_rag.return_value = False
        response = client.post("/api/v1/rag/initialize")
        assert response.status_code == 500


class TestTopicsEndpoint:
    """Tests for the topics endpoint."""

    def test_get_topics(self, client):
        response = client.get("/api/v1/topics")
        assert response.status_code == 200

        data = response.json()
        assert "domains" in data
        assert len(data["domains"]) > 0

    def test_topics_include_plc(self, client):
        response = client.get("/api/v1/topics")
        data = response.json()

        domain_names = [d["name"] for d in data["domains"]]
        assert "PLC Programming" in domain_names


class TestConversationEndpoint:
    """Tests for conversation management endpoint."""

    def test_clear_conversation(self, client, mock_chat_service):
        response = client.delete("/api/v1/conversations/test-conv-123")
        assert response.status_code == 200

    def test_clear_nonexistent_conversation(self, client, mock_chat_service):
        mock_chat_service.clear_conversation.return_value = False
        response = client.delete("/api/v1/conversations/nonexistent")
        assert response.status_code == 404
