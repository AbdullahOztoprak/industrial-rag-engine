"""
Unit tests for LLM Client.

Uses mocking to test LLM interactions without making real API calls.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.infrastructure.llm_client import LLMClient, LLMError
from src.domain import ChatMessage, MessageRole


class TestLLMClientInit:
    """Tests for LLMClient initialization."""

    @patch("src.infrastructure.llm_client.ChatOpenAI")
    def test_successful_init(self, mock_chat, test_settings):
        client = LLMClient(settings=test_settings)
        assert client.model_name == "gpt-3.5-turbo"
        assert client.total_requests == 0
        mock_chat.assert_called_once()

    def test_init_without_api_key_raises(self):
        from src.config.settings import Settings

        settings = Settings(openai_api_key=None)
        with pytest.raises(ValueError, match="API key is required"):
            LLMClient(settings=settings)


class TestLLMClientGenerate:
    """Tests for LLM response generation with mocked API."""

    @patch("src.infrastructure.llm_client.ChatOpenAI")
    def test_generate_returns_response(self, mock_chat_cls, test_settings):
        """Test that generate returns response text and latency."""
        # Setup mock
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "PID control uses Kp, Ki, Kd parameters."
        mock_instance.invoke.return_value = mock_response
        mock_chat_cls.return_value = mock_instance

        client = LLMClient(settings=test_settings)
        messages = [ChatMessage(role=MessageRole.USER, content="What is PID?")]

        text, latency = client.generate(messages, system_prompt="You are an expert.")

        assert text == "PID control uses Kp, Ki, Kd parameters."
        assert latency >= 0
        assert client.total_requests == 1
        mock_instance.invoke.assert_called_once()

    @patch("src.infrastructure.llm_client.ChatOpenAI")
    def test_generate_with_conversation_history(self, mock_chat_cls, test_settings):
        """Test generation with multi-turn conversation."""
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Follow-up answer"
        mock_instance.invoke.return_value = mock_response
        mock_chat_cls.return_value = mock_instance

        client = LLMClient(settings=test_settings)
        messages = [
            ChatMessage(role=MessageRole.USER, content="What is a PLC?"),
            ChatMessage(role=MessageRole.ASSISTANT, content="A PLC is..."),
            ChatMessage(role=MessageRole.USER, content="Tell me more"),
        ]

        text, _ = client.generate(messages)
        assert text == "Follow-up answer"

    @patch("src.infrastructure.llm_client.ChatOpenAI")
    def test_generate_failure_raises_llm_error(self, mock_chat_cls, test_settings):
        """Test that API failures are wrapped in LLMError."""
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = Exception("API timeout")
        mock_chat_cls.return_value = mock_instance

        client = LLMClient(settings=test_settings)
        messages = [ChatMessage(role=MessageRole.USER, content="test")]

        with pytest.raises(LLMError, match="LLM call failed"):
            client.generate(messages)

    @patch("src.infrastructure.llm_client.ChatOpenAI")
    def test_generate_raw(self, mock_chat_cls, test_settings):
        """Test convenience method for raw prompt generation."""
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Raw response"
        mock_instance.invoke.return_value = mock_response
        mock_chat_cls.return_value = mock_instance

        client = LLMClient(settings=test_settings)
        text, latency = client.generate_raw("Tell me about SCADA")

        assert text == "Raw response"
        assert latency >= 0


class TestLLMClientConfiguration:
    """Tests for LLM model configuration changes."""

    @patch("src.infrastructure.llm_client.ChatOpenAI")
    def test_update_model(self, mock_chat_cls, test_settings):
        client = LLMClient(settings=test_settings)
        client.update_model("gpt-4")
        assert client.model_name == "gpt-4"
        # Should have been called twice: init + update
        assert mock_chat_cls.call_count == 2

    @patch("src.infrastructure.llm_client.ChatOpenAI")
    def test_update_temperature(self, mock_chat_cls, test_settings):
        client = LLMClient(settings=test_settings)
        client.update_temperature(0.9)
        assert mock_chat_cls.call_count == 2

    @patch("src.infrastructure.llm_client.ChatOpenAI")
    def test_invalid_temperature_raises(self, mock_chat_cls, test_settings):
        client = LLMClient(settings=test_settings)
        with pytest.raises(ValueError, match="between 0.0 and 2.0"):
            client.update_temperature(3.0)


class TestMessageConversion:
    """Tests for domain → LangChain message conversion."""

    def test_convert_user_message(self):
        from langchain_core.messages import HumanMessage

        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        result = LLMClient._to_langchain_messages(messages)

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello"

    def test_convert_with_system_prompt(self):
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        result = LLMClient._to_langchain_messages(messages, system_prompt="Be helpful")

        assert len(result) == 2
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)

    def test_convert_mixed_messages(self):
        from langchain_core.messages import HumanMessage, AIMessage

        messages = [
            ChatMessage(role=MessageRole.USER, content="Q"),
            ChatMessage(role=MessageRole.ASSISTANT, content="A"),
        ]
        result = LLMClient._to_langchain_messages(messages)

        assert len(result) == 2
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)
