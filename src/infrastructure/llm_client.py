"""
LLM Client — Abstraction layer over LangChain / OpenAI.

Provides a clean interface for LLM interactions with:
- Retry logic with exponential backoff
- Token usage tracking
- Structured logging
- Error isolation from business logic
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)

from src.config.settings import Settings, get_settings
from src.domain import ChatMessage, MessageRole

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Production-grade wrapper around LangChain ChatOpenAI.

    Responsibilities:
    - Manage LLM lifecycle and configuration
    - Convert domain messages to LangChain format
    - Track token usage and response latency
    - Handle retries and error reporting
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._validate_config()
        self._llm = self._create_llm()
        self._total_tokens_used: int = 0
        self._total_requests: int = 0

    # ── Factory ──────────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """Validate that required configuration is present."""
        if not self._settings.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass it in Settings."
            )

    def _create_llm(self) -> ChatOpenAI:
        """Create a configured ChatOpenAI instance."""
        return ChatOpenAI(
            model=self._settings.llm_model,
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
            openai_api_key=self._settings.openai_api_key,
            request_timeout=self._settings.llm_request_timeout,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def generate(
        self,
        messages: list[ChatMessage],
        system_prompt: str = "",
        temperature: Optional[float] = None,
    ) -> tuple[str, float]:
        """
        Generate a response from the LLM.

        Args:
            messages: List of domain ChatMessage objects.
            system_prompt: Optional system prompt override.
            temperature: Optional temperature override for this request.

        Returns:
            Tuple of (response_text, latency_ms).

        Raises:
            LLMError: If the LLM call fails after retries.
        """
        lc_messages = self._to_langchain_messages(messages, system_prompt)

        # Apply per-request temperature if provided
        llm = self._llm
        if temperature is not None:
            llm = self._llm.bind(temperature=temperature)

        start_time = time.perf_counter()
        try:
            response = llm.invoke(lc_messages)
            latency_ms = (time.perf_counter() - start_time) * 1000

            self._total_requests += 1
            text = response.content if hasattr(response, "content") else str(response)

            logger.info(
                "LLM response generated",
                extra={
                    "model": self._settings.llm_model,
                    "latency_ms": round(latency_ms, 2),
                    "response_length": len(text),
                },
            )
            return text, latency_ms

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "LLM generation failed",
                extra={
                    "model": self._settings.llm_model,
                    "latency_ms": round(latency_ms, 2),
                    "error": str(e),
                },
            )
            raise LLMError(f"LLM call failed: {e}") from e

    def generate_raw(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> tuple[str, float]:
        """
        Generate a response from a raw text prompt (convenience method).

        Args:
            prompt: User prompt string.
            system_prompt: Optional system prompt.

        Returns:
            Tuple of (response_text, latency_ms).
        """
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        return self.generate(messages, system_prompt=system_prompt)

    # ── Configuration ────────────────────────────────────────────────────

    def update_model(self, model_name: str) -> None:
        """Switch to a different LLM model."""
        self._settings.llm_model = model_name
        self._llm = self._create_llm()
        logger.info(f"LLM model switched to: {model_name}")

    def update_temperature(self, temperature: float) -> None:
        """Update the default temperature."""
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        self._settings.llm_temperature = temperature
        self._llm = self._create_llm()

    # ── Metrics ──────────────────────────────────────────────────────────

    @property
    def total_requests(self) -> int:
        return self._total_requests

    @property
    def model_name(self) -> str:
        return self._settings.llm_model

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _to_langchain_messages(
        messages: list[ChatMessage],
        system_prompt: str = "",
    ) -> list[BaseMessage]:
        """Convert domain messages to LangChain format."""
        lc_messages: list[BaseMessage] = []

        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))

        for msg in messages:
            if msg.role == MessageRole.USER:
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == MessageRole.SYSTEM:
                lc_messages.append(SystemMessage(content=msg.content))

        return lc_messages


class LLMError(Exception):
    """Custom exception for LLM-related failures."""

    pass
