"""
Helper functions for Industrial AI Knowledge Assistant.

General-purpose utility functions used across the application.
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def validate_api_key(api_key: str) -> bool:
    """
    Validate if an API key is in the correct format.

    Args:
        api_key: The API key to validate.

    Returns:
        True if the API key appears valid.
    """
    if not api_key or not isinstance(api_key, str):
        return False
    return api_key.startswith("sk-") and len(api_key) > 20


def sanitize_input(text: str, max_length: int = 5000) -> str:
    """
    Sanitize user input to prevent injection and excessive length.

    Args:
        text: Raw input string.
        max_length: Maximum allowed length.

    Returns:
        Cleaned input string.
    """
    if not text:
        return ""
    # Strip whitespace and limit length
    cleaned = text.strip()[:max_length]
    # Remove potential prompt injection patterns
    cleaned = re.sub(
        r"(?i)(ignore previous|disregard|forget all)", "[FILTERED]", cleaned
    )
    return cleaned


def save_conversation(
    conversation_id: str,
    messages: list[dict[str, str]],
    save_dir: Optional[str] = None,
) -> bool:
    """
    Save a conversation to a JSON file.

    Args:
        conversation_id: Unique identifier for the conversation.
        messages: List of conversation messages.
        save_dir: Directory to save the conversation.

    Returns:
        True if saved successfully.
    """
    try:
        save_dir = save_dir or os.path.join(os.getcwd(), "conversations")
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{conversation_id}_{timestamp}.json"
        filepath = os.path.join(save_dir, filename)

        data = {
            "conversation_id": conversation_id,
            "timestamp": timestamp,
            "message_count": len(messages),
            "messages": messages,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return True
    except Exception:
        return False


def load_conversation(filepath: str) -> Optional[dict[str, Any]]:
    """
    Load a conversation from a JSON file.

    Args:
        filepath: Path to the conversation file.

    Returns:
        Conversation data or None if error.
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to a maximum length with a suffix."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_response_time(ms: float) -> str:
    """Format response time for display."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.2f}s"
