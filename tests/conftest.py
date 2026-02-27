"""
Test fixtures and shared configuration for all tests.
"""

import os
import sys
from pathlib import Path
import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment variables BEFORE importing settings
os.environ["OPENAI_API_KEY"] = "sk-test-key-for-unit-tests-only"
os.environ["ENVIRONMENT"] = "development"
os.environ["DEBUG"] = "true"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def test_settings():
    """Create test settings with mock API key."""
    from src.config.settings import Settings

    return Settings(
        openai_api_key="sk-test-key-for-unit-tests-only",
        environment="development",
        debug=True,
        llm_model="gpt-3.5-turbo",
        llm_temperature=0.3,
        llm_max_tokens=500,
    )


@pytest.fixture
def sample_messages():
    """Create sample chat messages for testing."""
    from src.domain import ChatMessage, MessageRole

    return [
        ChatMessage(role=MessageRole.USER, content="How do I program a PLC?"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="PLC programming uses IEC 61131-3 languages...",
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="What about ladder logic specifically?",
        ),
    ]


@pytest.fixture
def sample_chat_request():
    """Create a sample chat request."""
    from src.domain import ChatRequest

    return ChatRequest(
        message="How do I implement PID control in a Siemens PLC?",
        use_rag=True,
    )


@pytest.fixture
def sample_industrial_query():
    """Industrial query for testing domain classification."""
    return "How do I troubleshoot a BACnet communication failure in the building automation system?"


@pytest.fixture
def sample_scada_query():
    """SCADA-specific query for testing."""
    return "What are the best practices for SCADA alarm management per ISA-18.2?"


@pytest.fixture
def sample_safety_query():
    """Safety-critical query for testing safety warnings."""
    return "How do I configure an emergency stop circuit for a Safety PLC?"


@pytest.fixture
def mock_llm_response():
    """Mock LLM response text."""
    return (
        "To implement PID control in a Siemens PLC using TIA Portal, "
        "you would use the PID_Compact function block. The key parameters are:\n"
        "1. Proportional gain (Kp)\n"
        "2. Integral time (Ti)\n"
        "3. Derivative time (Td)\n\n"
        "Configure the setpoint and process variable, then tune using "
        "the built-in auto-tuning feature in TIA Portal."
    )


@pytest.fixture
def sample_document_text():
    """Sample industrial document text for RAG testing."""
    return """
# PLC Programming Best Practices

Programmable Logic Controllers (PLCs) are industrial computers adapted
for the control of manufacturing processes. Key programming languages
include Ladder Logic (LD), Structured Text (ST), and Function Block
Diagram (FBD) as defined in IEC 61131-3.

## PID Control Implementation

PID control requires proper tuning of Kp, Ki, and Kd parameters.
Anti-windup protection should be implemented for the integral term.
"""


@pytest.fixture
def test_docs_dir(tmp_path, sample_document_text):
    """Create a temporary directory with test documents."""
    docs_dir = tmp_path / "industrial_docs"
    docs_dir.mkdir()

    doc_file = docs_dir / "test_plc_guide.txt"
    doc_file.write_text(sample_document_text)

    return str(docs_dir)
