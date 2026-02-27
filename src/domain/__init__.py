"""
Domain models for Industrial AI Knowledge Assistant.

These are the core data structures that represent business concepts,
independent of any specific framework or infrastructure.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ─── Enumerations ─────────────────────────────────────────────────────────────
# ─── Core Domain Models ─────────────────────────────────────────────────────────
# ─── Chat Models ──────────────────────────────────────────────────────────────
# ─── RAG Models ───────────────────────────────────────────────────────────────
# ─── Industrial Analysis Models ───────────────────────────────────────────────


class RiskLevel(str, Enum):
    """Risk classification for industrial issues."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConfidenceLevel(str, Enum):
    """Confidence classification for AI responses."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class IndustrialDomain(str, Enum):
    """Supported industrial automation domains."""

    PLC_PROGRAMMING = "PLC Programming"
    SCADA_SYSTEMS = "SCADA Systems"
    BUILDING_AUTOMATION = "Building Automation"
    PREDICTIVE_MAINTENANCE = "Predictive Maintenance"
    INDUSTRIAL_IOT = "Industrial IoT"
    MANUFACTURING_EXECUTION = "Manufacturing Execution Systems"
    ALARM_MANAGEMENT = "Alarm Management"
    ENERGY_MANAGEMENT = "Energy Management"
    GENERAL = "General"


class MessageRole(str, Enum):
    """Chat message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ─── Chat Models ──────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    """A complete conversation with metadata."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    messages: list[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    domain: IndustrialDomain = IndustrialDomain.GENERAL

    def add_message(self, role: MessageRole, content: str) -> ChatMessage:
        """Add a message to the conversation."""
        message = ChatMessage(role=role, content=content)
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message


# ─── RAG Models ───────────────────────────────────────────────────────────────


class DocumentChunk(BaseModel):
    """A chunk of an industrial document with metadata."""

    content: str
    source: str
    page: Optional[int] = None
    chunk_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Result from a RAG retrieval query."""

    chunks: list[DocumentChunk] = Field(default_factory=list)
    query: str
    relevance_scores: list[float] = Field(default_factory=list)


# ─── Industrial Analysis Models ───────────────────────────────────────────────


class SourceAttribution(BaseModel):
    """Attribution to a source document."""

    document: str
    section: Optional[str] = None
    relevance_score: float = 0.0
    excerpt: str = ""


class SafetyWarning(BaseModel):
    """Safety warning associated with an industrial response."""

    level: RiskLevel
    message: str
    standard_reference: Optional[str] = None  # e.g., "IEC 61131-3"


class IndustrialResponse(BaseModel):
    """
    Structured response for industrial queries.

    This is the core output format that enforces structured,
    auditable responses for industrial automation questions.
    """

    answer: str = Field(description="Main answer to the query")

    # Structured analysis
    problem_summary: Optional[str] = Field(default=None, description="Concise problem statement")
    root_cause: Optional[str] = Field(default=None, description="Identified root cause")
    solution: Optional[str] = Field(default=None, description="Recommended solution")

    # Quality indicators
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    confidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Numeric confidence [0-1]"
    )
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    domain: IndustrialDomain = Field(default=IndustrialDomain.GENERAL)

    # Sources and safety
    sources: list[SourceAttribution] = Field(default_factory=list)
    safety_warnings: list[SafetyWarning] = Field(default_factory=list)

    # Metadata
    model_used: str = ""
    response_time_ms: float = 0.0
    hallucination_flags: list[str] = Field(
        default_factory=list, description="Potential hallucination indicators detected"
    )

    @property
    def has_safety_concerns(self) -> bool:
        """Check if the response has safety warnings."""
        return len(self.safety_warnings) > 0

    @property
    def is_high_confidence(self) -> bool:
        """Check if the response has high confidence."""
        return self.confidence_score >= 0.8


# ─── API Request/Response Models ─────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Incoming chat request."""

    message: str = Field(min_length=1, max_length=5000)
    conversation_id: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    use_rag: bool = True
    domain_hint: Optional[IndustrialDomain] = None


class ChatResponseDTO(BaseModel):
    """API response wrapper for chat responses."""

    conversation_id: str
    response: IndustrialResponse
    processing_time_ms: float


class HealthCheckResponse(BaseModel):
    """Health check response model."""

    status: str = "healthy"
    version: str
    environment: str
    rag_loaded: bool = False
    uptime_seconds: float = 0.0
