# Industrial AI Knowledge Assistant

![System Architecture](./mermaid-diagram%20(3).png)

![RAG Pipeline](./mermaid-diagram%20(4).png)

A production-grade, Retrieval-Augmented Generation (RAG) system for industrial automation knowledge work.

This project goes beyond a generic chatbot by combining retrieval, domain-aware analysis, and structured output contracts designed for safety-sensitive engineering contexts. Instead of returning plain free-form text, it provides confidence-aware and source-attributed responses tailored to PLC, SCADA, BAS, and adjacent industrial domains.

[![CI](https://github.com/your-username/Langchain_website_chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/Langchain_website_chatbot/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Key Features

- RAG architecture with ChromaDB retrieval and OpenAI generation
- Source attribution from retrieved industrial documents
- Confidence scoring with explicit confidence levels and numeric score
- Safety warning detection with industrial standard references
- Hallucination mitigation through post-generation signal checks
- Structured industrial analysis: Problem -> Root Cause -> Solution
- Clean architecture with clear layer boundaries and testable components
- Dual interface: FastAPI API + Streamlit UI

---

## Overview

The Industrial AI Knowledge Assistant is a domain-focused AI service that answers automation questions with traceability and risk awareness.

Why it matters:
- Industrial decisions can affect uptime, safety, and compliance
- Teams need evidence-backed answers, not just fluent text
- Knowledge changes quickly across manuals, standards, and internal docs

What makes it different from normal chatbots:
- It retrieves relevant documentation before generation
- It returns typed, structured responses (not only natural language)
- It reports confidence and risk indicators in every response path
- It includes source attribution for auditability

---

## Architecture

### Layered Design (Clean Architecture)

| Layer | Core Responsibility | Representative Modules |
|---|---|---|
| Interface Layer | Expose user and API entry points | Streamlit UI, FastAPI routes and middleware |
| Application Layer | Orchestrate use cases and workflow | ChatService, RAG service, IndustrialAnalyzer |
| Infrastructure Layer | Integrate external systems | OpenAI client, ChromaDB vector store, document loader |
| Domain Layer | Define business contracts and invariants | IndustrialResponse, Conversation, SafetyWarning, enums |

### How Data Flows Through the System

1. Request arrives from Streamlit UI or FastAPI endpoint.
2. Application services classify the query and optionally run retrieval.
3. RAG context is assembled from relevant industrial documents.
4. LLM generates a response with domain-aware prompting.
5. Post-processing computes confidence, risk level, and hallucination flags.
6. Domain DTOs enforce structured, auditable output before returning.

### Design Decisions

| Decision | Rationale |
|---|---|
| 4-layer clean architecture | Keeps dependencies directional, testable, and maintainable |
| Pydantic domain models | Provides strict contracts, validation, and serialization |
| RAG over fine-tuning | Supports fast document updates without model retraining |
| Confidence scoring | Improves trust and decision transparency in operations |
| Safety warning pipeline | Reduces risk from unsafe or ambiguous recommendations |
| FastAPI + Streamlit | Supports both integrations and rapid interactive demos |

### Key Tradeoffs

- In-memory vector store vs persistence: optimized for PoC speed, with configurable persistence path via VECTOR_STORE_PATH.
- Keyword domain classification vs ML classifier: deterministic and explainable behavior for safety-sensitive use cases.
- Single-process rate limiting vs distributed limiter: appropriate for current deployment scope, Redis-backed option for scale.

---

## Project Structure

```text
Langchain_website_chatbot/
├── src/
│   ├── config/                     # Centralized configuration (Pydantic Settings)
│   │   └── settings.py
│   ├── domain/                     # Domain models (framework-independent)
│   │   └── __init__.py
│   ├── application/                # Business logic layer
│   │   ├── chat_service.py
│   │   ├── rag_service.py
│   │   └── industrial_analyzer.py
│   ├── infrastructure/             # External integrations
│   │   ├── llm_client.py
│   │   ├── vector_store.py
│   │   └── document_loader.py
│   ├── interface/
│   │   ├── api/                    # FastAPI app, routes, middleware, DI
│   │   └── ui/                     # Streamlit UI
│   ├── utils/                      # Logging and helper utilities
│   ├── data/industrial_docs/       # Domain corpus
│   └── main.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── conftest.py
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

```bash
git clone https://github.com/your-username/Langchain_website_chatbot.git
cd Langchain_website_chatbot

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.template .env
# Add OPENAI_API_KEY to .env
```

### Run the Application

```bash
# Streamlit UI
streamlit run src/interface/ui/streamlit_app.py

# FastAPI API server
uvicorn src.interface.api.app:create_app --factory --reload

# Docker
docker-compose up --build
```

---

## API Reference

### POST /api/v1/chat

Processes a user message through retrieval, generation, and industrial post-analysis.

Example request:

```json
{
  "message": "How do I implement PID control in a Siemens PLC?",
  "use_rag": true,
  "domain_hint": "PLC Programming"
}
```

Example response:

```json
{
  "conversation_id": "uuid-...",
  "response": {
    "answer": "To implement PID control in a Siemens PLC using TIA Portal...",
    "confidence": "high",
    "confidence_score": 0.87,
    "risk_level": "low",
    "domain": "PLC Programming",
    "sources": [
      {
        "document": "plc_programming_guide",
        "relevance_score": 0.89,
        "excerpt": "PID control requires Kp, Ki, Kd tuning..."
      }
    ],
    "safety_warnings": [],
    "hallucination_flags": [],
    "model_used": "gpt-3.5-turbo",
    "response_time_ms": 1250.5
  },
  "processing_time_ms": 1300.0
}
```

### GET /api/v1/health

Health endpoint for readiness and monitoring checks.

### POST /api/v1/rag/initialize

Initializes or rebuilds the document retrieval pipeline.

### GET /api/v1/topics

Returns supported industrial automation topic domains.

---

## Industrial AI Features

### Supported Domains

| Domain | Typical Use Cases |
|---|---|
| PLC Programming | Ladder logic, Structured Text, IEC 61131-3 troubleshooting |
| SCADA Systems | Alarm handling, historians, IEC 62443 cybersecurity topics |
| Building Automation | BACnet, KNX, HVAC control, energy optimization |
| Predictive Maintenance | Condition monitoring, vibration analysis, reliability workflows |
| Industrial IoT | OPC UA, MQTT, edge connectivity, digital twins |
| Manufacturing Execution | MES design, ERP/SAP integration, production flow |
| Alarm Management | ISA-18.2 alignment and alarm rationalization |
| Energy Management | ISO 50001, load shedding, demand response |

### Response Quality Pipeline

```text
User Query
  -> Domain Classification
  -> RAG Retrieval (ChromaDB similarity + scores)
  -> Augmented Prompt Construction
  -> LLM Generation (OpenAI)
  -> Post-Processing
      - Confidence Scoring
      - Risk Assessment
      - Safety Warning Detection
      - Hallucination Signal Checks
  -> Structured IndustrialResponse
```

---

## Testing

Comprehensive testing is organized by layer to support fast feedback and reliable CI behavior.

| Layer | Test Type | Target | Tooling |
|---|---|---|---|
| Domain Models | Unit | 95%+ | pytest, Pydantic validation |
| IndustrialAnalyzer | Unit | 90%+ | pytest |
| LLM Client | Unit (mocked) | 85%+ | pytest, unittest.mock |
| ChatService | Unit (mocked) | 85%+ | pytest, unittest.mock |
| RAG Pipeline | Integration | 80%+ | pytest, real docs |
| API Endpoints | Integration | 85%+ | FastAPI TestClient |
| Helpers | Unit | 95%+ | pytest |

### CI Pipeline Scope

- Lint and formatting: flake8, black, isort, mypy
- Test matrix: pytest across Python 3.10, 3.11, 3.12
- Coverage gate: fail under 70%
- Security checks: safety and bandit

Run locally:

```bash
pip install -r requirements-dev.txt
pytest --cov=src --cov-report=term-missing
pytest tests/unit/ -v
pytest tests/integration/ -v
```

---

## Deployment

### Docker (Recommended)

```bash
docker-compose up --build
# API docs: http://localhost:8000/docs
# UI:       http://localhost:8501
```

### Cloud and Infrastructure Targets

| Platform | Typical Strategy |
|---|---|
| AWS | ECS Fargate + ALB, S3-backed document storage |
| Azure | Container Apps + Blob Storage |
| On-Premises | Docker Compose, optional air-gapped deployment |

### Production Checklist

- [ ] Set ENVIRONMENT=production
- [ ] Set DEBUG=false
- [ ] Configure LOG_FORMAT=json
- [ ] Set ALLOWED_API_KEYS
- [ ] Restrict CORS_ORIGINS
- [ ] Set VECTOR_STORE_PATH for persistence
- [ ] Add reverse proxy with TLS
- [ ] Define container resource limits
- [ ] Enable centralized log aggregation

---

## Technologies

| Category | Technology | Purpose |
|---|---|---|
| LLM | OpenAI GPT-3.5/4 via LangChain | Response generation |
| Retrieval | ChromaDB + OpenAI Embeddings | Knowledge grounding |
| Backend | FastAPI + Pydantic | API contracts and service layer |
| Frontend | Streamlit | Interactive interface |
| Testing | pytest + coverage | Quality assurance |
| CI/CD | GitHub Actions | Continuous integration |
| Containerization | Docker (multi-stage) | Reproducible deployment |
| Observability | Python logging + JSON formatter | Monitoring and diagnostics |
| Configuration | Pydantic Settings + .env | Environment-driven configuration |

---

## Limitations

- Single-user session assumptions without shared state backend
- English-focused terminology coverage today
- OpenAI dependency for generation path
- In-memory vector-store default for local PoC usage
- No automated RAG quality benchmark suite yet

---

## Roadmap

- [ ] Local LLM support via Ollama or vLLM for air-gapped deployments
- [ ] Multi-language support for German and Turkish industrial terminology
- [ ] Advanced PDF ingestion for Siemens/ABB/Schneider manuals
- [ ] RAGAS-based evaluation and continuous quality scoring
- [ ] Enterprise authentication (LDAP/SSO)
- [ ] Persistent production vector backends (for example pgvector)
- [ ] Streaming responses via SSE/WebSocket patterns
- [ ] End-to-end audit logging for compliance workflows

---

## License

Released under the MIT License. See [LICENSE](LICENSE).

---

## Portfolio Highlights

- Designed and implemented a production-style industrial AI assistant using RAG and clean architecture.
- Built domain-specific post-processing for safety, confidence, and hallucination-risk signals.
- Delivered automated testing strategy with unit and integration coverage plus CI/CD quality gates.
- Containerized deployment path with structured logging and environment-driven operations.
- Demonstrated practical AI engineering for high-trust industrial decision support.
