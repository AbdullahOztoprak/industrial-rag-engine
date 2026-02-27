# Industrial AI Knowledge Assistant

> A production-grade, RAG-enhanced LLM system for industrial automation knowledge management. Designed as a Proof-of-Concept for an industrial AI knowledge platform targeting PLC programming, SCADA systems, building automation, and predictive maintenance domains.

[![CI](https://github.com/your-username/Langchain_website_chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/Langchain_website_chatbot/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## System Overview

This system combines **Retrieval-Augmented Generation (RAG)** with **domain-specific prompt engineering** to deliver accurate, sourced, and safety-aware responses for industrial automation professionals. Unlike generic chatbots, every response includes:

- **Confidence scoring** — quantified reliability indicator
- **Source attribution** — traceable citations from indexed documentation
- **Safety warnings** — automatic detection of safety-critical topics (IEC 61508, OSHA)
- **Hallucination mitigation** — pattern-based detection of unreliable content
- **Structured analysis** — Problem → Root Cause → Solution → Risk Assessment format

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Interface Layer                          │
│  ┌─────────────────────┐    ┌──────────────────────────────┐   │
│  │   Streamlit UI       │    │   FastAPI REST API           │   │
│  │   (port 8501)        │    │   (port 8000)                │   │
│  └──────────┬──────────┘    └──────────────┬───────────────┘   │
│             │                               │                   │
├─────────────┴───────────────────────────────┴───────────────────┤
│                      Application Layer                          │
│  ┌──────────────────┐ ┌─────────────┐ ┌─────────────────────┐  │
│  │   ChatService     │ │ RAG Service │ │ IndustrialAnalyzer  │  │
│  │   (orchestrator)  │ │ (retrieval) │ │ (domain analysis)   │  │
│  └──────────────────┘ └─────────────┘ └─────────────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                     Infrastructure Layer                        │
│  ┌──────────────┐ ┌───────────────┐ ┌────────────────────────┐ │
│  │  LLM Client   │ │ Vector Store  │ │  Document Loader       │ │
│  │  (OpenAI)     │ │ (ChromaDB)    │ │  (TXT/PDF ingestion)   │ │
│  └──────────────┘ └───────────────┘ └────────────────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                        Domain Layer                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  IndustrialResponse │ ChatMessage │ Conversation          │  │
│  │  SourceAttribution  │ SafetyWarning │ ConfidenceLevel     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                   Cross-Cutting Concerns                        │
│  Config Management │ Structured Logging │ Rate Limiting         │
│  Input Sanitization │ Error Handling │ Health Monitoring         │
└─────────────────────────────────────────────────────────────────┘
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Clean Architecture (4-layer)** | Separation of concerns enables independent testing, easy swapping of LLM providers, and clear dependency direction |
| **Pydantic domain models** | Type-safe data contracts with validation, serialization, and OpenAPI schema generation |
| **RAG over fine-tuning** | Industrial documentation changes frequently; RAG allows real-time knowledge updates without retraining |
| **Confidence scoring** | Industrial environments demand auditability — users must know how reliable an answer is |
| **Safety warning system** | Automatic detection of safety-critical topics prevents dangerous misinformation |
| **FastAPI + Streamlit dual interface** | API-first design for system integration, Streamlit for rapid prototyping and demos |

### Key Tradeoffs

- **In-memory vector store** vs. persistent: Chose in-memory for PoC simplicity; persistence is configurable via `VECTOR_STORE_PATH`
- **Keyword-based domain classification** vs. ML classifier: Keyword matching is deterministic and explainable, preferred in safety-critical contexts
- **Single-process rate limiting** vs. Redis-backed: Adequate for PoC; production would use Redis/slowapi

---

## Project Structure

```
Langchain_website_chatbot/
├── src/
│   ├── config/                    # Centralized configuration (Pydantic Settings)
│   │   ├── __init__.py
│   │   └── settings.py           # Validated env-based config
│   ├── domain/                    # Domain models (framework-independent)
│   │   └── __init__.py           # Pydantic models: IndustrialResponse, ChatMessage, etc.
│   ├── application/               # Business logic layer
│   │   ├── chat_service.py       # Main orchestrator (LLM + RAG + Analysis)
│   │   ├── rag_service.py        # RAG pipeline management
│   │   └── industrial_analyzer.py # Domain classification, confidence, safety
│   ├── infrastructure/            # External service integrations
│   │   ├── llm_client.py         # LangChain/OpenAI abstraction
│   │   ├── vector_store.py       # ChromaDB vector storage
│   │   └── document_loader.py    # TXT/PDF document ingestion
│   ├── interface/                 # Presentation layer
│   │   ├── api/                  # FastAPI REST API
│   │   │   ├── app.py           # Application factory
│   │   │   ├── routes.py        # API endpoints
│   │   │   ├── middleware.py    # Rate limiting, logging
│   │   │   └── dependencies.py  # Dependency injection
│   │   └── ui/
│   │       └── streamlit_app.py  # Streamlit frontend
│   ├── utils/                     # Cross-cutting utilities
│   │   ├── helpers.py            # Input sanitization, formatting
│   │   └── logging_config.py    # Structured JSON/text logging
│   ├── data/
│   │   └── industrial_docs/      # Industrial documentation corpus
│   │       ├── plc_programming_guide.txt
│   │       └── building_automation_systems.txt
│   └── main.py                   # Application entry point
├── tests/
│   ├── conftest.py               # Shared fixtures and test config
│   ├── unit/                     # Unit tests (mocked dependencies)
│   │   ├── test_domain_models.py
│   │   ├── test_industrial_analyzer.py
│   │   ├── test_llm_client.py
│   │   ├── test_chat_service.py
│   │   └── test_helpers.py
│   ├── integration/              # Integration tests
│   │   ├── test_api_endpoints.py
│   │   └── test_rag_pipeline.py
│   └── fixtures/                 # Test data
│       └── sample_industrial_doc.txt
├── .github/workflows/ci.yml     # GitHub Actions CI pipeline
├── Dockerfile                    # Multi-stage production build
├── docker-compose.yml            # Service orchestration
├── pyproject.toml                # Tool configuration (black, mypy, pytest)
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── .env.template                 # Environment variable reference
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Installation

```bash
git clone https://github.com/your-username/Langchain_website_chatbot.git
cd Langchain_website_chatbot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env and add your OPENAI_API_KEY
```

### Run the Application

```bash
# Option 1: Streamlit UI
streamlit run src/interface/ui/streamlit_app.py

# Option 2: FastAPI Server
uvicorn src.interface.api.app:create_app --factory --reload

# Option 3: Docker
docker-compose up --build
```

### Run Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests with coverage
pytest --cov=src --cov-report=term-missing

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v
```

---

## API Reference

### `POST /api/v1/chat`

Process a message through the industrial AI pipeline.

**Request:**
```json
{
  "message": "How do I implement PID control in a Siemens PLC?",
  "use_rag": true,
  "domain_hint": "PLC Programming"
}
```

**Response:**
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

### `GET /api/v1/health`

Health check for monitoring and load balancers.

### `POST /api/v1/rag/initialize`

Initialize or reinitialize the RAG pipeline with industrial documentation.

### `GET /api/v1/topics`

List supported industrial automation domains.

---

## Industrial AI Features

### Supported Domains

| Domain | Use Cases |
|--------|-----------|
| **PLC Programming** | Ladder Logic, Structured Text, IEC 61131-3, troubleshooting |
| **SCADA Systems** | Alarm management, historian, cybersecurity (IEC 62443) |
| **Building Automation** | BACnet, KNX, HVAC control, energy optimization |
| **Predictive Maintenance** | Vibration analysis, condition monitoring, reliability |
| **Industrial IoT** | OPC UA, MQTT, edge computing, digital twins |
| **Manufacturing Execution** | MES design, SAP integration, production planning |
| **Alarm Management** | ISA-18.2 compliance, alarm rationalization |
| **Energy Management** | ISO 50001, load shedding, demand response |

### Response Quality Pipeline

```
User Query
    ↓
[Domain Classification] → keyword-based, deterministic
    ↓
[RAG Retrieval] → ChromaDB similarity search with scores
    ↓
[Augmented Prompt] → domain-specific system prompt + context
    ↓
[LLM Generation] → OpenAI with temperature control
    ↓
[Post-Processing]
  ├── Confidence Scoring (source quality + hedging + length analysis)
  ├── Risk Assessment (safety keyword detection)
  ├── Safety Warnings (IEC/ISO/OSHA standard references)
  └── Hallucination Detection (regex pattern matching)
    ↓
[Structured IndustrialResponse]
```

---

## Testing Strategy

| Layer | Test Type | Coverage Target | Tools |
|-------|-----------|----------------|-------|
| Domain Models | Unit | 95%+ | pytest, Pydantic validation |
| IndustrialAnalyzer | Unit | 90%+ | pytest |
| LLM Client | Unit (mocked) | 85%+ | pytest, unittest.mock |
| ChatService | Unit (mocked) | 85%+ | pytest, unittest.mock |
| RAG Pipeline | Integration | 80%+ | pytest, real documents |
| API Endpoints | Integration | 85%+ | FastAPI TestClient |
| Helpers | Unit | 95%+ | pytest |

### CI Pipeline

- **Lint**: flake8, black, isort, mypy
- **Test**: pytest across Python 3.10/3.11/3.12
- **Coverage**: fail threshold at 70%
- **Security**: safety (dependency scan), bandit (code scan)

---

## Deployment

### Docker (Recommended)

```bash
# Build and run both services
docker-compose up --build

# API: http://localhost:8000/docs
# UI:  http://localhost:8501
```

### Cloud Deployment

| Platform | Strategy |
|----------|----------|
| **AWS** | ECS Fargate + ALB, S3 for document storage |
| **Azure** | Container Apps, Blob Storage for docs |
| **On-Premises** | Docker Compose on industrial server, air-gapped option |

### Production Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Set `DEBUG=false`
- [ ] Configure `LOG_FORMAT=json` for ELK/Grafana
- [ ] Set `ALLOWED_API_KEYS` for API authentication
- [ ] Configure `CORS_ORIGINS` to specific domains
- [ ] Enable `VECTOR_STORE_PATH` for persistent embeddings
- [ ] Set up reverse proxy (Nginx/Traefik) with TLS
- [ ] Configure container resource limits
- [ ] Set up log aggregation pipeline

---

## Limitations

- **Single-user sessions**: No persistent session store (Redis needed for multi-instance)
- **English only**: NLP pipeline optimized for English industrial terminology
- **OpenAI dependency**: Requires OpenAI API; local model support planned
- **Vector store**: In-memory by default; production needs persistent storage
- **Evaluation**: No automated answer quality benchmark yet (planned: RAGAS framework)

## Roadmap

- [ ] **Local LLM support** — Ollama/vLLM integration for air-gapped environments
- [ ] **Multi-language** — German and Turkish industrial terminology support
- [ ] **PDF ingestion** — Siemens/ABB/Schneider manuals with table extraction
- [ ] **Evaluation framework** — RAGAS-based automated quality scoring
- [ ] **LDAP/SSO** — Enterprise authentication integration
- [ ] **Persistent vector store** — PostgreSQL + pgvector for production
- [ ] **Streaming responses** — Server-Sent Events for real-time UI updates
- [ ] **Audit logging** — Complete query/response audit trail for compliance

---

## Technologies

| Category | Technology | Purpose |
|----------|-----------|---------|
| **LLM** | OpenAI GPT-3.5/4 via LangChain | Text generation |
| **RAG** | ChromaDB + OpenAI Embeddings | Document retrieval |
| **Backend** | FastAPI + Pydantic | REST API |
| **Frontend** | Streamlit | Interactive UI |
| **Testing** | pytest + coverage | Test automation |
| **CI/CD** | GitHub Actions | Continuous integration |
| **Container** | Docker (multi-stage) | Deployment |
| **Logging** | Python logging + JSON formatter | Observability |
| **Config** | Pydantic Settings + .env | Environment management |

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## CV & Portfolio Bullets

### Resume Bullets

- **Designed and built a production-grade Industrial AI Knowledge Assistant** using RAG architecture (LangChain + ChromaDB + OpenAI), achieving structured response generation with confidence scoring, source attribution, and safety warning systems for industrial automation domains
- **Implemented clean architecture (4-layer separation)** with domain models (Pydantic), application services, infrastructure abstractions, and dual interface layer (FastAPI REST API + Streamlit UI), following enterprise software engineering practices
- **Developed comprehensive test automation** (50+ test cases) covering unit, integration, and API endpoint testing with mocked LLM calls, achieving 70%+ code coverage with CI/CD pipeline (GitHub Actions, flake8, black, mypy)
- **Built a domain-specific NLP pipeline** for 8 industrial domains (PLC, SCADA, BAS, predictive maintenance) with automatic domain classification, hallucination detection, and IEC/ISO safety standard referencing
- **Containerized for production deployment** using multi-stage Docker builds with health checks, rate limiting, structured JSON logging, and environment-based configuration management

### LinkedIn Project Description

**Industrial AI Knowledge Assistant** — A RAG-enhanced LLM system for industrial automation knowledge management. Built with clean architecture principles using Python, LangChain, FastAPI, and ChromaDB. Features domain-specific analysis for PLC programming, SCADA systems, and building automation with confidence scoring, source attribution, and automated safety warnings. Includes 50+ automated tests, CI/CD pipeline, Docker deployment, and structured logging. Demonstrates production-grade AI engineering for industrial environments.

### Elevator Pitch (30 seconds)

"I built an Industrial AI Knowledge Assistant that goes beyond a simple chatbot. It uses Retrieval-Augmented Generation to answer questions about PLC programming, SCADA systems, and building automation by searching through indexed technical documentation. What makes it different is that every answer comes with a confidence score, cited sources, and automatic safety warnings when the topic involves hazardous operations. The system is built with clean architecture, has 50+ automated tests, CI/CD, and is containerized for deployment. It's essentially a PoC for how AI can support industrial engineers in the field."

### Deep Technical Interview Explanation

"The system follows a 4-layer clean architecture: domain models at the core using Pydantic for type safety, an application layer with ChatService as the orchestrator, an infrastructure layer that abstracts OpenAI and ChromaDB, and an interface layer with both FastAPI and Streamlit.

When a query comes in, it goes through a pipeline: first, domain classification using keyword scoring across 8 industrial domains. Then, the RAG service retrieves relevant chunks from indexed documentation using ChromaDB similarity search. The augmented prompt combines retrieved context with a domain-specific system prompt that enforces structured reasoning — Problem, Root Cause, Solution format.

After LLM generation, the IndustrialAnalyzer post-processes the response: it computes a confidence score based on source quality, response length, and hedging language detection. It also scans for safety-critical keywords and generates warnings referencing specific IEC/ISO standards. There's a hallucination detection layer using regex patterns for overconfident claims and unverifiable source references.

For testing, I use pytest with mocked LLM calls — the infrastructure layer is fully mockable thanks to dependency injection. Integration tests cover the full RAG pipeline with real documents and API endpoints using FastAPI's TestClient. CI runs on GitHub Actions with flake8, black, mypy, and security scanning."
